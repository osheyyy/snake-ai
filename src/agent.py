from __future__ import annotations

import random
from collections import deque
from pathlib import Path

import torch

from model import LinearQNet, QTrainer


class Agent:
    """DQN agent for Snake.

    The agent stores experience in replay memory, trains on the latest step
    immediately, and trains again from random past experiences after each game.
    """

    def __init__(
        self,
        state_mode: str = "simple",
        width: int = 20,
        height: int = 20,
        device: str = "auto",
        lr: float = 0.001,
        batch_size: int = 1000,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: float = 0.001,
    ) -> None:
        self.state_mode = state_mode
        self.width = width
        self.height = height
        self.batch_size = batch_size
        
        self.n_games = 0
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.epsilon = epsilon_start
        self.gamma = 0.9
        
        # Experience replay memory size is limited to 100,000 transitions.
        self.memory = deque(maxlen=100_000)

        # Device selection:
        # - "auto": Use CUDA if a compatible GPU is available, else CPU.
        # - "cuda": Explicitly request GPU training (falls back to CPU with warning if unavailable).
        # - "cpu": Force CPU inference/training (standard for play.py and laptops).
        # 
        # Why Training uses GPU but Play uses CPU:
        # 1. Training uses GPU: Training requires running thousands of parallel calculations in large
        #    batches (e.g. batch_size = 1000), backpropagating errors, and updating weights. GPUs are
        #    highly parallel architectures designed to accelerate these massive tensor operations.
        # 2. Play uses CPU: Running inference to decide the next move requires only a single forward
        #    pass on a single state vector (1x11 or 1x400). A CPU can compute this single prediction
        #    in a fraction of a millisecond, making GPU acceleration entirely unnecessary for playback.
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif device == "cuda":
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                print("WARNING: CUDA was requested but is not available. Falling back to CPU.")
                self.device = torch.device("cpu")
        else:
            self.device = torch.device("cpu")

        # input_size depends on state_mode:
        # - "simple": 11 features (handcrafted localized checks)
        # - "grid": width * height elements (full-board vision representation)
        if self.state_mode == "simple":
            self.input_size = 11
        elif self.state_mode == "grid":
            self.input_size = self.width * self.height
        else:
            raise ValueError(f"Unknown state mode: {self.state_mode}")

        # Instantiate our upgraded MLP model structure: input -> 256 -> 256 -> 128 -> 3
        self.model = LinearQNet(self.input_size, 256, 256, 128, 3).to(self.device)
        self.trainer = QTrainer(self.model, lr=lr, gamma=self.gamma, device=self.device)

    def get_state(self, game) -> list[int] | list[float]:
        return game.get_state()

    def remember(self, state, action, reward, next_state, done) -> None:
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self) -> None:
        if len(self.memory) > self.batch_size:
            mini_sample = random.sample(self.memory, self.batch_size)
        else:
            mini_sample = self.memory

        if not mini_sample:
            return

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done) -> None:
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state: list[int] | list[float], explore: bool = True) -> int:
        """Choose an action with epsilon-greedy exploration.

        During training, epsilon starts high so the snake tries many random
        moves. As games pass, epsilon gets smaller and the neural network is
        trusted more often. During play, explore=False makes the agent greedy.
        """
        self.epsilon = max(self.epsilon_end, self.epsilon_start - self.epsilon_decay * self.n_games)

        if explore and random.random() < self.epsilon:
            return random.randint(0, 2)

        state_tensor = torch.tensor(state, dtype=torch.float, device=self.device).unsqueeze(0)
        was_training = self.model.training
        self.model.eval()
        with torch.no_grad():
            prediction = self.model(state_tensor)
        self.model.train(was_training)
        return int(torch.argmax(prediction).item())

    def get_actions(self, states: list[list[int] | list[float]], explore: bool = True) -> list[int]:
        """Choose actions for a batch of states concurrently.

        This optimizes performance by grouping all states requiring neural network exploitation
        into a single PyTorch tensor. The model executes them in a single batch forward pass,
        which is highly efficient for hardware acceleration (like CUDA on Colab). Epsilon-greedy
        decisions are evaluated independently for each state.
        """
        self.epsilon = max(self.epsilon_end, self.epsilon_start - self.epsilon_decay * self.n_games)

        actions = [0] * len(states)
        exploit_indices = []
        exploit_states = []

        for idx, state in enumerate(states):
            if explore and random.random() < self.epsilon:
                actions[idx] = random.randint(0, 2)
            else:
                exploit_indices.append(idx)
                exploit_states.append(state)

        # Batch model inference for all environments requiring exploitation
        if exploit_states:
            state_tensor = torch.tensor(exploit_states, dtype=torch.float, device=self.device)
            was_training = self.model.training
            self.model.eval()
            with torch.no_grad():
                predictions = self.model(state_tensor)
            self.model.train(was_training)

            predicted_actions = torch.argmax(predictions, dim=1).cpu().tolist()
            for idx, action in zip(exploit_indices, predicted_actions):
                actions[idx] = int(action)

        return actions

    def save_model(self, file_path: str | Path) -> None:
        """Save the network weights along with diagnostic training metadata."""
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "state_dict": self.model.state_dict(),
            "metadata": {
                "state_mode": self.state_mode,
                "width": self.width,
                "height": self.height,
                "input_size": self.input_size,
                "architecture": f"{self.input_size} -> 256 -> 256 -> 128 -> 3",
            }
        }
        torch.save(checkpoint, file_path)

    def load_model(self, file_path: str | Path) -> None:
        """Load model state dict and validate metadata configuration compatibility."""
        file_path = Path(file_path)
        checkpoint = torch.load(file_path, map_location=self.device)

        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
            metadata = checkpoint.get("metadata", {})

            # Check and validate state mode
            meta_state_mode = metadata.get("state_mode")
            if meta_state_mode is not None and meta_state_mode != self.state_mode:
                raise ValueError(
                    f"State mode mismatch: checkpoint used '{meta_state_mode}', "
                    f"but current agent is configured for '{self.state_mode}'"
                )

            # Check and validate board dimensions for grid mode
            if self.state_mode == "grid":
                meta_width = metadata.get("width")
                meta_height = metadata.get("height")
                if meta_width is not None and meta_width != self.width:
                    raise ValueError(
                        f"Board width mismatch: checkpoint has width {meta_width}, "
                        f"but current game has width {self.width}"
                    )
                if meta_height is not None and meta_height != self.height:
                    raise ValueError(
                        f"Board height mismatch: checkpoint has height {meta_height}, "
                        f"but current game has height {self.height}"
                    )

            # Check and validate input layer sizes
            meta_input_size = metadata.get("input_size")
            if meta_input_size is not None and meta_input_size != self.input_size:
                raise ValueError(
                    f"Input size mismatch: checkpoint has input_size {meta_input_size}, "
                    f"but current agent expects input_size {self.input_size}"
                )
        else:
            # Fall back gracefully to loading raw weights if no metadata dictionary is present
            state_dict = checkpoint

        self.model.load_state_dict(state_dict)
        self.model.eval()
