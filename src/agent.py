from __future__ import annotations

import random
from collections import deque
from pathlib import Path

import torch

from model import LinearQNet, QTrainer


MAX_MEMORY = 100_000
BATCH_SIZE = 1_000
LR = 0.001


class Agent:
    """DQN agent for Snake.

    The agent stores experience in replay memory, trains on the latest step
    immediately, and trains again from random past experiences after each game.
    """

    def __init__(self) -> None:
        self.n_games = 0
        self.epsilon = 1.0
        self.epsilon_start = 1.0
        self.epsilon_end = 0.01
        self.epsilon_decay = 0.005
        self.gamma = 0.9
        self.memory = deque(maxlen=MAX_MEMORY)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = LinearQNet(11, 256, 3).to(self.device)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma, device=self.device)

    def get_state(self, game) -> list[int]:
        return game.get_state()

    def remember(self, state, action, reward, next_state, done) -> None:
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self) -> None:
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory

        if not mini_sample:
            return

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done) -> None:
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state: list[int], explore: bool = True) -> int:
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

    def save_model(self, file_path: str | Path) -> None:
        self.model.save(file_path)

    def load_model(self, file_path: str | Path) -> None:
        file_path = Path(file_path)
        state_dict = torch.load(file_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()
