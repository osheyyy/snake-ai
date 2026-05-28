from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim


class LinearQNet(nn.Module):
    """Small neural network that estimates Q-values for the three actions."""

    def __init__(self, input_size: int = 11, hidden_size: int = 256, output_size: int = 3) -> None:
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.linear1(x))
        return self.linear2(x)

    def save(self, file_path: str | Path) -> None:
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), file_path)


class QTrainer:
    """Runs one DQN optimization step.

    DQN learns by making the current Q-value prediction closer to a target:
        target = reward + gamma * best_future_q

    If the snake died, there is no future state, so the target is just reward.
    """

    def __init__(self, model: LinearQNet, lr: float, gamma: float, device: torch.device) -> None:
        self.model = model
        self.gamma = gamma
        self.device = device
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, done) -> None:
        state = torch.tensor(state, dtype=torch.float, device=self.device)
        next_state = torch.tensor(next_state, dtype=torch.float, device=self.device)
        action = torch.tensor(action, dtype=torch.long, device=self.device)
        reward = torch.tensor(reward, dtype=torch.float, device=self.device)

        if len(state.shape) == 1:
            state = state.unsqueeze(0)
            next_state = next_state.unsqueeze(0)
            action = action.unsqueeze(0)
            reward = reward.unsqueeze(0)
            done = (done,)

        prediction = self.model(state)
        target = prediction.clone().detach()

        # Build the Bellman targets without tracking gradients through next_state.
        with torch.no_grad():
            next_prediction = self.model(next_state)

        for index in range(len(done)):
            q_new = reward[index]
            if not done[index]:
                q_new = reward[index] + self.gamma * torch.max(next_prediction[index])

            target[index][action[index].item()] = q_new

        self.optimizer.zero_grad()
        loss = self.criterion(prediction, target)
        loss.backward()
        self.optimizer.step()
