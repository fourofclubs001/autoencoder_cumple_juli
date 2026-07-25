from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from .q_network import QNetwork
from .replay_buffer import ReplayBuffer

NEG_INF = -1e9


class DQNAgent:
    """DQN baseline per Section 3.1: TD learning with a periodically-updated
    target network, Huber loss, gradients clipped to max norm 1.0, epsilon-greedy
    over legal moves only."""

    def __init__(
        self,
        state_dim: int = 16,
        num_actions: int = 4,
        hidden_dim: int = 256,
        lr: float = 1e-4,
        gamma: float = 0.99,
        buffer_size: int = 1_000_000,
        batch_size: int = 256,
        tau: float = 5e-3,
        grad_clip_norm: float = 1.0,
        device: str | None = None,
    ):
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.num_actions = num_actions
        self.gamma = gamma
        self.batch_size = batch_size
        self.tau = tau
        self.grad_clip_norm = grad_clip_norm

        self.online = QNetwork(state_dim, num_actions, hidden_dim).to(self.device)
        self.target = QNetwork(state_dim, num_actions, hidden_dim).to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()

        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=lr)
        self.buffer = ReplayBuffer(buffer_size)

    def act(self, state: np.ndarray, legal_mask: np.ndarray, epsilon: float) -> int:
        legal_actions = np.flatnonzero(legal_mask)
        if np.random.random() < epsilon:
            return int(np.random.choice(legal_actions))

        with torch.no_grad():
            state_t = torch.as_tensor(
                state, dtype=torch.float32, device=self.device
            ).unsqueeze(0)
            q_values = self.online(state_t).squeeze(0).cpu().numpy()

        q_values = np.where(legal_mask, q_values, NEG_INF)
        return int(np.argmax(q_values))

    def remember(self, state, action, reward, next_state, done, next_legal_mask):
        self.buffer.push(state, action, reward, next_state, done, next_legal_mask)

    def learn(self) -> float | None:
        if len(self.buffer) < self.batch_size:
            return None

        batch = self.buffer.sample(self.batch_size)

        states = torch.as_tensor(
            np.array(batch.state), dtype=torch.float32, device=self.device
        )
        actions = torch.as_tensor(
            batch.action, dtype=torch.long, device=self.device
        ).unsqueeze(1)
        rewards = torch.as_tensor(
            batch.reward, dtype=torch.float32, device=self.device
        ).unsqueeze(1)
        next_states = torch.as_tensor(
            np.array(batch.next_state), dtype=torch.float32, device=self.device
        )
        dones = torch.as_tensor(
            batch.done, dtype=torch.float32, device=self.device
        ).unsqueeze(1)
        next_legal_masks = torch.as_tensor(
            np.array(batch.next_legal_mask), dtype=torch.bool, device=self.device
        )

        q_values = self.online(states).gather(1, actions)

        with torch.no_grad():
            next_q = self.target(next_states)
            next_q = next_q.masked_fill(~next_legal_masks, NEG_INF)
            max_next_q = next_q.max(dim=1, keepdim=True).values
            target = rewards + self.gamma * (1.0 - dones) * max_next_q

        loss = F.smooth_l1_loss(q_values, target)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online.parameters(), self.grad_clip_norm)
        self.optimizer.step()

        self._soft_update_target()
        return loss.item()

    def _soft_update_target(self) -> None:
        for target_param, online_param in zip(
            self.target.parameters(), self.online.parameters()
        ):
            target_param.data.copy_(
                self.tau * online_param.data + (1.0 - self.tau) * target_param.data
            )
