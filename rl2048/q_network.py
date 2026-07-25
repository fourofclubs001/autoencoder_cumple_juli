import torch
import torch.nn as nn


class QNetwork(nn.Module):
    """MLP over the flattened log2 board, matching Section 3.1: two 256-unit
    ReLU hidden layers, Qθ(s, a) = fθ(log2(board))."""

    def __init__(self, state_dim: int = 16, num_actions: int = 4, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
