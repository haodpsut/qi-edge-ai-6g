"""Plain MLP baseline. Width chosen to land near 3.6K params at d_in=43."""
from __future__ import annotations

import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, d_in: int, d_hidden: int = 50, n_classes: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, d_hidden // 2),
            nn.ReLU(),
            nn.Linear(d_hidden // 2, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)