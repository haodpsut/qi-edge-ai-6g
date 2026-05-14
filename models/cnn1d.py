"""1D-CNN baseline. Treats flat NetFlow features as a 1D sequence.

Parameter budget ~3.6K via small channel counts.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class CNN1D(nn.Module):
    def __init__(self, d_in: int, n_classes: int = 2, ch: int = 22):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, ch, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(ch, ch * 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(8),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(ch * 2 * 8, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, d_in] -> [B, 1, d_in]
        x = x.unsqueeze(1)
        x = self.conv(x)
        return self.head(x)