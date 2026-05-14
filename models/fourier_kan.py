"""Fourier-KAN: replaces B-spline edges with Fourier-series edges.

Each edge: f(x) = sum_k a_k cos(k*x) + b_k sin(k*x).
Width chosen so total params ~ 3.6K.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


class FourierKANLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int, grid_size: int = 5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.coeff = nn.Parameter(
            torch.randn(2, out_features, in_features, grid_size)
            / (math.sqrt(in_features) * grid_size)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, in_features] -> embed into Fourier modes
        k = torch.arange(1, self.grid_size + 1, device=x.device, dtype=x.dtype)
        xk = x.unsqueeze(-1) * k  # [B, in, grid]
        c, s = torch.cos(xk), torch.sin(xk)
        # einsum: (B, in, grid), (out, in, grid) -> (B, out)
        out = torch.einsum("bid,oid->bo", c, self.coeff[0]) + torch.einsum(
            "bid,oid->bo", s, self.coeff[1]
        )
        return out


class FourierKAN(nn.Module):
    def __init__(self, d_in: int, d_hidden: int = 8, n_classes: int = 2, grid_size: int = 5):
        super().__init__()
        self.l1 = FourierKANLayer(d_in, d_hidden, grid_size)
        self.l2 = FourierKANLayer(d_hidden, n_classes, grid_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.l2(torch.tanh(self.l1(x)))