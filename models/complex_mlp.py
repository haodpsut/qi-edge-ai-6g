"""Complex-valued MLP — a quantum-inspired neural design.

Real input is mapped to complex domain via learnable phase; intermediate layers
use complex linear maps; output magnitude feeds a real classifier head.
This pattern is the standard "deep complex network" archetype that quantum
analogs reduce to in the classical limit.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class ComplexLinear(nn.Module):
    """y = W x where W = Wr + j Wi, x = xr + j xi."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.wr = nn.Linear(in_features, out_features, bias=True)
        self.wi = nn.Linear(in_features, out_features, bias=True)
        nn.init.orthogonal_(self.wr.weight)
        nn.init.orthogonal_(self.wi.weight)

    def forward(self, xr: torch.Tensor, xi: torch.Tensor):
        yr = self.wr(xr) - self.wi(xi)
        yi = self.wr(xi) + self.wi(xr)
        return yr, yi


def cmod_relu(xr: torch.Tensor, xi: torch.Tensor, bias: torch.Tensor):
    """modReLU activation (Arjovsky et al. 2016)."""
    mag = torch.sqrt(xr * xr + xi * xi + 1e-6)
    scale = torch.clamp(mag + bias, min=0.0) / mag
    return xr * scale, xi * scale


class ComplexMLP(nn.Module):
    def __init__(self, d_in: int, d_hidden: int = 25, n_classes: int = 2):
        super().__init__()
        self.l1 = ComplexLinear(d_in, d_hidden)
        self.b1 = nn.Parameter(torch.zeros(d_hidden))
        self.l2 = ComplexLinear(d_hidden, d_hidden)
        self.b2 = nn.Parameter(torch.zeros(d_hidden))
        self.head = nn.Linear(d_hidden, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xr, xi = x, torch.zeros_like(x)
        xr, xi = self.l1(xr, xi)
        xr, xi = cmod_relu(xr, xi, self.b1)
        xr, xi = self.l2(xr, xi)
        xr, xi = cmod_relu(xr, xi, self.b2)
        mag = torch.sqrt(xr * xr + xi * xi + 1e-6)
        return self.head(mag)