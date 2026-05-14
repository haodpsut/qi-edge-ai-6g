"""Compact Efficient-KAN implementation (Liu et al. 2024 style).

Each edge carries a learnable univariate function = base(x) + sum_i c_i * B_i(x)
where B_i are B-spline basis functions on a uniform grid.
Width [d_in, h, n_classes] chosen so total params ~ 3.6K.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class KANLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        grid_size: int = 5,
        spline_order: int = 3,
        scale_base: float = 1.0,
        scale_spline: float = 1.0,
        grid_range: tuple[float, float] = (-1.0, 1.0),
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order

        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (
            torch.arange(-spline_order, grid_size + spline_order + 1) * h
            + grid_range[0]
        )
        self.register_buffer("grid", grid.expand(in_features, -1).contiguous())

        self.base_weight = nn.Parameter(torch.empty(out_features, in_features))
        self.spline_weight = nn.Parameter(
            torch.empty(out_features, in_features, grid_size + spline_order)
        )
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)
        with torch.no_grad():
            noise = (torch.rand_like(self.spline_weight) - 0.5) * 0.1
            self.spline_weight.copy_(noise * self.scale_spline)

    def b_splines(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, in_features] -> [B, in_features, grid_size+spline_order]
        grid = self.grid  # [in_features, grid_size+2*spline_order+1]
        x = x.unsqueeze(-1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            left = (x - grid[:, : -(k + 1)]) / (grid[:, k:-1] - grid[:, : -(k + 1)])
            right = (grid[:, k + 1 :] - x) / (grid[:, k + 1 :] - grid[:, 1:(-k)])
            bases = left * bases[..., :-1] + right * bases[..., 1:]
        return bases  # [B, in_features, grid_size+spline_order]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = F.silu(x) @ self.base_weight.t()
        bs = self.b_splines(x).reshape(x.size(0), -1)  # [B, in*(g+k)]
        spline = bs @ self.spline_weight.reshape(self.out_features, -1).t()
        return base + spline


class EfficientKAN(nn.Module):
    def __init__(self, d_in: int, d_hidden: int = 9, n_classes: int = 2,
                 grid_size: int = 5, spline_order: int = 3):
        super().__init__()
        self.l1 = KANLinear(d_in, d_hidden, grid_size, spline_order)
        self.l2 = KANLinear(d_hidden, n_classes, grid_size, spline_order)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.l2(self.l1(x))