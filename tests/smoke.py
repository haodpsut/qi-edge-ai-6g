"""Local smoke test: instantiate each model + forward pass on random data.

Run: python tests/smoke.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from models import MODELS


def main() -> None:
    d_in = 43
    x = torch.randn(16, d_in)
    for name, cls in MODELS.items():
        m = cls(d_in=d_in, n_classes=2)
        n = sum(p.numel() for p in m.parameters() if p.requires_grad)
        y = m(x)
        assert y.shape == (16, 2), f"{name} bad shape: {y.shape}"
        print(f"{name:8s}  params={n:>6d}  out={tuple(y.shape)}")


if __name__ == "__main__":
    main()