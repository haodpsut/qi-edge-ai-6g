"""Single train+eval run.

Outputs: results/{model}_{seed}.json with accuracy, F1, params, FLOPs, latency.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, precision_score, recall_score
from torch.optim import AdamW

from data import load_nf_bot_iot_v2, make_loaders
from models import MODELS
from utils import seed_everything


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def measure_latency_cpu(model: nn.Module, d_in: int, n_iter: int = 100) -> float:
    """Inference latency in ms per sample, batch=1, CPU (edge proxy)."""
    model = model.cpu().eval()
    x = torch.randn(1, d_in)
    with torch.no_grad():
        for _ in range(10):
            model(x)
        t0 = time.perf_counter()
        for _ in range(n_iter):
            model(x)
        t1 = time.perf_counter()
    return (t1 - t0) / n_iter * 1000.0


def measure_flops(model: nn.Module, d_in: int) -> float:
    """FLOPs per sample via thop. Returns float (M-FLOPs)."""
    try:
        from thop import profile

        model = model.cpu().eval()
        x = torch.randn(1, d_in)
        flops, _ = profile(model, inputs=(x,), verbose=False)
        return float(flops) / 1e6
    except Exception:
        return float("nan")


def evaluate(model: nn.Module, loader, device: str) -> dict:
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            logits = model(x)
            ps.append(logits.argmax(dim=1).cpu().numpy())
            ys.append(y.numpy())
    y = np.concatenate(ys)
    p = np.concatenate(ps)
    return {
        "accuracy": float((p == y).mean()),
        "f1_macro": float(f1_score(y, p, average="macro")),
        "f1_binary": float(f1_score(y, p, average="binary")),
        "precision": float(precision_score(y, p, average="binary")),
        "recall": float(recall_score(y, p, average="binary")),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODELS))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--data-path", default="data/raw/NF-BoT-IoT-v2.csv")
    ap.add_argument("--n-samples", type=int, default=1_000_000)
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--quick", action="store_true",
                    help="Smoke test: 10k samples, 2 epochs")
    args = ap.parse_args()

    if args.quick:
        args.n_samples = 10_000
        args.epochs = 2

    seed_everything(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    X_tr, X_te, y_tr, y_te, d_in = load_nf_bot_iot_v2(
        args.data_path, n_samples=args.n_samples, seed=args.seed
    )
    tr_loader, te_loader = make_loaders(X_tr, X_te, y_tr, y_te, args.batch_size)

    cls = MODELS[args.model]
    model = cls(d_in=d_in, n_classes=2).to(device)
    n_params = count_params(model)
    opt = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    t_train_start = time.perf_counter()
    for ep in range(args.epochs):
        model.train()
        for x, y in tr_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
    t_train = time.perf_counter() - t_train_start

    metrics = evaluate(model, te_loader, device)
    flops_m = measure_flops(cls(d_in=d_in, n_classes=2), d_in)
    latency_ms = measure_latency_cpu(cls(d_in=d_in, n_classes=2), d_in)

    out = {
        "model": args.model,
        "seed": args.seed,
        "n_samples": args.n_samples,
        "epochs": args.epochs,
        "params": n_params,
        "flops_m_per_sample": flops_m,
        "latency_ms_per_sample_cpu": latency_ms,
        "train_time_s": t_train,
        **metrics,
    }
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.model}_{args.seed}.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()