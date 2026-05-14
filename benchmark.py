"""Aggregate results/*.json into a single CSV + stats summary.

Usage: python benchmark.py [--out results/summary.csv]
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--out", default="results/summary.csv")
    args = ap.parse_args()

    rows = []
    for p in sorted(Path(args.results_dir).glob("*.json")):
        if p.name == "summary.csv":
            continue
        try:
            rows.append(json.loads(p.read_text()))
        except Exception as e:
            print(f"skip {p}: {e}")

    if not rows:
        print("no result files found")
        return

    df = pd.DataFrame(rows)
    # per-model aggregation
    grp = df.groupby("model").agg(
        accuracy_mean=("accuracy", "mean"),
        accuracy_std=("accuracy", "std"),
        f1_macro_mean=("f1_macro", "mean"),
        f1_macro_std=("f1_macro", "std"),
        params=("params", "first"),
        flops_m=("flops_m_per_sample", "mean"),
        latency_ms=("latency_ms_per_sample_cpu", "mean"),
        train_time_s_mean=("train_time_s", "mean"),
        n_seeds=("seed", "count"),
    ).reset_index()
    print(grp.to_string(index=False))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    grp.to_csv(out, index=False)
    df.to_csv(out.with_name("raw.csv"), index=False)
    print(f"\nwrote {out} and {out.with_name('raw.csv')}")


if __name__ == "__main__":
    main()