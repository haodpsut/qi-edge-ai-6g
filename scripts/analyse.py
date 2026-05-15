"""Aggregate results/raw.csv into magazine-paper figures + Welch t-tests.

Outputs:
  results/analysis.txt   — full text summary with stats
  results/welch.csv      — pairwise Welch t-test matrix
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from scipy import stats


def main() -> None:
    df = pd.read_csv("results/raw.csv")
    out = Path("results")
    out.mkdir(exist_ok=True)

    lines: list[str] = []
    p = lines.append

    p("=== Per-model summary (mean over 5 seeds) ===")
    g = df.groupby("model").agg(
        params=("params", "first"),
        acc_mean=("accuracy", "mean"),
        acc_std=("accuracy", "std"),
        f1_mean=("f1_macro", "mean"),
        f1_std=("f1_macro", "std"),
        latency_ms=("latency_ms_per_sample_cpu", "mean"),
        latency_std=("latency_ms_per_sample_cpu", "std"),
        flops_m=("flops_m_per_sample", "first"),
        train_s=("train_time_s", "mean"),
    ).round(5)
    p(g.to_string())
    p("")

    p("=== Welch t-test on F1-macro (KAN vs each baseline) ===")
    kan = df[df.model == "kan"]["f1_macro"].values
    welch_rows: list[dict] = []
    for m in ["mlp", "cnn1d", "fkan", "cmlp"]:
        other = df[df.model == m]["f1_macro"].values
        t, pv = stats.ttest_ind(kan, other, equal_var=False)
        sig = "***" if pv < 0.001 else "**" if pv < 0.01 else "*" if pv < 0.05 else "n.s."
        p(f"  KAN vs {m:6s}: t={t:+.3f}  p={pv:.4f}  {sig}")
        welch_rows.append({"vs": m, "t": t, "p": pv, "sig": sig})
    pd.DataFrame(welch_rows).to_csv(out / "welch.csv", index=False)
    p("")

    p("=== Pareto frontier (F1 vs latency) ===")
    pareto = []
    pts = list(g[["latency_ms", "f1_mean"]].itertuples(index=True))
    for i, ri in enumerate(pts):
        dominated = False
        for j, rj in enumerate(pts):
            if i == j:
                continue
            if rj.latency_ms <= ri.latency_ms and rj.f1_mean >= ri.f1_mean \
               and (rj.latency_ms < ri.latency_ms or rj.f1_mean > ri.f1_mean):
                dominated = True
                break
        if not dominated:
            pareto.append(ri.Index)
    p(f"  Pareto-optimal models: {pareto}")
    p("")

    (out / "analysis.txt").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
