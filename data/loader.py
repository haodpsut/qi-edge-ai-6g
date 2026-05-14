"""NF-BoT-IoT-v2 loader (Sarhan et al. NetFlow v2 format).

Binary classification: label == 0 -> benign, label == 1 -> attack.
Sub-samples to a configurable cap for fast benchmarking.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

# 43 NetFlow features (drop categorical IPs/ports + Attack/Label cols)
DROP_COLS = [
    "IPV4_SRC_ADDR", "IPV4_DST_ADDR",
    "L4_SRC_PORT", "L4_DST_PORT",
    "Attack",
]
LABEL_COL = "Label"


def _read_any(path: Path) -> pd.DataFrame:
    """Read CSV or Parquet, auto-detect by extension. If path doesn't exist,
    try the sibling with the other extension."""
    if path.exists():
        if path.suffix.lower() in (".parquet", ".pq"):
            return pd.read_parquet(path)
        return pd.read_csv(path)
    # auto-fallback
    for alt_suffix in (".parquet", ".csv"):
        alt = path.with_suffix(alt_suffix)
        if alt.exists():
            if alt_suffix == ".parquet":
                return pd.read_parquet(alt)
            return pd.read_csv(alt)
    raise FileNotFoundError(
        f"Dataset not found. Tried {path} and {path.with_suffix('.parquet')} / "
        f"{path.with_suffix('.csv')}. Run scripts/fetch_data.sh first."
    )


def load_nf_bot_iot_v2(
    csv_path: str | Path,
    n_samples: int | None = 1_000_000,
    test_size: float = 0.2,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """Return (X_train, X_test, y_train, y_test, n_features) as numpy arrays.

    Accepts CSV or Parquet (dhoogla/nfbotiotv2 Kaggle mirror ships parquet)."""
    df = _read_any(Path(csv_path))
    if n_samples and len(df) > n_samples:
        df = df.sample(n=n_samples, random_state=seed).reset_index(drop=True)

    y = df[LABEL_COL].astype(np.int64).values
    X = df.drop(columns=[c for c in DROP_COLS + [LABEL_COL] if c in df.columns])
    X = X.select_dtypes(include=[np.number]).astype(np.float32).values

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=seed
    )
    sc = StandardScaler().fit(X_tr)
    X_tr = sc.transform(X_tr).astype(np.float32)
    X_te = sc.transform(X_te).astype(np.float32)
    return X_tr, X_te, y_tr, y_te, X_tr.shape[1]


def make_loaders(
    X_tr, X_te, y_tr, y_te, batch_size: int = 512
) -> tuple[DataLoader, DataLoader]:
    tr = TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr))
    te = TensorDataset(torch.from_numpy(X_te), torch.from_numpy(y_te))
    return (
        DataLoader(tr, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True),
        DataLoader(te, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True),
    )