"""SWaT (Secure Water Treatment) dataset loader for RQ7.

Source: /home/interestrate/workspace/anomaly-benchmark/data/SWAT/
  train.csv  — 495 000 rows × 52 cols (51 sensors + Normal/Attack label)
  test.csv   — 449 919 rows × 52 cols (same schema; has attack periods)

Strategy:
- Z-score each sensor using train statistics.
- Use test set only (attack periods live there).
- Split into K=20 equal time-chunks as pseudo-runs for loop DTW.
- Label: Normal/Attack column → 0 (normal) / 1 (attack).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

SWAT_DATA_DIR = Path(
    "/home/interestrate/workspace/anomaly-benchmark/data/SWAT"
)
LABEL_COL = "Normal/Attack"
N_PSEUDO_RUNS = 20


def load_swat_binary(
    data_dir: Path = SWAT_DATA_DIR,
    n_pseudo_runs: int = N_PSEUDO_RUNS,
) -> dict[str, np.ndarray]:
    """Return dict with keys X, mode_id, run_id, window_start_placeholder.

    X        : [T, C] float32  — full test signal, z-scored
    mode_id  : [T]    int32    — 0=normal, 1=attack
    run_id   : [T]    int32    — pseudo-run index 0..n_pseudo_runs-1
    """
    train = pd.read_csv(data_dir / "train.csv")
    test  = pd.read_csv(data_dir / "test.csv")

    sensor_cols = [c for c in train.columns if c != LABEL_COL]

    # Z-score using train statistics
    mu  = train[sensor_cols].mean().values.astype(np.float32)
    std = train[sensor_cols].std().values.astype(np.float32)
    std[std < 1e-8] = 1.0

    X = ((test[sensor_cols].values.astype(np.float32) - mu) / std)
    labels = test[LABEL_COL].values.astype(np.int32)

    T = len(X)
    chunk = T // n_pseudo_runs
    run_id = np.zeros(T, dtype=np.int32)
    for k in range(n_pseudo_runs):
        start = k * chunk
        end   = (k + 1) * chunk if k < n_pseudo_runs - 1 else T
        run_id[start:end] = k

    return {
        "X":       X,
        "mode_id": labels,
        "run_id":  run_id,
    }
