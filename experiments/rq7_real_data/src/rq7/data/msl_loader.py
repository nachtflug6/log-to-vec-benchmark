"""MSL (Mars Science Laboratory) dataset loader for RQ7.

Source: /home/interestrate/workspace/anomaly-benchmark/data/SMAP_MSL/
  MSL_train.npy       — (58317, 55) float32, already z-scored
  MSL_test.npy        — (73729, 55) float32, already z-scored
  MSL_test_label.npy  — (73729,)   bool

Strategy:
- Data is already normalised by the anomaly-benchmark pipeline — use as-is.
- Use test set only (anomaly labels live there).
- Split into K=20 equal time-chunks as pseudo-runs for loop DTW.
- Label: test_label → 0 (normal) / 1 (anomaly).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

MSL_DATA_DIR = Path(
    "/home/interestrate/workspace/anomaly-benchmark/data/SMAP_MSL"
)
N_PSEUDO_RUNS = 20


def load_msl_binary(
    data_dir: Path = MSL_DATA_DIR,
    n_pseudo_runs: int = N_PSEUDO_RUNS,
) -> dict[str, np.ndarray]:
    """Return dict with keys X, mode_id, run_id.

    X        : [T, C] float32
    mode_id  : [T]    int32    — 0=normal, 1=anomaly
    run_id   : [T]    int32    — pseudo-run index 0..n_pseudo_runs-1
    """
    X      = np.load(data_dir / "MSL_test.npy").astype(np.float32)
    labels = np.load(data_dir / "MSL_test_label.npy").astype(np.int32)

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
