"""TEP (Tennessee Eastman Process, Reinartz 2021) dataset loader for RQ7.

Reference: Reinartz et al. (2021) "An Extended Tennessee Eastman Simulation
Dataset for Fault-Detection and Decision Support Systems."
Download: https://data.dtu.dk/articles/dataset/Tennessee_Eastman_Reference_Data.../13385936

Expected location after download:
  experiments/rq7_real_data/raw/tep/

The dataset provides 500 simulation runs per fault type (IDV0 = normal,
IDV1-IDV28 = faults), each as a separate CSV file.

Format (each CSV):
  - 52 numeric columns (41 measurement + 11 manipulated variables)
  - Column 'simulationRun' — run index within fault type
  - Column 'sample' — time step
  - Column 'faultNumber' — fault ID (0-28)
  - 500 rows per run × 960 time steps

Scenarios implemented here:
  tep_modes   — fault-free runs (IDV0) with 6 production modes
                (mode encoded in 'xmv(11)' / manipulated variable 11)
  tep_faults  — runs with faults IDV0-IDV8 + normal, labelled by fault type

NOTE: This loader is a stub until the TEP data is downloaded to the cluster.
      The download_tep() helper prints instructions.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

TEP_RAW_DIR = Path(__file__).resolve().parents[3] / "raw" / "tep"

# 52 process variables used as channels
TEP_FEATURE_COLS = [
    "xmeas(1)",  "xmeas(2)",  "xmeas(3)",  "xmeas(4)",  "xmeas(5)",
    "xmeas(6)",  "xmeas(7)",  "xmeas(8)",  "xmeas(9)",  "xmeas(10)",
    "xmeas(11)", "xmeas(12)", "xmeas(13)", "xmeas(14)", "xmeas(15)",
    "xmeas(16)", "xmeas(17)", "xmeas(18)", "xmeas(19)", "xmeas(20)",
    "xmeas(21)", "xmeas(22)", "xmeas(23)", "xmeas(24)", "xmeas(25)",
    "xmeas(26)", "xmeas(27)", "xmeas(28)", "xmeas(29)", "xmeas(30)",
    "xmeas(31)", "xmeas(32)", "xmeas(33)", "xmeas(34)", "xmeas(35)",
    "xmeas(36)", "xmeas(37)", "xmeas(38)", "xmeas(39)", "xmeas(40)",
    "xmeas(41)",
    "xmv(1)",  "xmv(2)",  "xmv(3)",  "xmv(4)",  "xmv(5)",
    "xmv(6)",  "xmv(7)",  "xmv(8)",  "xmv(9)",  "xmv(10)", "xmv(11)",
]

# Production mode signal: xmv(11) maps to modes 0-5 (6 G/H ratio setpoints)
MODE_SIGNAL_COL = "xmv(11)"


def _check_available(raw_dir: Path) -> bool:
    return raw_dir.exists() and any(raw_dir.glob("*.csv"))


def download_tep() -> None:
    print(
        "\nTEP (Reinartz 2021) data not found.\n"
        "Download from DTU Data Repository:\n"
        "  https://data.dtu.dk/articles/dataset/Tennessee_Eastman_Reference_Data.../13385936\n"
        f"Extract to: {TEP_RAW_DIR}\n"
        "Or on Alvis:\n"
        f"  ssh alvis1 'mkdir -p ~/log-to-vec-benchmark/experiments/rq7_real_data/raw/tep && "
        "wget <url> -O ~/tep.zip && unzip ~/tep.zip -d "
        "~/log-to-vec-benchmark/experiments/rq7_real_data/raw/tep/'\n"
    )


def _load_fault_csvs(raw_dir: Path, fault_ids: list[int]) -> dict:
    """Load one CSV per fault_id, return arrays X, mode_id, run_id."""
    import pandas as pd

    all_X, all_mode, all_run = [], [], []
    global_run = 0
    for fid in fault_ids:
        pattern = f"*IDV({fid})*.csv" if fid > 0 else "*IDV(0)*.csv"
        files = sorted(raw_dir.glob(pattern))
        if not files:
            # also try simple numbering
            files = sorted(raw_dir.glob(f"*d{fid:02d}*.csv"))
        for fp in files:
            df = pd.read_csv(fp)
            # handle varying column names
            feat_cols = [c for c in TEP_FEATURE_COLS if c in df.columns]
            if not feat_cols:
                # try alternate naming without parentheses
                feat_cols = [c for c in df.columns
                             if c.startswith("xmeas") or c.startswith("xmv")]
                feat_cols = feat_cols[:52]
            X = df[feat_cols].values.astype(np.float32)
            all_X.append(X)
            all_mode.append(np.full(len(X), fid, dtype=np.int32))
            all_run.append(np.full(len(X), global_run, dtype=np.int32))
            global_run += 1

    if not all_X:
        raise FileNotFoundError(
            f"No TEP CSV files found in {raw_dir} for fault_ids={fault_ids}"
        )

    X       = np.concatenate(all_X,    axis=0)
    mode_id = np.concatenate(all_mode, axis=0)
    run_id  = np.concatenate(all_run,  axis=0)

    # z-score each channel over all loaded data
    mu  = X.mean(axis=0, keepdims=True)
    std = X.std(axis=0,  keepdims=True)
    std[std < 1e-8] = 1.0
    X = (X - mu) / std

    return {"X": X, "mode_id": mode_id, "run_id": run_id}


def load_tep_modes(raw_dir: Path = TEP_RAW_DIR) -> dict[str, np.ndarray]:
    """Fault-free (IDV0) runs; mode derived from production-mode signal."""
    if not _check_available(raw_dir):
        download_tep()
        sys.exit(1)
    return _load_fault_csvs(raw_dir, fault_ids=[0])


def load_tep_faults(
    raw_dir: Path = TEP_RAW_DIR,
    fault_ids: list[int] | None = None,
) -> dict[str, np.ndarray]:
    """Subset of fault types; mode_id = fault number (0=normal)."""
    if not _check_available(raw_dir):
        download_tep()
        sys.exit(1)
    ids = fault_ids if fault_ids is not None else list(range(9))  # IDV0-IDV8
    return _load_fault_csvs(raw_dir, fault_ids=ids)
