"""Step 2: Extract FFT baseline embeddings for RQ7 real-world scenarios.

All channels in SWaT/MSL/TEP are continuous — no discrete_cols needed.
Z-scores each channel across the full dataset before FFT.

Usage:
  python 02_extract_fft.py --data_dir ../results/data \
      --output_dir ../results/embeddings [--top_k 16]
"""
from __future__ import annotations

import argparse, sys, time
from pathlib import Path

import numpy as np

_HERE    = Path(__file__).resolve().parent
_RQ7_SRC = _HERE.parent / "src"
if str(_RQ7_SRC) not in sys.path:
    sys.path.insert(0, str(_RQ7_SRC))

from rq7.config.scenarios import SCENARIO_IDS


def _fft_features(X: np.ndarray, top_k: int) -> np.ndarray:
    """X: [N, L, C].  Returns [N, top_k*C] feature matrix."""
    N, L, C = X.shape
    mu  = X.reshape(-1, C).mean(axis=0)
    std = X.reshape(-1, C).std(axis=0) + 1e-8
    X_z = (X - mu) / std
    fft_amp = np.abs(np.fft.rfft(X_z, axis=1))[:, 1:top_k+1, :]  # [N, top_k, C]
    return fft_amp.reshape(N, -1).astype(np.float32)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir",   default="../results/data")
    p.add_argument("--output_dir", default="../results/embeddings")
    p.add_argument("--scenarios",  nargs="+", default=None)
    p.add_argument("--top_k",      type=int, default=16)
    p.add_argument("--force",      action="store_true")
    return p.parse_args()


def main():
    args     = parse_args()
    data_dir = Path(args.data_dir)
    out_dir  = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ids = args.scenarios or SCENARIO_IDS

    for sid in ids:
        out_path = out_dir / f"{sid}_fft.npz"
        if out_path.exists() and not args.force:
            print(f"  [skip] {sid}_fft")
            continue

        windows_path = data_dir / sid / "windows.npz"
        if not windows_path.exists():
            print(f"  [miss] {sid} — run 01_prepare_data first")
            continue

        t0   = time.time()
        data = np.load(windows_path)
        X    = data["X"].astype(np.float32)

        embeddings = _fft_features(X, args.top_k)

        np.savez_compressed(
            out_path,
            embeddings   = embeddings,
            mode_id      = data["mode_id"],
            run_id       = data["run_id"],
            window_start = data["window_start"],
        )
        print(f"  [done] {sid}_fft  shape={embeddings.shape}  {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
