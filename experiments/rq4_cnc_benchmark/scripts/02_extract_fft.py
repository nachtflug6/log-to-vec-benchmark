"""Step 2: Extract FFT baseline embeddings for all RQ4 scenarios.

Continuous channels → per-channel amplitude spectrum (top-k bins).
Discrete channels   → window-level mean (passed as a single scalar feature).

Z-scores each channel across the full dataset before FFT so that physical
unit differences (mm vs RPM vs °C) don't dominate.

Usage:
  python 02_extract_fft.py --data_dir ../results/data \
      --output_dir ../results/embeddings [--top_k 16]
"""
from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path

import numpy as np

_HERE    = Path(__file__).resolve().parent
_RQ4_SRC = _HERE.parent / "src"
if str(_RQ4_SRC) not in sys.path:
    sys.path.insert(0, str(_RQ4_SRC))

from rq4.config.scenarios import SCENARIO_IDS, get_scenario


def _fft_features(X: np.ndarray, top_k: int, discrete_cols: list) -> np.ndarray:
    """X: [N, L, C].  Returns [N, D] feature matrix."""
    N, L, C = X.shape
    cont_cols = [c for c in range(C) if c not in discrete_cols]

    feats = []

    # FFT on continuous channels
    if cont_cols:
        X_cont = X[:, :, cont_cols]                    # [N, L, len(cont_cols)]
        # z-score per channel over the whole dataset
        mu  = X_cont.reshape(-1, len(cont_cols)).mean(axis=0)
        std = X_cont.reshape(-1, len(cont_cols)).std(axis=0) + 1e-8
        X_z = (X_cont - mu) / std
        # FFT amplitude spectrum
        fft_amp = np.abs(np.fft.rfft(X_z, axis=1))[:, 1:top_k+1, :]  # [N, top_k, C_cont]
        feats.append(fft_amp.reshape(N, -1))

    # Discrete channels: z-scored window mean
    if discrete_cols:
        X_disc = X[:, :, discrete_cols].mean(axis=1)   # [N, len(discrete_cols)]
        mu_d  = X_disc.mean(axis=0)
        std_d = X_disc.std(axis=0) + 1e-8
        feats.append((X_disc - mu_d) / std_d)

    return np.concatenate(feats, axis=1).astype(np.float32)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir",    default="../results/data")
    p.add_argument("--output_dir",  default="../results/embeddings")
    p.add_argument("--scenarios",   nargs="+", default=None)
    p.add_argument("--top_k",       type=int, default=16)
    p.add_argument("--force",       action="store_true")
    return p.parse_args()


def main():
    args    = parse_args()
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
            print(f"  [miss] {sid} — run 01_generate first")
            continue

        t0   = time.time()
        data = np.load(windows_path)
        X    = data["X"]                              # [N, L, C]
        meta = json.loads((data_dir / sid / "metadata.json").read_text())
        disc = meta.get("discrete_channels", [])

        embeddings = _fft_features(X, args.top_k, disc)

        np.savez_compressed(
            out_path,
            embeddings      = embeddings,
            program_step_id = data["program_step_id"],
            op_state_id     = data["op_state_id"],
            run_id          = data["run_id"],
            part_type_id    = data["part_type_id"],
            trajectory_id   = data["trajectory_id"],
            window_start    = data["window_start"],
        )
        print(f"  [done] {sid}_fft  shape={embeddings.shape}  {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
