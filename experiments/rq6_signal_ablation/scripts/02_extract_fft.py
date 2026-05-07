"""Step 2: Extract FFT embeddings (8-bin and 32-bin) for all RQ6 scenarios.

Run directly on the login node (CPU only, <1 min total).

Usage:
  python 02_extract_fft.py --data_dir ../results/data --output_dir ../results/embeddings
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Tuple

import numpy as np

_RQ6_SRC = Path(__file__).parent.parent / "src"
_V2_SRC  = Path(__file__).parent.parent.parent.parent / "src"
sys.path.insert(0, str(_RQ6_SRC))
sys.path.insert(0, str(_V2_SRC))

from rq6.config.scenarios import SCENARIO_IDS
from version2.evaluation.baseline_features import fft_features


def _fill_missing(X: np.ndarray, mask: np.ndarray, strategy: str) -> np.ndarray:
    X_out = X.copy()
    if strategy == "zero":
        return np.nan_to_num(X_out, nan=0.0)
    elif strategy == "mean":
        N, L, C = X.shape
        for i in range(N):
            for c in range(C):
                obs = mask[i, :, c]
                if obs.any() and not obs.all():
                    X_out[i, ~obs, c] = float(X[i, obs, c].mean())
                elif not obs.any():
                    X_out[i, :, c] = 0.0
        return X_out
    raise ValueError(f"Unknown fill strategy: {strategy!r}")


def _extract_and_save(
    scenario_id: str,
    data_dir: Path,
    output_dir: Path,
    bins_list: Tuple[int, ...],
    fill_strategy: str,
    force: bool,
) -> None:
    npz_path = data_dir / scenario_id / "windows.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"Dataset not found: {npz_path}")

    data = np.load(npz_path)
    X_raw = data["X"]
    mask  = data["mask"]
    meta_keys = ["mode_id", "trajectory_id", "window_start",
                 "is_transition_window", "distance_to_boundary"]

    X_filled = _fill_missing(X_raw, mask, fill_strategy)

    for keep_bins in bins_list:
        out_path = output_dir / f"{scenario_id}_fft{keep_bins}.npz"
        if out_path.exists() and not force:
            print(f"    [skip] fft{keep_bins}")
            continue
        emb = fft_features(X_filled, keep_bins=keep_bins)
        save_dict = {"embeddings": emb.astype(np.float32)}
        for k in meta_keys:
            save_dict[k] = data[k]
        np.savez_compressed(out_path, **save_dict)
        print(f"    [done] fft{keep_bins}  shape={emb.shape}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir",   type=str, default="../results/data")
    p.add_argument("--output_dir", type=str, default="../results/embeddings")
    p.add_argument("--scenarios",  nargs="+", default=None)
    p.add_argument("--bins",       nargs="+", type=int, default=[8, 32])
    p.add_argument("--fill",       choices=["zero", "mean"], default="mean")
    p.add_argument("--force",      action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    data_dir   = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    scenario_ids = args.scenarios if args.scenarios else SCENARIO_IDS
    bins_list    = tuple(args.bins)

    print(f"FFT extraction — bins={bins_list}, fill={args.fill}")
    t0_total = time.time()

    for sid in scenario_ids:
        t0 = time.time()
        print(f"  {sid}")
        _extract_and_save(sid, data_dir, output_dir, bins_list, args.fill, args.force)
        print(f"    {time.time()-t0:.1f}s")

    print(f"\nAll done in {time.time()-t0_total:.1f}s")


if __name__ == "__main__":
    main()
