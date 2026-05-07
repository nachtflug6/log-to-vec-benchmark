"""Step 2c: Extract Distribution-per-Bucket (DBF) embeddings for all RQ3 scenarios.

DBF splits each window into K time buckets and computes [mean, std, min, max] per
channel per bucket. This gives a 128-dim feature vector (K=8, C=4, 4 stats) that
captures *local distributional shape* rather than global spectral content.

Key properties:
- Step-signal modes have near-zero std within each bucket; sine modes have high std.
  This directly addresses the gap where FFT fails on step signals.
- NaN-aware: each statistic is computed over observed values only. Fully-missing
  buckets default to zeros (flagged by zero std).
- No training required; CPU-only; runs in seconds.

Usage:
  python 02c_extract_dbf.py --data_dir ../results/data \
      --output_dir ../results/embeddings [--num_buckets 8] [--force]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

_RQ3_SRC = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(_RQ3_SRC))

from rq3.config.scenarios import SCENARIO_IDS


def _dbf_features(X: np.ndarray, mask: np.ndarray, num_buckets: int) -> np.ndarray:
    """Compute distribution-per-bucket features.

    X:    [N, L, C] float32, NaN at missing positions
    mask: [N, L, C] bool, True = observed
    Returns: [N, num_buckets * C * 4] float32
             Statistics order per bucket per channel: [mean, std, min, max]
    """
    N, L, C = X.shape
    bucket_size = L / num_buckets
    out = np.zeros((N, num_buckets * C * 4), dtype=np.float32)

    for b in range(num_buckets):
        t_start = int(round(b * bucket_size))
        t_end   = int(round((b + 1) * bucket_size))
        t_end   = max(t_end, t_start + 1)          # guarantee ≥1 timestep
        t_end   = min(t_end, L)

        X_bucket    = X[:, t_start:t_end, :]      # [N, w, C]
        mask_bucket = mask[:, t_start:t_end, :]   # [N, w, C]

        for c in range(C):
            obs   = mask_bucket[:, :, c]           # [N, w] bool
            vals  = X_bucket[:, :, c].copy()       # [N, w]
            vals[~obs] = np.nan

            base = b * C * 4 + c * 4
            n_obs = obs.sum(axis=1)                # [N]
            any_obs = n_obs > 0

            # mean
            with np.errstate(all="ignore"):
                m = np.nanmean(vals, axis=1)
            out[any_obs, base + 0] = m[any_obs]

            # std (ddof=0; 0 when only 1 observed value → step signal indicator)
            with np.errstate(all="ignore"):
                s = np.nanstd(vals, axis=1)
            out[any_obs, base + 1] = s[any_obs]

            # min
            with np.errstate(all="ignore"):
                mn = np.nanmin(vals, axis=1)
            out[any_obs, base + 2] = mn[any_obs]

            # max
            with np.errstate(all="ignore"):
                mx = np.nanmax(vals, axis=1)
            out[any_obs, base + 3] = mx[any_obs]

    return out


def _extract_and_save(
    scenario_id: str,
    data_dir: Path,
    output_dir: Path,
    num_buckets: int,
    force: bool,
) -> None:
    out_path = output_dir / f"{scenario_id}_dbf.npz"
    if out_path.exists() and not force:
        print(f"    [skip] dbf (already exists)")
        return

    npz_path = data_dir / scenario_id / "windows.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"Dataset not found: {npz_path}")

    data    = np.load(npz_path)
    X_raw   = data["X"]        # [N, L, C] with NaN
    mask    = data["mask"]     # [N, L, C] bool
    meta_keys = ["mode_id", "trajectory_id", "window_start",
                 "is_transition_window", "distance_to_boundary"]

    emb = _dbf_features(X_raw, mask, num_buckets)

    save_dict = {"embeddings": emb}
    for k in meta_keys:
        save_dict[k] = data[k]

    np.savez_compressed(out_path, **save_dict)
    print(f"    [done] dbf  shape={emb.shape}  (K={num_buckets})")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir",    type=str, default="../results/data")
    p.add_argument("--output_dir",  type=str, default="../results/embeddings")
    p.add_argument("--scenarios",   nargs="+", default=None)
    p.add_argument("--num_buckets", type=int, default=8)
    p.add_argument("--force",       action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    data_dir   = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    scenario_ids = args.scenarios if args.scenarios else SCENARIO_IDS

    print(f"DBF extraction — K={args.num_buckets} buckets × C channels × 4 stats")
    print(f"  data:   {data_dir}")
    print(f"  output: {output_dir}")
    t0_total = time.time()

    for sid in scenario_ids:
        t0 = time.time()
        print(f"  {sid}")
        _extract_and_save(sid, data_dir, output_dir, args.num_buckets, args.force)
        print(f"    {time.time() - t0:.1f}s")

    print(f"\nAll done in {time.time() - t0_total:.1f}s")


if __name__ == "__main__":
    main()
