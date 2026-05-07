"""Step 2b: Extract ROCKET embeddings for all RQ3 scenarios.

Uses a standalone numba-based ROCKET implementation (no sktime required).
The kernel transform output is reduced to --pca_dim principal components before
saving (standard practice; keeps npz size manageable and metrics stable).

Run on the login node (CPU + numba, ~10 min total for all 14 scenarios).

Usage:
  python 02b_extract_rocket.py --data_dir ../results/data \
      --output_dir ../results/embeddings [--pca_dim 128] [--num_kernels 10000] [--force]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA

_SCRIPTS = Path(__file__).parent
_RQ3_SRC = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(_RQ3_SRC))
sys.path.insert(0, str(_SCRIPTS))

from rq3.config.scenarios import SCENARIO_IDS
from _rocket_numba import rocket_features


def _fill_missing(X: np.ndarray) -> np.ndarray:
    """Zero-fill NaNs (matches FFT strategy)."""
    return np.nan_to_num(X, nan=0.0)


def _extract_rocket_features(X_train: np.ndarray, X_all: np.ndarray,
                              num_kernels: int, pca_dim: int):
    """Fit ROCKET on train split, transform all windows, apply PCA."""
    N_train = X_train.shape[0]
    features_all = rocket_features(X_train, X_all, num_kernels=num_kernels)

    pca = PCA(n_components=min(pca_dim, features_all.shape[1]), random_state=42)
    pca.fit(features_all[:N_train])                   # fit on train only
    return pca.transform(features_all).astype(np.float32)


def _extract_and_save(
    scenario_id: str,
    data_dir: Path,
    output_dir: Path,
    pca_dim: int,
    num_kernels: int,
    force: bool,
) -> None:
    out_path = output_dir / f"{scenario_id}_rocket.npz"
    if out_path.exists() and not force:
        print(f"    [skip] rocket (already exists)")
        return

    npz_path = data_dir / scenario_id / "windows.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"Dataset not found: {npz_path}")

    data    = np.load(npz_path)
    X_raw   = data["X"]          # [N, L, C] may contain NaN
    traj_ids = data["trajectory_id"]
    meta_keys = ["mode_id", "trajectory_id", "window_start",
                 "is_transition_window", "distance_to_boundary"]

    X_filled = _fill_missing(X_raw)

    # Identify train split (trajectory ids ≤ 70th percentile)
    unique_trajs = np.unique(traj_ids)
    n_train = max(1, int(len(unique_trajs) * 0.70))
    train_trajs = set(unique_trajs[:n_train].tolist())
    train_mask = np.array([t in train_trajs for t in traj_ids])

    X_train = X_filled[train_mask]
    N_train  = X_train.shape[0]

    emb = _extract_rocket_features(X_train, X_filled, num_kernels, pca_dim)

    save_dict = {"embeddings": emb}
    for k in meta_keys:
        save_dict[k] = data[k]

    np.savez_compressed(out_path, **save_dict)
    print(f"    [done] rocket  shape={emb.shape}  (pca_dim={pca_dim})")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir",     type=str, default="../results/data")
    p.add_argument("--output_dir",   type=str, default="../results/embeddings")
    p.add_argument("--scenarios",    nargs="+", default=None)
    p.add_argument("--pca_dim",      type=int, default=128)
    p.add_argument("--num_kernels",  type=int, default=10_000)
    p.add_argument("--force",        action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    data_dir   = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    scenario_ids = args.scenarios if args.scenarios else SCENARIO_IDS

    print(f"ROCKET extraction — num_kernels={args.num_kernels}, pca_dim={args.pca_dim}")
    print(f"  data:   {data_dir}")
    print(f"  output: {output_dir}")
    t0_total = time.time()

    for sid in scenario_ids:
        t0 = time.time()
        print(f"  {sid}")
        _extract_and_save(sid, data_dir, output_dir, args.pca_dim, args.num_kernels, args.force)
        print(f"    {time.time() - t0:.1f}s")

    print(f"\nAll done in {time.time() - t0_total:.1f}s")


if __name__ == "__main__":
    main()
