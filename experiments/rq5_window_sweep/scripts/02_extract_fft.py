"""Step 2: Extract FFT embeddings for all window-size variants.

Note: top_k is capped to (window_length // 2) so short windows don't crash.

Usage:
  python 02_extract_fft.py --data_dir ../results/data \
      --output_dir ../results/embeddings
"""
from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path

import numpy as np

_HERE    = Path(__file__).resolve().parent
_RQ5_SRC = _HERE.parent / "src"
_RQ4_SRC = _HERE.parent.parent / "rq4_cnc_benchmark" / "src"
for _p in [str(_RQ5_SRC), str(_RQ4_SRC)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from rq5.config.sweep import SCENARIO_IDS, get_scenario


def _fft_features(X: np.ndarray, top_k: int, discrete_cols: list) -> np.ndarray:
    N, L, C = X.shape
    cont_cols = [c for c in range(C) if c not in discrete_cols]
    # Cap top_k to available frequency bins (rfft gives L//2+1 bins, skip DC)
    usable_k = min(top_k, L // 2)
    feats = []
    if cont_cols:
        X_cont = X[:, :, cont_cols]
        mu  = X_cont.reshape(-1, len(cont_cols)).mean(axis=0)
        std = X_cont.reshape(-1, len(cont_cols)).std(axis=0) + 1e-8
        X_z = (X_cont - mu) / std
        fft_amp = np.abs(np.fft.rfft(X_z, axis=1))[:, 1:usable_k+1, :]
        feats.append(fft_amp.reshape(N, -1))
    if discrete_cols:
        X_disc = X[:, :, discrete_cols].mean(axis=1)
        mu_d   = X_disc.mean(axis=0)
        std_d  = X_disc.std(axis=0) + 1e-8
        feats.append((X_disc - mu_d) / std_d)
    return np.concatenate(feats, axis=1).astype(np.float32)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir",   default="../results/data")
    p.add_argument("--output_dir", default="../results/embeddings")
    p.add_argument("--scenarios",  nargs="+", default=None)
    p.add_argument("--top_k",      type=int,  default=16)
    p.add_argument("--force",      action="store_true")
    return p.parse_args()


def main():
    args    = parse_args()
    data_dir = Path(args.data_dir)
    out_dir  = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ids = args.scenarios or SCENARIO_IDS

    for sid in ids:
        cfg      = get_scenario(sid)
        out_path = out_dir / f"{sid}_fft.npz"
        if out_path.exists() and not args.force:
            continue
        t1 = time.time()
        data_path = data_dir / sid / "windows.npz"
        meta_path = data_dir / sid / "metadata.json"
        if not data_path.exists():
            print(f"  [miss] {sid}")
            continue
        d    = np.load(data_path)
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        disc = meta.get("discrete_channels", [])

        embs = _fft_features(d["X"], args.top_k, disc)
        np.savez_compressed(out_path,
            embeddings      = embs,
            program_step_id = d["program_step_id"],
            op_state_id     = d["op_state_id"],
            trajectory_id   = d["trajectory_id"],
            window_start    = d["window_start"],
        )
        print(f"  [done] {sid}_fft  shape={embs.shape}  {time.time()-t1:.1f}s")


if __name__ == "__main__":
    main()
