"""Step 3: Extract MOMENT embeddings for RQ7 real-world scenarios.

MOMENT expects [B, 1, T] or [B, C, T].  We z-score each channel per-window,
pad to 512 if window_length < 512, then process all channels independently
and mean-pool the per-channel 768-dim representations.

Run on Alvis GPU node via rq7_full.slurm.

Usage:
  python 03_extract_moment.py --data_dir ../results/data \
      --output_dir ../results/embeddings [--device cuda]
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

MOMENT_MIN_LEN = 512


def _load_moment(device: str = "cpu"):
    import torch
    from momentfm import MOMENTPipeline
    model = MOMENTPipeline.from_pretrained(
        "AutonLab/MOMENT-1-base",
        model_kwargs={"task_name": "embedding"},
    )
    model.init()
    model.to(torch.device(device))
    return model


def _embed_batch(model, X_batch: np.ndarray, device: str = "cpu") -> np.ndarray:
    """X_batch: [B, L, C].  Returns [B, 768]."""
    import torch
    B, L, C = X_batch.shape

    mu  = X_batch.mean(axis=1, keepdims=True)
    std = X_batch.std(axis=1,  keepdims=True) + 1e-8
    X_z = (X_batch - mu) / std

    if L < MOMENT_MIN_LEN:
        pad = np.zeros((B, MOMENT_MIN_LEN - L, C), dtype=np.float32)
        X_z = np.concatenate([X_z, pad], axis=1)

    channel_embs = []
    for c in range(C):
        x_c = torch.tensor(
            X_z[:, :, c:c+1].transpose(0, 2, 1), dtype=torch.float32
        ).to(device)
        with torch.no_grad():
            out = model(x_enc=x_c)
        channel_embs.append(out.embeddings.cpu().numpy())

    return np.stack(channel_embs, axis=0).mean(axis=0).astype(np.float32)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir",   default="../results/data")
    p.add_argument("--output_dir", default="../results/embeddings")
    p.add_argument("--scenarios",  nargs="+", default=None)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--device",     default="cpu")
    p.add_argument("--force",      action="store_true")
    return p.parse_args()


def main():
    args     = parse_args()
    data_dir = Path(args.data_dir)
    out_dir  = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ids = args.scenarios or SCENARIO_IDS

    print(f"Loading MOMENT (device={args.device}) …")
    model = _load_moment(args.device)

    for sid in ids:
        out_path = out_dir / f"{sid}_moment.npz"
        if out_path.exists() and not args.force:
            print(f"  [skip] {sid}_moment")
            continue

        windows_path = data_dir / sid / "windows.npz"
        if not windows_path.exists():
            print(f"  [miss] {sid} — run 01_prepare_data first")
            continue

        t0   = time.time()
        data = np.load(windows_path)
        X    = data["X"].astype(np.float32)
        N    = len(X)

        embs = []
        for i in range(0, N, args.batch_size):
            embs.append(_embed_batch(model, X[i:i+args.batch_size], args.device))
            if (i // args.batch_size) % 10 == 0:
                print(f"    {i}/{N}", end="\r", flush=True)

        embeddings = np.concatenate(embs, axis=0)
        np.savez_compressed(
            out_path,
            embeddings   = embeddings,
            mode_id      = data["mode_id"],
            run_id       = data["run_id"],
            window_start = data["window_start"],
        )
        print(f"  [done] {sid}_moment  shape={embeddings.shape}  {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
