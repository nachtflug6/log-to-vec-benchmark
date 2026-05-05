"""Step 3: Extract MOMENT embeddings for all RQ4 scenarios.

MOMENT expects [B, 1, T] or [B, C, T].  We z-score each channel, pad to 512
if window_length < 512, then pass all channels independently and mean-pool the
per-channel 768-dim representations into a single vector.

Run on Alvis GPU node via rq4_moment.slurm.

Usage:
  python 03_extract_moment.py --data_dir ../results/data \
      --output_dir ../results/embeddings
"""
from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path

import numpy as np

_HERE    = Path(__file__).resolve().parent
_RQ4_SRC = _HERE.parent / "src"
if str(_RQ4_SRC) not in sys.path:
    sys.path.insert(0, str(_RQ4_SRC))

from rq4.config.scenarios import SCENARIO_IDS


MOMENT_MIN_LEN = 512


def _load_moment():
    from momentfm import MOMENTPipeline
    model = MOMENTPipeline.from_pretrained(
        "AutonLab/MOMENT-1-large",
        model_kwargs={"output_attentions": False},
        task="embedding",
    )
    model.init()
    return model


def _embed_batch(model, X_batch: np.ndarray) -> np.ndarray:
    """X_batch: [B, L, C].  Returns [B, 768]."""
    import torch
    B, L, C = X_batch.shape

    # z-score per channel per sample
    mu  = X_batch.mean(axis=1, keepdims=True)
    std = X_batch.std(axis=1, keepdims=True) + 1e-8
    X_z = (X_batch - mu) / std

    # pad to MOMENT_MIN_LEN if needed
    if L < MOMENT_MIN_LEN:
        pad = np.zeros((B, MOMENT_MIN_LEN - L, C), dtype=np.float32)
        X_z = np.concatenate([X_z, pad], axis=1)

    # Embed each channel independently, then mean-pool
    channel_embs = []
    for c in range(C):
        x_c = torch.tensor(X_z[:, :, c:c+1].transpose(0, 2, 1), dtype=torch.float32)
        with torch.no_grad():
            out = model(x_enc=x_c)
        channel_embs.append(out.embeddings.cpu().numpy())   # [B, 768]

    return np.stack(channel_embs, axis=0).mean(axis=0).astype(np.float32)  # [B, 768]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir",   default="../results/data")
    p.add_argument("--output_dir", default="../results/embeddings")
    p.add_argument("--scenarios",  nargs="+", default=None)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--force",      action="store_true")
    return p.parse_args()


def main():
    args    = parse_args()
    data_dir = Path(args.data_dir)
    out_dir  = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ids = args.scenarios or SCENARIO_IDS

    print("Loading MOMENT…")
    model = _load_moment()

    for sid in ids:
        out_path = out_dir / f"{sid}_moment.npz"
        if out_path.exists() and not args.force:
            print(f"  [skip] {sid}_moment")
            continue

        windows_path = data_dir / sid / "windows.npz"
        if not windows_path.exists():
            print(f"  [miss] {sid} — run 01_generate first")
            continue

        t0   = time.time()
        data = np.load(windows_path)
        X    = data["X"].astype(np.float32)   # [N, L, C]
        N    = len(X)

        embs = []
        for i in range(0, N, args.batch_size):
            embs.append(_embed_batch(model, X[i:i+args.batch_size]))
            if (i // args.batch_size) % 10 == 0:
                print(f"    {i}/{N}", end="\r", flush=True)

        embeddings = np.concatenate(embs, axis=0)
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
        print(f"  [done] {sid}_moment  shape={embeddings.shape}  {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
