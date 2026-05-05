"""Step 3: Extract MOMENT embeddings for all window-size variants.

Shorter windows (< 512) are zero-padded on the right to meet MOMENT's
input requirement. This is intentional — we want to measure what happens
at each window size, not hold window size constant for MOMENT.

Usage:
  python 03_extract_moment.py --data_dir ../results/data \
      --output_dir ../results/embeddings [--batch_size 64]
"""
from __future__ import annotations

import argparse, sys, time
from pathlib import Path

import numpy as np

_HERE    = Path(__file__).resolve().parent
_RQ5_SRC = _HERE.parent / "src"
_RQ4_SRC = _HERE.parent.parent / "rq4_cnc_benchmark" / "src"
for _p in [str(_RQ5_SRC), str(_RQ4_SRC)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from rq5.config.sweep import SCENARIO_IDS

_MOMENT_LEN = 512


def _load_moment():
    from momentfm import MOMENTPipeline
    model = MOMENTPipeline.from_pretrained(
        "AutonLab/MOMENT-1-large", model_kwargs={"output_attentions": False}
    )
    model.init()
    return model


def _embed_batch(model, X_batch: np.ndarray) -> np.ndarray:
    import torch
    with torch.no_grad():
        t = torch.tensor(X_batch, dtype=torch.float32)
        out = model(x_enc=t)
    return out.embeddings.cpu().numpy()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir",   default="../results/data")
    p.add_argument("--output_dir", default="../results/embeddings")
    p.add_argument("--scenarios",  nargs="+", default=None)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--force",      action="store_true")
    return p.parse_args()


def main():
    args     = parse_args()
    data_dir = Path(args.data_dir)
    out_dir  = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ids      = args.scenarios or SCENARIO_IDS

    print("Loading MOMENT model...")
    model = _load_moment()

    for sid in ids:
        out_path = out_dir / f"{sid}_moment.npz"
        if out_path.exists() and not args.force:
            continue
        t1        = time.time()
        data_path = data_dir / sid / "windows.npz"
        if not data_path.exists():
            print(f"  [miss] {sid}")
            continue
        d = np.load(data_path)
        W = d["X"]                         # [N, L, C]
        N, L, C = W.shape

        # Normalise per channel across dataset
        mu  = W.reshape(-1, C).mean(axis=0)
        std = W.reshape(-1, C).std(axis=0) + 1e-8
        W_z = ((W - mu) / std).transpose(0, 2, 1)  # [N, C, L]

        # Pad time axis to _MOMENT_LEN if needed
        if L < _MOMENT_LEN:
            pad = np.zeros((N, C, _MOMENT_LEN - L), dtype=W_z.dtype)
            W_z = np.concatenate([W_z, pad], axis=2)

        # Batch inference — mean-pool across channels
        emb_list = []
        for start in range(0, N, args.batch_size):
            batch = W_z[start:start + args.batch_size]     # [B, C, T]
            ch_embs = []
            for ch in range(C):
                ch_embs.append(_embed_batch(model, batch[:, ch:ch+1, :]))
            emb_list.append(np.stack(ch_embs, axis=1).mean(axis=1))

        embs = np.concatenate(emb_list, axis=0)
        np.savez_compressed(out_path,
            embeddings      = embs,
            program_step_id = d["program_step_id"],
            op_state_id     = d["op_state_id"],
            trajectory_id   = d["trajectory_id"],
            window_start    = d["window_start"],
        )
        print(f"  [done] {sid}_moment  shape={embs.shape}  {time.time()-t1:.1f}s")


if __name__ == "__main__":
    main()
