"""Step 3: Extract MOMENT embeddings for all RQ6 scenarios.

Shorter windows (< 512) are zero-padded on the right to meet MOMENT's input
requirement.  Channels are embedded independently and mean-pooled.

Run on Alvis GPU node via rq6_full.slurm.

Usage:
  python 03_extract_moment.py --data_dir ../results/data \
      --output_dir ../results/embeddings [--batch_size 64] [--device cuda]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

_RQ6_SRC = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(_RQ6_SRC))

from rq6.config.scenarios import SCENARIO_IDS

_MOMENT_LEN = 512


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
    """X_batch: [B, 1, T].  Returns [B, 768]."""
    import torch
    with torch.no_grad():
        t   = torch.tensor(X_batch, dtype=torch.float32).to(device)
        out = model(x_enc=t)
    return out.embeddings.cpu().numpy()


def _extract_and_save(
    scenario_id: str,
    data_dir: Path,
    out_dir: Path,
    batch_size: int,
    device: str,
    force: bool,
) -> None:
    out_path = out_dir / f"{scenario_id}_moment.npz"
    if out_path.exists() and not force:
        print(f"  [skip] {scenario_id}_moment")
        return

    data_path = data_dir / scenario_id / "windows.npz"
    if not data_path.exists():
        print(f"  [miss] {scenario_id} — run 01_generate first")
        return

    d = np.load(data_path)
    W = d["X"].astype(np.float32)            # [N, L, C]
    N, L, C = W.shape

    # Z-score per channel across dataset
    mu  = W.reshape(-1, C).mean(axis=0)
    std = W.reshape(-1, C).std(axis=0) + 1e-8
    W_z = ((W - mu) / std).transpose(0, 2, 1)   # [N, C, L]

    # Pad to MOMENT minimum length
    if L < _MOMENT_LEN:
        pad = np.zeros((N, C, _MOMENT_LEN - L), dtype=W_z.dtype)
        W_z = np.concatenate([W_z, pad], axis=2)

    # Batch inference — mean-pool across channels
    emb_list = []
    for start in range(0, N, batch_size):
        batch = W_z[start:start + batch_size]   # [B, C, T]
        ch_embs = [
            _embed_batch(model, batch[:, ch:ch + 1, :], device)
            for ch in range(C)
        ]
        emb_list.append(np.stack(ch_embs, axis=1).mean(axis=1))

    embs = np.concatenate(emb_list, axis=0)
    meta_keys = ["mode_id", "trajectory_id", "window_start",
                 "is_transition_window", "distance_to_boundary"]
    save_dict = {"embeddings": embs}
    for k in meta_keys:
        save_dict[k] = d[k]
    np.savez_compressed(out_path, **save_dict)
    print(f"  [done] {scenario_id}_moment  shape={embs.shape}")


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
    args    = parse_args()
    data_dir = Path(args.data_dir)
    out_dir  = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ids      = args.scenarios or SCENARIO_IDS

    print(f"Loading MOMENT model (device={args.device})...")
    global model
    model = _load_moment(args.device)

    t0_total = time.time()
    for sid in ids:
        t0 = time.time()
        _extract_and_save(sid, data_dir, out_dir, args.batch_size, args.device, args.force)
        if (out_dir / f"{sid}_moment.npz").exists():
            print(f"    {time.time()-t0:.1f}s")

    print(f"\nAll done in {time.time()-t0_total:.1f}s")


if __name__ == "__main__":
    main()
