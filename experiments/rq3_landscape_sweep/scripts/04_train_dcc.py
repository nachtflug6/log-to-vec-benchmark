"""Step 4: Train DCC (Dilated-Conv-Contrast) encoder for one scenario (GPU job).

DCC is our custom contrastive encoder: dilated residual blocks + NT-Xent loss
with instance-level (window-level) average pooling.  It is NOT the official
TS2Vec (Yue et al. 2022), which uses timestamp-level hierarchical contrast.
We label all outputs 'dcc' to avoid confusion.

Run via SLURM array job: sbatch slurm/rq3_dcc.slurm
Or directly: python 04_train_dcc.py --scenario_id baseline
"""
from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

# Make rq3 importable
_RQ3_SRC = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(_RQ3_SRC))

# Reuse DCC implementation from RQ2 without copying
_RQ2_SCRIPTS = Path(__file__).parent.parent.parent / "rq2_trace_comparison" / "scripts"
sys.path.insert(0, str(_RQ2_SCRIPTS))

from rq3.config.scenarios import SCENARIO_IDS, get_scenario
from train_ts2vec import (  # noqa: E402  (rq2 script name)
    TS2VecStyleEncoder,
    WindowDataset,
    _augment,
    _nt_xent,
    _run_epoch,
    _export,
)


def _augment_with_structural_mask(
    x: torch.Tensor,
    obs_mask: torch.Tensor,   # [B, L, C] bool — True = observed
    noise_std: float,
    scale_std: float,
    min_crop: float,
) -> torch.Tensor:
    """Apply the pre-existing missing mask, then standard augmentation (no extra random mask)."""
    x = x.masked_fill(~obs_mask, 0.0)
    return _augment(x, noise_std=noise_std, mask_ratio=0.0, scale_std=scale_std, min_crop_ratio=min_crop)


class MaskedWindowDataset(WindowDataset):
    """WindowDataset extended to also return the obs_mask alongside each window."""

    def __init__(self, X: np.ndarray, obs_mask: np.ndarray) -> None:
        super().__init__(X)
        self.obs_mask = torch.tensor(obs_mask, dtype=torch.bool)

    def __getitem__(self, idx):
        x = self.data[idx]          # [L, C] tensor
        m = self.obs_mask[idx]      # [L, C] bool tensor
        return x, m


def _run_epoch_masked(model, loader, optimizer, device, args) -> float:
    model.train()
    total_loss = 0.0
    for x, obs_mask in loader:
        x = x.to(device, dtype=torch.float32)
        obs_mask = obs_mask.to(device)
        x1 = _augment_with_structural_mask(x, obs_mask, args.noise_std, args.scale_std, args.min_crop_ratio)
        x2 = _augment_with_structural_mask(x, obs_mask, args.noise_std, args.scale_std, args.min_crop_ratio)
        z1 = model(x1)
        z2 = model(x2)
        loss = _nt_xent(z1, z2, args.temperature)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(x)
    return total_loss / len(loader.dataset)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--scenario_id", type=str, required=True)
    p.add_argument("--data_dir", type=str, default="../results/data")
    p.add_argument("--output_dir", type=str, default="../results/embeddings")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--hidden_dim", type=int, default=64)
    p.add_argument("--embedding_dim", type=int, default=64)
    p.add_argument("--projection_dim", type=int, default=64)
    p.add_argument("--dropout", type=float, default=0.10)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--temperature", type=float, default=0.20)
    p.add_argument("--noise_std", type=float, default=0.03)
    p.add_argument("--mask_ratio", type=float, default=0.08,
                   help="Random mask ratio for clean scenarios; overridden by structural mask when present")
    p.add_argument("--scale_std", type=float, default=0.10)
    p.add_argument("--min_crop_ratio", type=float, default=0.65)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    cfg = get_scenario(args.scenario_id)
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    out_path = output_dir / f"{args.scenario_id}_dcc.npz"
    if out_path.exists() and not args.force:
        print(f"[skip] {args.scenario_id} — DCC embeddings already exist")
        return

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[{args.scenario_id}] DCC training on {device}")

    npz_path = data_dir / args.scenario_id / "windows.npz"
    data = np.load(npz_path)
    X = data["X"]                   # [N, L, C] — may contain NaN
    obs_mask = data["mask"]         # [N, L, C] bool
    has_missing = not obs_mask.all()

    # Zero-fill NaN so PyTorch tensors are valid; mask handles the rest
    X_filled = np.nan_to_num(X, nan=0.0).astype(np.float32)

    N, L, C = X_filled.shape

    # Simple 80/20 train/val split (no leakage concerns for contrastive pre-training)
    rng = np.random.default_rng(args.seed)
    idx = rng.permutation(N)
    split = int(0.8 * N)
    train_idx, val_idx = idx[:split], idx[split:]

    if has_missing:
        train_ds = MaskedWindowDataset(X_filled[train_idx], obs_mask[train_idx])
        val_ds   = MaskedWindowDataset(X_filled[val_idx],   obs_mask[val_idx])
    else:
        train_ds = WindowDataset(X_filled[train_idx])
        val_ds   = WindowDataset(X_filled[val_idx])

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader   = torch.utils.data.DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False)

    model = TS2VecStyleEncoder(
        input_dim=C,
        hidden_dim=args.hidden_dim,
        embedding_dim=args.embedding_dim,
        projection_dim=args.projection_dim,
        dropout=args.dropout,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val = float("inf")
    best_state = None

    for epoch in range(1, args.epochs + 1):
        if has_missing:
            train_loss = _run_epoch_masked(model, train_loader, optimizer, device, args)
            with torch.no_grad():
                val_loss = _run_epoch_masked(model, val_loader, None, device, args) \
                    if False else _eval_val_masked(model, val_loader, device, args)
        else:
            train_loss = _run_epoch(model, train_loader, optimizer, device, args)
            val_loss = _eval_val(model, val_loader, device, args)

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            print(f"  epoch {epoch:3d}  train={train_loss:.4f}  val={val_loss:.4f}")

    model.load_state_dict(best_state)
    model.eval()

    # Export embeddings for all windows
    all_ds = MaskedWindowDataset(X_filled, obs_mask) if has_missing else WindowDataset(X_filled)
    all_loader = torch.utils.data.DataLoader(all_ds, batch_size=args.batch_size, shuffle=False)
    emb = _export(model, all_loader, device)   # [N, embedding_dim]

    meta_keys = ["mode_id", "trajectory_id", "window_start",
                 "is_transition_window", "distance_to_boundary"]
    save_dict = {"embeddings": emb.astype(np.float32)}
    for k in meta_keys:
        save_dict[k] = data[k]

    np.savez_compressed(out_path, **save_dict)
    print(f"  → {out_path}  shape={emb.shape}")


def _eval_val(model, loader, device, args) -> float:
    model.eval()
    total = 0.0
    with torch.no_grad():
        for x in loader:
            if isinstance(x, (list, tuple)):
                x = x[0]
            x = x.to(device, dtype=torch.float32)
            x1 = _augment(x, args.noise_std, args.mask_ratio, args.scale_std, args.min_crop_ratio)
            x2 = _augment(x, args.noise_std, args.mask_ratio, args.scale_std, args.min_crop_ratio)
            loss = _nt_xent(model(x1), model(x2), args.temperature)
            total += loss.item() * len(x)
    return total / len(loader.dataset)


def _eval_val_masked(model, loader, device, args) -> float:
    model.eval()
    total = 0.0
    with torch.no_grad():
        for x, obs_mask in loader:
            x = x.to(device, dtype=torch.float32)
            obs_mask = obs_mask.to(device)
            x1 = _augment_with_structural_mask(x, obs_mask, args.noise_std, args.scale_std, args.min_crop_ratio)
            x2 = _augment_with_structural_mask(x, obs_mask, args.noise_std, args.scale_std, args.min_crop_ratio)
            loss = _nt_xent(model(x1), model(x2), args.temperature)
            total += loss.item() * len(x)
    return total / len(loader.dataset)


if __name__ == "__main__":
    main()
