"""Step 3: Train a TS2Vec-style encoder on a periodic-mode dataset and extract embeddings.

Reuses the TS2VecStyleEncoder architecture and NT-Xent training loop from
experiments/rq1_recoverability/scripts/11_run_ts2vec_recoverability.py.

Usage:
  python 03_train_ts2vec.py --data_dir data --output_dir embeddings --problem p1_simple_1d
  python 03_train_ts2vec.py --data_dir data --output_dir embeddings --problem p2_multichannel --epochs 60
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from rq2.generation.periodic_mode_generator import create_splits

PROBLEMS = ["p1_simple_1d", "p2_multichannel", "p3_hard_noisy"]


# ---------------------------------------------------------------------------
# Dataset & model (self-contained copy so this script is importable standalone)
# ---------------------------------------------------------------------------

class WindowDataset(Dataset):
    def __init__(self, X: np.ndarray, mean: np.ndarray | None = None, std: np.ndarray | None = None):
        self.x = X.astype(np.float32)
        self.mean = mean
        self.std = std

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int) -> torch.Tensor:
        x = self.x[idx]
        if self.mean is not None and self.std is not None:
            x = (x - self.mean) / self.std
        return torch.from_numpy(x)


class DilatedResidualBlock(nn.Module):
    def __init__(self, channels: int, dilation: int, dropout: float):
        super().__init__()
        padding = dilation
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=3, padding=padding, dilation=dilation),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size=3, padding=padding, dilation=dilation),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.norm = nn.BatchNorm1d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x + self.net(x))


class TS2VecStyleEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, embedding_dim: int, projection_dim: int, dropout: float):
        super().__init__()
        self.input = nn.Conv1d(input_dim, hidden_dim, kernel_size=1)
        self.blocks = nn.Sequential(
            DilatedResidualBlock(hidden_dim, dilation=1, dropout=dropout),
            DilatedResidualBlock(hidden_dim, dilation=2, dropout=dropout),
            DilatedResidualBlock(hidden_dim, dilation=4, dropout=dropout),
            DilatedResidualBlock(hidden_dim, dilation=8, dropout=dropout),
        )
        self.embedding = nn.Sequential(
            nn.Linear(hidden_dim, embedding_dim),
            nn.GELU(),
            nn.LayerNorm(embedding_dim),
        )
        self.projector = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.GELU(),
            nn.Linear(embedding_dim, projection_dim),
        )

    def encode(self, x_btc: torch.Tensor) -> torch.Tensor:
        h = self.input(x_btc.transpose(1, 2))
        h = self.blocks(h)
        pooled = F.adaptive_avg_pool1d(h, output_size=1).squeeze(-1)
        return self.embedding(pooled)

    def forward(self, x_btc: torch.Tensor):
        emb = self.encode(x_btc)
        return emb, self.projector(emb)


def _augment(x: torch.Tensor, noise_std: float, mask_ratio: float, scale_std: float, min_crop: float) -> torch.Tensor:
    B, L, C = x.shape
    crop_len = max(4, int(L * random.uniform(min_crop, 1.0)))
    if crop_len < L:
        start = random.randint(0, L - crop_len)
        x = x[:, start : start + crop_len, :]
        x = F.interpolate(x.transpose(1, 2), size=L, mode="linear", align_corners=False).transpose(1, 2)
    if scale_std > 0:
        x = x * (1.0 + torch.randn(B, 1, C, device=x.device) * scale_std)
    if noise_std > 0:
        x = x + torch.randn_like(x) * noise_std
    if mask_ratio > 0:
        mask = torch.rand(B, L, 1, device=x.device) < mask_ratio
        x = x.masked_fill(mask, 0.0)
    return x


def _nt_xent(z1: torch.Tensor, z2: torch.Tensor, temperature: float) -> torch.Tensor:
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    z = torch.cat([z1, z2], dim=0)
    sim = torch.matmul(z, z.T) / temperature
    B = z1.shape[0]
    mask = torch.eye(2 * B, device=z.device, dtype=torch.bool)
    sim = sim.masked_fill(mask, -1e9)
    targets = (torch.arange(2 * B, device=z.device) + B) % (2 * B)
    return F.cross_entropy(sim, targets)


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def _run_epoch(model, loader, optimizer, device, args) -> float:
    training = optimizer is not None
    model.train(training)
    losses = []
    for batch in loader:
        x = batch.to(device)
        v1 = _augment(x, args.noise_std, args.mask_ratio, args.scale_std, args.min_crop_ratio)
        v2 = _augment(x, args.noise_std, args.mask_ratio, args.scale_std, args.min_crop_ratio)
        _, z1 = model(v1)
        _, z2 = model(v2)
        loss = _nt_xent(z1, z2, args.temperature)
        if training:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses)) if losses else float("nan")


@torch.no_grad()
def _export(model, loader, device) -> np.ndarray:
    model.eval()
    chunks = []
    for batch in loader:
        chunks.append(model.encode(batch.to(device)).cpu())
    return torch.cat(chunks, dim=0).numpy().astype(np.float32)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, default="data")
    p.add_argument("--output_dir", type=str, default="embeddings")
    p.add_argument("--problem", type=str, choices=PROBLEMS, required=True)
    p.add_argument("--seed", type=int, default=42)
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
    p.add_argument("--mask_ratio", type=float, default=0.08)
    p.add_argument("--scale_std", type=float, default=0.10)
    p.add_argument("--min_crop_ratio", type=float, default=0.65)
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[{args.problem}] Training TS2Vec on {device}")

    data_dir = Path(args.data_dir) / args.problem
    splits = create_splits(data_dir, seed=args.seed)

    X_train = splits["train"]["X"]
    mean = X_train.mean(axis=(0, 1)).astype(np.float32)
    std = X_train.std(axis=(0, 1)).clip(min=1e-6).astype(np.float32)

    def _loader(split_name: str, shuffle: bool) -> DataLoader:
        return DataLoader(
            WindowDataset(splits[split_name]["X"], mean, std),
            batch_size=args.batch_size,
            shuffle=shuffle,
            drop_last=shuffle,
            num_workers=0,
        )

    train_loader = _loader("train", shuffle=True)
    val_loader = _loader("val", shuffle=False)

    input_dim = X_train.shape[2] if X_train.ndim == 3 else 1
    model = TS2VecStyleEncoder(
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        embedding_dim=args.embedding_dim,
        projection_dim=args.projection_dim,
        dropout=args.dropout,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val = float("inf")
    for epoch in range(1, args.epochs + 1):
        train_loss = _run_epoch(model, train_loader, optimizer, device, args)
        val_loss = _run_epoch(model, val_loader, None, device, args)
        if epoch % 10 == 0 or epoch == 1:
            print(f"  epoch {epoch:3d}  train={train_loss:.4f}  val={val_loss:.4f}")
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)

    # Export embeddings for all splits
    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    for split_name in ["train", "val", "test"]:
        loader = DataLoader(
            WindowDataset(splits[split_name]["X"], mean, std),
            batch_size=args.batch_size, shuffle=False, num_workers=0,
        )
        emb = _export(model, loader, device)
        split_data = splits[split_name]
        out_path = out_root / f"{args.problem}_ts2vec_{split_name}.npz"
        np.savez_compressed(
            out_path,
            embeddings=emb,
            mode_id=split_data["mode_id"],
            trajectory_id=split_data["trajectory_id"],
            window_start=split_data["window_start"],
            is_transition_window=split_data["is_transition_window"],
            distance_to_boundary=split_data["distance_to_boundary"],
        )
        print(f"  -> {out_path}  shape={emb.shape}")

    # Also save the full (unsplit) embeddings for trace visualization
    full_loader = DataLoader(
        WindowDataset(
            np.load(data_dir / "windows.npz")["X"], mean, std
        ),
        batch_size=args.batch_size, shuffle=False, num_workers=0,
    )
    full_data = np.load(data_dir / "windows.npz")
    full_emb = _export(model, full_loader, device)
    out_full = out_root / f"{args.problem}_ts2vec.npz"
    np.savez_compressed(
        out_full,
        embeddings=full_emb,
        mode_id=full_data["mode_id"],
        trajectory_id=full_data["trajectory_id"],
        window_start=full_data["window_start"],
        is_transition_window=full_data["is_transition_window"],
        distance_to_boundary=full_data["distance_to_boundary"],
    )
    print(f"  -> {out_full}  (full, shape={full_emb.shape})")


if __name__ == "__main__":
    main()
