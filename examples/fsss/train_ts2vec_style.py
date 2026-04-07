from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from version2.data.fsss_dataset import FSSSWindowDataset
from version2.models.tcn_hybrid import TCNBackbone
from version2.training.hybrid_losses import variance_loss, covariance_loss


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def l2_normalize(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return x / (x.norm(dim=-1, keepdim=True) + eps)


# -----------------------------------------------------------------------------
# TS2Vec-style TCN encoder
# -----------------------------------------------------------------------------

class TS2VecStyleTCNEncoder(nn.Module):
    """
    TS2Vec-style encoder built on top of the existing TCN backbone.

    Input:
        x: [B, T, C]

    Output:
        {
            "timestamp_embeddings": [B, T, D],
            "window_embeddings": [B, D],
            "projections": [B, P]
        }
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        embedding_dim: int = 128,
        projection_dim: int = 64,
        num_blocks: int = 4,
        kernel_size: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.projection_dim = projection_dim
        self.num_blocks = num_blocks
        self.kernel_size = kernel_size
        self.dropout = dropout

        self.backbone = TCNBackbone(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_blocks=num_blocks,
            kernel_size=kernel_size,
            dropout=dropout,
        )

        # Map timestamp hidden states to timestamp embeddings.
        self.timestamp_head = nn.Sequential(
            nn.Linear(hidden_dim, embedding_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Projection head for contrastive learning on window-level embedding.
        self.projection_head = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.GELU(),
            nn.Linear(embedding_dim, projection_dim),
        )

    def encode_timestamp(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns timestamp-level embeddings: [B, T, D]
        """
        h = self.backbone(x)              # [B, H, T]
        h = h.transpose(1, 2)             # [B, T, H]
        z_t = self.timestamp_head(h)      # [B, T, D]
        return z_t

    def pool_window(self, timestamp_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Mean pooling over time, as recommended by the memo for a clean baseline.
        """
        return timestamp_embeddings.mean(dim=1)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        z_t = self.encode_timestamp(x)                  # [B, T, D]
        z_w = self.pool_window(z_t)                     # [B, D]
        proj = l2_normalize(self.projection_head(z_w))  # [B, P]

        return {
            "timestamp_embeddings": z_t,
            "window_embeddings": z_w,
            "projections": proj,
        }


# -----------------------------------------------------------------------------
# View generation
# -----------------------------------------------------------------------------

def weak_augment(
    x: torch.Tensor,
    noise_std: float = 0.01,
    scale_std: float = 0.02,
    max_time_shift: int = 1,
) -> torch.Tensor:
    """
    Conservative augmentation.
    Avoid strong warping/permutation to preserve meaningful structure.
    """
    out = x.clone()

    # Small additive noise
    if noise_std > 0:
        out = out + torch.randn_like(out) * noise_std

    # Mild per-sample global scaling
    if scale_std > 0:
        scales = 1.0 + torch.randn(out.shape[0], 1, 1, device=out.device) * scale_std
        out = out * scales

    # Small temporal roll
    if max_time_shift > 0:
        shifts = torch.randint(
            low=-max_time_shift,
            high=max_time_shift + 1,
            size=(out.shape[0],),
            device=out.device,
        )
        out = torch.stack(
            [torch.roll(out[i], shifts=int(shifts[i].item()), dims=0) for i in range(out.shape[0])],
            dim=0,
        )

    return out


def random_overlapping_crop_pair(
    x: torch.Tensor,
    crop_min_ratio: float = 0.6,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create two overlapping crops from the same window and resize them back to T.
    This is closer to TS2Vec-style shared-content learning than temporal-neighbor pairing.

    Input:
        x: [B, T, C]

    Returns:
        crop1, crop2: [B, T, C], resized back to original T
    """
    B, T, C = x.shape
    min_crop_len = max(2, int(T * crop_min_ratio))

    crop1_list = []
    crop2_list = []

    for b in range(B):
        # Sample crop length
        L = np.random.randint(min_crop_len, T + 1)

        # Sample first crop
        s1 = np.random.randint(0, T - L + 1)
        e1 = s1 + L

        # Force overlap with crop1
        overlap_min = max(0, s1 - L // 2)
        overlap_max = min(T - L, e1 - 1)
        if overlap_max < overlap_min:
            s2 = s1
        else:
            s2 = np.random.randint(overlap_min, overlap_max + 1)
        e2 = s2 + L

        crop1 = x[b:b + 1, s1:e1, :]  # [1, L, C]
        crop2 = x[b:b + 1, s2:e2, :]  # [1, L, C]

        # Resize back to original T using linear interpolation along time.
        crop1 = F.interpolate(
            crop1.transpose(1, 2), size=T, mode="linear", align_corners=False
        ).transpose(1, 2)  # [1, T, C]

        crop2 = F.interpolate(
            crop2.transpose(1, 2), size=T, mode="linear", align_corners=False
        ).transpose(1, 2)  # [1, T, C]

        crop1_list.append(crop1)
        crop2_list.append(crop2)

    return torch.cat(crop1_list, dim=0), torch.cat(crop2_list, dim=0)


def build_two_views(
    x: torch.Tensor,
    crop_min_ratio: float,
    noise_std: float,
    scale_std: float,
    max_time_shift: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Two-view pipeline:
    1. overlapping crop pair
    2. weak augmentations on each view
    """
    v1, v2 = random_overlapping_crop_pair(x, crop_min_ratio=crop_min_ratio)
    v1 = weak_augment(v1, noise_std=noise_std, scale_std=scale_std, max_time_shift=max_time_shift)
    v2 = weak_augment(v2, noise_std=noise_std, scale_std=scale_std, max_time_shift=max_time_shift)
    return v1, v2


# -----------------------------------------------------------------------------
# Losses and diagnostics
# -----------------------------------------------------------------------------

def nt_xent_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.2) -> torch.Tensor:
    """
    Standard SimCLR-style NT-Xent loss on window-level projections.
    """
    if z1.shape != z2.shape:
        raise ValueError(f"Shape mismatch: z1={z1.shape}, z2={z2.shape}")

    batch_size = z1.shape[0]
    z = torch.cat([z1, z2], dim=0)                     # [2B, P]
    sim = torch.matmul(z, z.T) / temperature          # [2B, 2B]

    eye = torch.eye(2 * batch_size, device=z.device, dtype=torch.bool)
    sim = sim.masked_fill(eye, -1e9)

    targets = torch.arange(batch_size, device=z.device)
    targets = torch.cat([targets + batch_size, targets], dim=0)

    return F.cross_entropy(sim, targets)


def compute_batch_neighbor_gap(z1: torch.Tensor, z2: torch.Tensor) -> float:
    """
    Positive similarity minus random negative similarity.
    Useful as a training-time unsupervised signal.
    """
    a = l2_normalize(z1)
    b = l2_normalize(z2)

    pos = (a * b).sum(dim=1).mean().item()

    perm = torch.randperm(a.shape[0], device=a.device)
    neg = (a * b[perm]).sum(dim=1).mean().item()

    return float(pos - neg)


def compute_embedding_health(embeddings: List[torch.Tensor]) -> Dict[str, float]:
    if len(embeddings) == 0:
        return {"overall_std": 0.0, "mean_dim_std": 0.0, "frac_low_variance_dims": 1.0}

    z = torch.cat(embeddings, dim=0)
    dim_std = z.std(dim=0, unbiased=False)
    overall_std = z.std(unbiased=False).item()
    frac_low_variance_dims = (dim_std < 1e-3).float().mean().item()

    return {
        "overall_std": float(overall_std),
        "mean_dim_std": float(dim_std.mean().item()),
        "frac_low_variance_dims": float(frac_low_variance_dims),
    }


def compute_selection_score(
    val_loss: float,
    emb_health: Dict[str, float],
    neighbor_gap: float,
) -> float:
    """
    Higher is better.
    Favor low loss, good positive-vs-negative separation, and healthy variance.
    """
    return (
        -1.0 * val_loss
        + 0.50 * neighbor_gap
        + 0.20 * emb_health["mean_dim_std"]
        - 2.0 * emb_health["frac_low_variance_dims"]
    )


# -----------------------------------------------------------------------------
# Train / eval loop
# -----------------------------------------------------------------------------

def run_epoch(
    model: TS2VecStyleTCNEncoder,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    train: bool,
    temperature: float,
    crop_min_ratio: float,
    noise_std: float,
    scale_std: float,
    max_time_shift: int,
    variance_weight: float,
    covariance_weight: float,
) -> Tuple[Dict[str, float], Dict[str, float], float]:
    if train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_neighbor_gap = 0.0
    num_batches = 0
    collected_embeddings: List[torch.Tensor] = []

    context = torch.enable_grad() if train else torch.no_grad()

    with context:
        for batch in tqdm(dataloader, leave=False):
            x = batch["x"].to(device, non_blocking=True)   # [B, T, C]

            v1, v2 = build_two_views(
                x=x,
                crop_min_ratio=crop_min_ratio,
                noise_std=noise_std,
                scale_std=scale_std,
                max_time_shift=max_time_shift,
            )

            out1 = model(v1)
            out2 = model(v2)

            loss_contrastive = nt_xent_loss(
                out1["projections"],
                out2["projections"],
                temperature=temperature,
            )

            emb_cat = torch.cat(
                [out1["window_embeddings"], out2["window_embeddings"]],
                dim=0,
            )

            loss_var = variance_loss(emb_cat)
            loss_cov = covariance_loss(emb_cat)

            loss = (
                    loss_contrastive
                    + variance_weight * loss_var
                    + covariance_weight * loss_cov
            )

            if train and optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            total_loss += float(loss.detach().item())
            total_neighbor_gap += compute_batch_neighbor_gap(
                out1["window_embeddings"],
                out2["window_embeddings"],
            )
            collected_embeddings.append(out1["window_embeddings"].detach().cpu())
            num_batches += 1

    if num_batches == 0:
        stats = {"loss_total": 0.0}
        emb_health = {"overall_std": 0.0, "mean_dim_std": 0.0, "frac_low_variance_dims": 1.0}
        return stats, emb_health, 0.0

    stats = {
        "loss_total": total_loss / num_batches,
    }
    emb_health = compute_embedding_health(collected_embeddings)
    avg_neighbor_gap = total_neighbor_gap / num_batches
    return stats, emb_health, avg_neighbor_gap


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train TS2Vec-style TCN encoder on FSSS splits.")

    parser.add_argument("--train_file", type=str, required=True)
    parser.add_argument("--val_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)

    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--embedding_dim", type=int, default=128)
    parser.add_argument("--projection_dim", type=int, default=64)
    parser.add_argument("--num_blocks", type=int, default=4)
    parser.add_argument("--kernel_size", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.1)

    parser.add_argument("--temperature", type=float, default=0.2)

    parser.add_argument("--variance_weight", type=float, default=0)
    parser.add_argument("--covariance_weight", type=float, default=0)

    # TS2Vec-style view generation
    parser.add_argument("--crop_min_ratio", type=float, default=0.6)
    parser.add_argument("--noise_std", type=float, default=0.01)
    parser.add_argument("--scale_std", type=float, default=0.02)
    parser.add_argument("--max_time_shift", type=int, default=1)

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_ds = FSSSWindowDataset(args.train_file)
    val_ds = FSSSWindowDataset(args.val_file)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
    )

    input_dim = train_ds.X.shape[2]

    model = TS2VecStyleTCNEncoder(
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        embedding_dim=args.embedding_dim,
        projection_dim=args.projection_dim,
        num_blocks=args.num_blocks,
        kernel_size=args.kernel_size,
        dropout=args.dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    best_score = -float("inf")
    history = []

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")

        train_stats, train_emb_health, train_neighbor_gap = run_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            device=device,
            train=True,
            temperature=args.temperature,
            crop_min_ratio=args.crop_min_ratio,
            noise_std=args.noise_std,
            scale_std=args.scale_std,
            max_time_shift=args.max_time_shift,
            variance_weight=args.variance_weight,
            covariance_weight=args.covariance_weight,
        )

        val_stats, val_emb_health, val_neighbor_gap = run_epoch(
            model=model,
            dataloader=val_loader,
            optimizer=None,
            device=device,
            train=False,
            temperature=args.temperature,
            crop_min_ratio=args.crop_min_ratio,
            noise_std=args.noise_std,
            scale_std=args.scale_std,
            max_time_shift=args.max_time_shift,
            variance_weight=args.variance_weight,
            covariance_weight=args.covariance_weight,
        )

        selection_score = compute_selection_score(
            val_loss=val_stats["loss_total"],
            emb_health=val_emb_health,
            neighbor_gap=val_neighbor_gap,
        )

        print(f"train_total={train_stats['loss_total']:.4f}")
        print(f"val_total={val_stats['loss_total']:.4f}")
        print(
            f"val_neighbor_gap={val_neighbor_gap:.4f} "
            f"val_mean_dim_std={val_emb_health['mean_dim_std']:.4f} "
            f"val_frac_low_var={val_emb_health['frac_low_variance_dims']:.4f} "
            f"selection_score={selection_score:.4f}"
        )

        history.append(
            {
                "epoch": epoch + 1,
                "train": train_stats,
                "train_embedding_health": train_emb_health,
                "train_neighbor_gap": train_neighbor_gap,
                "val": val_stats,
                "val_embedding_health": val_emb_health,
                "val_neighbor_gap": val_neighbor_gap,
                "selection_score": selection_score,
            }
        )

        ckpt = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "selection_score": selection_score,
            "val_loss": val_stats["loss_total"],
            "input_dim": input_dim,
            "hidden_dim": args.hidden_dim,
            "embedding_dim": args.embedding_dim,
            "projection_dim": args.projection_dim,
            "num_blocks": args.num_blocks,
            "kernel_size": args.kernel_size,
            "dropout": args.dropout,
            "model_type": "ts2vec_style_tcn",
        }

        torch.save(ckpt, output_dir / "last_model.pt")

        if selection_score > best_score:
            best_score = selection_score
            torch.save(ckpt, output_dir / "best_model.pt")
            print(f"Saved best checkpoint -> {output_dir / 'best_model.pt'}")

    with open(output_dir / "training_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    print(f"\nDone. Best selection score: {best_score:.4f}")


if __name__ == "__main__":
    main()