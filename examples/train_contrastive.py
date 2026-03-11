"""
Train a contrastive sequence encoder on processed train/val/test splits.

This script:
1. Loads processed sequence datasets from .npz files
2. Builds dataloaders for contrastive learning
3. Trains an encoder with NT-Xent loss
4. Evaluates on validation and test sets
5. Saves the best checkpoint and exported test embeddings
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from version1.data.dataset import ContrastiveDataset
from version1.models.contrastive import LSTMContrastiveEncoder


def nt_xent_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
    """
    Compute NT-Xent / InfoNCE loss for a batch of positive pairs.

    Args:
        z1: (B, P)
        z2: (B, P)
        temperature: temperature scaling

    Returns:
        Scalar contrastive loss
    """
    batch_size = z1.size(0)

    if batch_size < 2:
        raise ValueError("Batch size must be at least 2 for NT-Xent loss.")

    z = torch.cat([z1, z2], dim=0)  # (2B, P)
    z = F.normalize(z, dim=1)

    similarity = torch.matmul(z, z.T) / temperature  # (2B, 2B)

    # mask self-similarity
    mask = torch.eye(2 * batch_size, device=z.device, dtype=torch.bool)
    similarity = similarity.masked_fill(mask, -1e9)

    # positive index mapping:
    # for i in [0, B-1], positive is i+B
    # for i in [B, 2B-1], positive is i-B
    targets = torch.arange(2 * batch_size, device=z.device)
    targets = (targets + batch_size) % (2 * batch_size)

    loss = F.cross_entropy(similarity, targets)
    return loss


@torch.no_grad()
def compute_contrastive_metrics(z1: torch.Tensor, z2: torch.Tensor, eps: float = 1e-8) -> dict:
    """
    Compute simple contrastive representation metrics.

    Metrics:
        positive_cosine
        alignment
        uniformity
        embedding_variance
    """
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)

    positive_cosine = (z1 * z2).sum(dim=1).mean().item()

    alignment = ((z1 - z2) ** 2).sum(dim=1).mean().item()

    z = torch.cat([z1, z2], dim=0)

    if z.size(0) > 1:
        dists = torch.pdist(z, p=2)
        uniformity = torch.log(torch.exp(-2 * dists.pow(2)).mean() + eps).item()
    else:
        uniformity = float("nan")

    embedding_variance = z.std(dim=0).mean().item()

    return {
        "positive_cosine": positive_cosine,
        "alignment": alignment,
        "uniformity": uniformity,
        "embedding_variance": embedding_variance,
    }


def train_epoch(model, dataloader, optimizer, temperature, device):
    """
    Train for one epoch.
    """
    model.train()
    total_loss = 0.0
    num_batches = 0

    for batch in tqdm(dataloader, desc="Training", leave=False):
        x1 = batch["x_i"].to(device, non_blocking=True)  # (B, T, D)
        x2 = batch["x_j"].to(device, non_blocking=True)  # (B, T, D)

        optimizer.zero_grad()

        out1 = model(x1)
        out2 = model(x2)

        z1 = out1["projections"]
        z2 = out2["projections"]

        loss = nt_xent_loss(z1, z2, temperature=temperature)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / max(1, num_batches)


@torch.no_grad()
def validate_epoch(model, dataloader, temperature, device):
    """
    Validate model for one epoch.

    Returns:
        avg_loss: average NT-Xent loss
        metrics: dictionary of contrastive metrics
        embeddings: encoder embeddings for later export/visualization
    """
    model.eval()

    total_loss = 0.0
    num_batches = 0

    all_z1 = []
    all_z2 = []
    all_h = []

    for batch in tqdm(dataloader, desc="Validation", leave=False):
        x1 = batch["x_i"].to(device, non_blocking=True)
        x2 = batch["x_j"].to(device, non_blocking=True)

        out1 = model(x1)
        out2 = model(x2)

        z1 = out1["projections"]
        z2 = out2["projections"]

        loss = nt_xent_loss(z1, z2, temperature=temperature)

        total_loss += loss.item()
        num_batches += 1

        all_z1.append(z1)
        all_z2.append(z2)
        all_h.append(out1["embeddings"])

    avg_loss = total_loss / max(1, num_batches)

    if len(all_z1) == 0:
        return avg_loss, {}, np.empty((0,), dtype=np.float32)

    z1 = torch.cat(all_z1, dim=0)
    z2 = torch.cat(all_z2, dim=0)

    metrics = compute_contrastive_metrics(z1, z2)

    embeddings = torch.cat(all_h, dim=0).cpu().numpy()

    return avg_loss, metrics, embeddings


def parse_args():
    parser = argparse.ArgumentParser(description="Train contrastive encoder on processed toy log splits")

    parser.add_argument("--train_file", type=str, default="data/processed/version1/train.npz")
    parser.add_argument("--val_file", type=str, default="data/processed/version1/val.npz")
    parser.add_argument("--test_file", type=str, default="data/processed/version1/test.npz")

    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=0)

    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--projection_dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.1)

    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--num_epochs", type=int, default=30)

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval_every_n_epochs", type=int, default=5)

    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/version1/contrastive")

    return parser.parse_args()


def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # -------------------------------------------------------------------------
    # Datasets
    # -------------------------------------------------------------------------
    print("\nLoading processed datasets...")

    train_ds = ContrastiveDataset(args.train_file)
    val_ds = ContrastiveDataset(args.val_file)
    test_ds = ContrastiveDataset(args.test_file)

    print(f"Train size: {len(train_ds)}")
    print(f"Val size:   {len(val_ds)}")
    print(f"Test size:  {len(test_ds)}")

    sample = train_ds[0]
    x_sample = sample["x_i"]
    _, input_dim = x_sample.shape

    print(f"Sample sequence shape: {x_sample.shape}")
    print(f"Input feature dimension: {input_dim}")

    # -------------------------------------------------------------------------
    # Dataloaders
    # -------------------------------------------------------------------------
    print("\nCreating dataloaders...")

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

    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
    )

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches:   {len(val_loader)}")
    print(f"Test batches:  {len(test_loader)}")

    # -------------------------------------------------------------------------
    # Model
    # -------------------------------------------------------------------------
    print("\nCreating contrastive model...")

    model = LSTMContrastiveEncoder(
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        projection_dim=args.projection_dim,
        dropout=args.dropout,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    # -------------------------------------------------------------------------
    # Checkpoint directory
    # -------------------------------------------------------------------------
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # Training
    # -------------------------------------------------------------------------
    print("\nStarting training...")
    best_val_loss = float("inf")

    for epoch in range(args.num_epochs):
        print(f"\nEpoch {epoch + 1}/{args.num_epochs}")

        train_loss = train_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            temperature=args.temperature,
            device=device,
        )
        print(f"Train Loss: {train_loss:.4f}")

        val_loss, val_metrics, _ = validate_epoch(
            model=model,
            dataloader=val_loader,
            temperature=args.temperature,
            device=device,
        )
        print(f"Val Loss: {val_loss:.4f}")

        if (epoch + 1) % args.eval_every_n_epochs == 0 and val_metrics:
            print("\nContrastive Metrics (val):")
            for k, v in val_metrics.items():
                print(f"  {k}: {v:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = checkpoint_dir / "best_model.pt"

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "input_dim": input_dim,
                    "hidden_dim": args.hidden_dim,
                    "num_layers": args.num_layers,
                    "projection_dim": args.projection_dim,
                    "dropout": args.dropout,
                },
                checkpoint_path,
            )
            print(f"Saved best model to {checkpoint_path}")

    # -------------------------------------------------------------------------
    # Final test evaluation
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Final evaluation on test set")
    print("=" * 60)

    test_loss, test_metrics, test_embeddings = validate_epoch(
        model=model,
        dataloader=test_loader,
        temperature=args.temperature,
        device=device,
    )

    print(f"Test Loss: {test_loss:.4f}")

    if test_metrics:
        print("\nContrastive Metrics (test):")
        for k, v in test_metrics.items():
            print(f"  {k}: {v:.6f}")

    if test_embeddings.size != 0:
        embeddings_path = checkpoint_dir / "test_embeddings.npy"
        np.save(embeddings_path, test_embeddings)
        print(f"\nSaved test embeddings to {embeddings_path}")

    print("\nTraining complete!")


if __name__ == "__main__":
    main()