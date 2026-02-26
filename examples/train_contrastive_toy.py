"""
Training Script for Contrastive Toy Example

Pipeline:
1) Load NPZ with 'sequences' (N, T, D)
2) Build ContrastiveDataset producing (x1, x2) pairs
3) Train an encoder-only model with NT-Xent (InfoNCE) loss
4) Save best model checkpoint and embeddings
"""

import argparse
import yaml
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).parent.parent / "src"))

from log_to_vec.data.contrastive_dataset import ContrastiveDataset
from log_to_vec.models.contrastive import LSTMContrastiveEncoder
from log_to_vec.training.contrastive_losses import nt_xent_loss
from log_to_vec.evaluation.contrastive_evaluation import (
    compute_contrastive_metrics,
    ContrastiveEvalConfig,
)

def split_indices(num_items: int,
                  train_ratio: float,
                  val_ratio: float,
                  seed: int = 42,
                  shuffle: bool = True):
    """Split indices into train/val/test."""
    if train_ratio <= 0 or val_ratio <= 0 or train_ratio + val_ratio >= 1.0:
        raise ValueError("Invalid split ratios. Need train>0, val>0, train+val<1.")

    indices = np.arange(num_items)
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(indices)

    n_train = int(num_items * train_ratio)
    n_val = int(num_items * val_ratio)

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]
    return train_idx, val_idx, test_idx


def train_epoch(model, dataloader, optimizer, temperature, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0

    for batch in tqdm(dataloader, desc="Training"):
        batch = {k: v.to(device) for k, v in batch.items()}

        x1 = batch["x1"]  # (B, T, D)
        x2 = batch["x2"]  # (B, T, D)

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
def validate_epoch(model, dataloader, temperature, device, eval_cfg: ContrastiveEvalConfig):
    """
    Validate model for one epoch:
    - compute NT-Xent loss
    - compute mode-agnostic contrastive metrics
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0

    # Collect projections for metrics (and optionally embeddings)
    all_z1 = []
    all_z2 = []
    all_h = []

    for batch in tqdm(dataloader, desc="Validation"):
        batch = {k: v.to(device) for k, v in batch.items()}

        x1 = batch["x1"]
        x2 = batch["x2"]

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
        return avg_loss, {}, np.empty((0,))

    z1 = torch.cat(all_z1, dim=0)
    z2 = torch.cat(all_z2, dim=0)

    metrics = compute_contrastive_metrics(z1=z1, z2=z2, cfg=eval_cfg)

    # Export embeddings (h) for later visualization/evaluation
    embeddings = torch.cat(all_h, dim=0).cpu().numpy()

    return avg_loss, metrics, embeddings


def main():
    parser = argparse.ArgumentParser(description="Train contrastive log embedding model on toy example")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/contrastive_toy.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--npz-file",
        type=str,
        default=None,
        help="Path to processed .npz file (overrides config)"
    )
    args = parser.parse_args()

    # Load configuration
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Override npz file if specified
    if args.npz_file:
        config["data"]["npz_file"] = args.npz_file

    # Seeds
    torch.manual_seed(config["training"]["seed"])
    np.random.seed(config["training"]["seed"])

    # Device
    device = torch.device(config["training"]["device"] if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Inspect NPZ
    npz_path = config["data"]["npz_file"]
    print(f"\nLoading processed data from {npz_path}.")
    npz = np.load(npz_path)
    if "sequences" not in npz:
        raise KeyError(f"'sequences' not found in {npz_path}. Keys: {list(npz.keys())}")
    sequences = npz["sequences"]
    if sequences.ndim != 3:
        raise ValueError(f"'sequences' must have shape (N,T,D), got {sequences.shape}")

    N, T, D = sequences.shape
    print(f"Loaded sequences with shape (N,T,D)=({N},{T},{D})")

    # Dataset
    print("\nCreating dataset.")
    base_ds = ContrastiveDataset(
        npz_path=npz_path,
        pair_mode=config["data"].get("pair_mode", "neighbor"),
        normalize=config["data"].get("normalize", False),
        return_meta=config["data"].get("return_meta", False),
        # If you implement augment later, pass transform here or wrap dataset outside
    )
    num_items = len(base_ds)
    print(f"Available items (len(dataset)): {num_items}")

    # Split
    print("\nSplitting train/val/test.")
    train_idx, val_idx, test_idx = split_indices(
        num_items=num_items,
        train_ratio=config["data"]["train_ratio"],
        val_ratio=config["data"]["val_ratio"],
        seed=config["training"]["seed"],
        shuffle=config["data"].get("shuffle_split", True),
    )
    print(f"Split sizes -> train: {len(train_idx)}, val: {len(val_idx)}, test: {len(test_idx)}")

    train_ds = Subset(base_ds, train_idx.tolist())
    val_ds = Subset(base_ds, val_idx.tolist())
    test_ds = Subset(base_ds, test_idx.tolist())

    # Dataloaders
    print("\nCreating dataloaders.")
    train_loader = DataLoader(
        train_ds,
        batch_size=config["data"]["batch_size"],
        shuffle=True,
        num_workers=config["data"].get("num_workers", 0),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config["data"]["batch_size"],
        shuffle=False,
        num_workers=config["data"].get("num_workers", 0),
        drop_last=False,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=config["data"]["batch_size"],
        shuffle=False,
        num_workers=config["data"].get("num_workers", 0),
        drop_last=False,
    )
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}, Test batches: {len(test_loader)}")

    # Model
    print("\nCreating contrastive model.")
    model = LSTMContrastiveEncoder(
        input_dim=D,
        hidden_dim=config["model"]["hidden_dim"],
        num_layers=config["model"]["num_layers"],
        projection_dim=config["model"]["projection_dim"],
        dropout=config["model"]["dropout"],
    ).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )
    temperature = config["training"]["temperature"]

    # Evaluation config
    eval_cfg = ContrastiveEvalConfig(
        uniformity_t=config["evaluation"].get("uniformity_t", 2.0),
        uniformity_max_samples=config["evaluation"].get("uniformity_max_samples", 2048),
        eps=1e-8,
    )

    # Checkpoint dir
    checkpoint_dir = Path(config["logging"]["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    print("\nStarting training.")
    best_val_loss = float("inf")
    eval_every = config["evaluation"].get("eval_every_n_epochs", 5)

    for epoch in range(config["training"]["num_epochs"]):
        print(f"\nEpoch {epoch + 1}/{config['training']['num_epochs']}")

        train_loss = train_epoch(model, train_loader, optimizer, temperature, device)
        print(f"Train Loss: {train_loss:.4f}")

        val_loss, val_metrics, _ = validate_epoch(model, val_loader, temperature, device, eval_cfg)
        print(f"Val Loss: {val_loss:.4f}")

        if (epoch + 1) % eval_every == 0 and val_metrics:
            print("\nContrastive Metrics (val):")
            for k, v in val_metrics.items():
                print(f"  {k}: {v:.6f}")

        # Save best checkpoint by val loss (stable early stopping criterion)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = checkpoint_dir / "best_model.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "config": config,
            }, checkpoint_path)
            print(f"Saved best model to {checkpoint_path}")

    # Final evaluation on test set
    print("\n" + "=" * 50)
    print("Final evaluation on test set.")

    test_loss, test_metrics, test_embeddings = validate_epoch(model, test_loader, temperature, device, eval_cfg)
    print(f"Test Loss: {test_loss:.4f}")

    if test_metrics:
        print("\nContrastive Metrics (test):")
        for k, v in test_metrics.items():
            print(f"  {k}: {v:.6f}")

    # Save embeddings (encoder representations h)
    if test_embeddings.size != 0:
        embeddings_path = checkpoint_dir / "test_embeddings.npy"
        np.save(embeddings_path, test_embeddings)
        print(f"\nSaved test embeddings to {embeddings_path}")

    print("\nTraining complete!")


if __name__ == "__main__":
    main()