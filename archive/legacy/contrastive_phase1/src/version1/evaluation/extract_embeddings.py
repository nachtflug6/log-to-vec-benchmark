"""
Load a trained contrastive encoder checkpoint and extract sequence embeddings
from processed train / val / test datasets.

This script:
1. Loads processed dataset splits
2. Loads a trained model checkpoint
3. Runs the encoder on each split
4. Saves extracted embeddings for later evaluation and visualization
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from version1.data.dataset import SequenceDataset
from version1.models.contrastive import LSTMContrastiveEncoder


@torch.no_grad()
def extract_split_embeddings(model, dataloader, device):
    """
    Extract encoder embeddings for one dataset split.

    Args:
        model: trained encoder model
        dataloader: dataloader for one split
        device: torch device

    Returns:
        embeddings: numpy array of shape [N, embedding_dim]
    """
    model.eval()

    all_embeddings = []

    for batch in tqdm(dataloader, desc="Extracting", leave=False):
        x = batch["x"].to(device, non_blocking=True)

        outputs = model(x)
        h = outputs["embeddings"]  # [B, embedding_dim]

        all_embeddings.append(h.cpu())

    if len(all_embeddings) == 0:
        return np.empty((0,), dtype=np.float32)

    embeddings = torch.cat(all_embeddings, dim=0).numpy().astype(np.float32)
    return embeddings


def save_embeddings(output_path, embeddings):
    """
    Save extracted embeddings to compressed .npz format.
    """
    np.savez_compressed(output_path, embeddings=embeddings)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract embeddings from a trained contrastive encoder"
    )

    parser.add_argument(
        "--train_file",
        type=str,
        default="data/processed/version1/train.npz"
    )
    parser.add_argument(
        "--val_file",
        type=str,
        default="data/processed/version1/val.npz"
    )
    parser.add_argument(
        "--test_file",
        type=str,
        default="data/processed/version1/test.npz"
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/version1/contrastive/best_model.pt"
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/embeddings/version1/contrastive"
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=256
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # Load processed datasets
    # -------------------------------------------------------------------------
    print("\nLoading processed datasets...")

    train_ds = SequenceDataset(args.train_file)
    val_ds = SequenceDataset(args.val_file)
    test_ds = SequenceDataset(args.test_file)

    print(f"Train size: {len(train_ds)}")
    print(f"Val size:   {len(val_ds)}")
    print(f"Test size:  {len(test_ds)}")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
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

    # -------------------------------------------------------------------------
    # Load checkpoint
    # -------------------------------------------------------------------------
    print("\nLoading checkpoint...")
    checkpoint = torch.load(args.checkpoint, map_location=device)

    model = LSTMContrastiveEncoder(
        input_dim=checkpoint["input_dim"],
        hidden_dim=checkpoint["hidden_dim"],
        num_layers=checkpoint["num_layers"],
        projection_dim=checkpoint["projection_dim"],
        dropout=checkpoint["dropout"],
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"Loaded checkpoint from: {args.checkpoint}")
    print(f"Checkpoint epoch: {checkpoint['epoch'] + 1}")
    print(f"Validation loss at save time: {checkpoint['val_loss']:.4f}")

    # -------------------------------------------------------------------------
    # Extract embeddings
    # -------------------------------------------------------------------------
    print("\nExtracting train embeddings...")
    train_embeddings = extract_split_embeddings(model, train_loader, device)
    print(f"Train embeddings shape: {train_embeddings.shape}")

    print("\nExtracting val embeddings...")
    val_embeddings = extract_split_embeddings(model, val_loader, device)
    print(f"Val embeddings shape: {val_embeddings.shape}")

    print("\nExtracting test embeddings...")
    test_embeddings = extract_split_embeddings(model, test_loader, device)
    print(f"Test embeddings shape: {test_embeddings.shape}")

    # -------------------------------------------------------------------------
    # Save outputs
    # -------------------------------------------------------------------------
    train_out = output_dir / "train_embeddings.npz"
    val_out = output_dir / "val_embeddings.npz"
    test_out = output_dir / "test_embeddings.npz"

    save_embeddings(train_out, train_embeddings)
    save_embeddings(val_out, val_embeddings)
    save_embeddings(test_out, test_embeddings)

    print("\nSaved embedding files:")
    print(f"  {train_out}")
    print(f"  {val_out}")
    print(f"  {test_out}")

    print("\nDone.")


if __name__ == "__main__":
    main()