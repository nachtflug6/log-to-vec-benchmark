from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from version2.data.fsss_dataset import FSSSWindowDataset
from version2.models.tcn_hybrid import TCNHybridEncoder


@torch.no_grad()
def extract_split_embeddings(model, dataloader, device):
    model.eval()
    all_embeddings = []

    for batch in tqdm(dataloader, desc="Extracting", leave=False):
        x = batch["x"].to(device, non_blocking=True)
        outputs = model(x)
        all_embeddings.append(outputs["embeddings"].cpu())

    if len(all_embeddings) == 0:
        return np.empty((0,), dtype=np.float32)

    return torch.cat(all_embeddings, dim=0).numpy().astype(np.float32)


def save_embeddings(output_path, embeddings):
    np.savez_compressed(output_path, embeddings=embeddings)


def parse_args():
    parser = argparse.ArgumentParser(description="Extract embeddings from version2 TCN hybrid encoder.")
    parser.add_argument("--train_file", type=str, required=True)
    parser.add_argument("--val_file", type=str, required=True)
    parser.add_argument("--test_file", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def main():
    args = parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_ds = FSSSWindowDataset(args.train_file)
    val_ds = FSSSWindowDataset(args.val_file)
    test_ds = FSSSWindowDataset(args.test_file)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    checkpoint = torch.load(args.checkpoint, map_location=device)

    model = TCNHybridEncoder(
        input_dim=checkpoint["input_dim"],
        hidden_dim=checkpoint["hidden_dim"],
        embedding_dim=checkpoint["embedding_dim"],
        projection_dim=checkpoint["projection_dim"],
        num_blocks=checkpoint["num_blocks"],
        kernel_size=checkpoint["kernel_size"],
        dropout=checkpoint["dropout"],
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"Loaded checkpoint from: {args.checkpoint}")
    print(f"Checkpoint epoch: {checkpoint['epoch'] + 1}")
    print(f"Validation loss at save time: {checkpoint['val_loss']:.4f}")

    train_emb = extract_split_embeddings(model, train_loader, device)
    val_emb = extract_split_embeddings(model, val_loader, device)
    test_emb = extract_split_embeddings(model, test_loader, device)

    save_embeddings(output_dir / "train_embeddings.npz", train_emb)
    save_embeddings(output_dir / "val_embeddings.npz", val_emb)
    save_embeddings(output_dir / "test_embeddings.npz", test_emb)

    print("Saved:")
    print(output_dir / "train_embeddings.npz")
    print(output_dir / "val_embeddings.npz")
    print(output_dir / "test_embeddings.npz")


if __name__ == "__main__":
    main()