from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from version2.data.fsss_dataset import FSSSWindowDataset
from version2.models.tcn_hybrid import TCNBackbone


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

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

        self.timestamp_head = nn.Sequential(
            nn.Linear(hidden_dim, embedding_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.projection_head = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.GELU(),
            nn.Linear(embedding_dim, projection_dim),
        )

    def encode_timestamp(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns timestamp-level embeddings: [B, T, D]
        """
        h = self.backbone(x)          # [B, H, T]
        h = h.transpose(1, 2)         # [B, T, H]
        z_t = self.timestamp_head(h)  # [B, T, D]
        return z_t

    def pool_window(self, timestamp_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Mean pooling over time for the final window embedding.
        """
        return timestamp_embeddings.mean(dim=1)

    def forward(self, x: torch.Tensor):
        z_t = self.encode_timestamp(x)                   # [B, T, D]
        z_w = self.pool_window(z_t)                      # [B, D]
        proj = l2_normalize(self.projection_head(z_w))   # [B, P]

        return {
            "timestamp_embeddings": z_t,
            "window_embeddings": z_w,
            "projections": proj,
        }


# -----------------------------------------------------------------------------
# Embedding extraction
# -----------------------------------------------------------------------------

@torch.no_grad()
def extract_split_embeddings(
    model: TS2VecStyleTCNEncoder,
    dataloader: DataLoader,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    all_embeddings = []

    for batch in tqdm(dataloader, desc="Extracting", leave=False):
        x = batch["x"].to(device, non_blocking=True)
        outputs = model(x)
        all_embeddings.append(outputs["window_embeddings"].cpu())

    if len(all_embeddings) == 0:
        return np.empty((0,), dtype=np.float32)

    return torch.cat(all_embeddings, dim=0).numpy().astype(np.float32)


def save_embeddings(output_path: Path, embeddings: np.ndarray) -> None:
    np.savez_compressed(output_path, embeddings=embeddings)


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract embeddings from TS2Vec-style TCN encoder.")
    parser.add_argument("--train_file", type=str, required=True)
    parser.add_argument("--val_file", type=str, required=True)
    parser.add_argument("--test_file", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_ds = FSSSWindowDataset(args.train_file)
    val_ds = FSSSWindowDataset(args.val_file)
    test_ds = FSSSWindowDataset(args.test_file)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)

    model_type = checkpoint.get("model_type", None)
    if model_type is not None and model_type != "ts2vec_style_tcn":
        raise ValueError(
            f"Checkpoint model_type={model_type} is not compatible with TS2Vec-style extractor."
        )

    model = TS2VecStyleTCNEncoder(
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