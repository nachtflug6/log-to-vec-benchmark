"""Train a TS2Vec-style self-supervised encoder on an RQ1 split and evaluate recoverability.

This is intentionally lightweight and local to the RQ1 pipeline. It trains only on
unlabeled windows. Positive pairs are two independently augmented views of the
same window; all other windows in the batch act as negatives.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import subprocess
from typing import Literal

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset


class WindowDataset(Dataset):
    def __init__(self, npz_path: str | Path, mean: np.ndarray | None = None, std: np.ndarray | None = None):
        with np.load(npz_path) as data:
            self.x = data["X"].astype(np.float32)
        self.mean = mean
        self.std = std

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, index: int) -> torch.Tensor:
        x = self.x[index]
        if self.mean is not None and self.std is not None:
            x = (x - self.mean) / self.std
        return torch.from_numpy(x.astype(np.float32))


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
        # [B, T, C] -> [B, C, T]
        h = self.input(x_btc.transpose(1, 2))
        h = self.blocks(h)
        pooled = F.adaptive_avg_pool1d(h, output_size=1).squeeze(-1)
        return self.embedding(pooled)

    def forward(self, x_btc: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        emb = self.encode(x_btc)
        proj = self.projector(emb)
        return emb, proj


def random_crop_resize(x: torch.Tensor, min_crop_ratio: float) -> torch.Tensor:
    batch, length, channels = x.shape
    crop_len = max(4, int(length * random.uniform(min_crop_ratio, 1.0)))
    if crop_len >= length:
        return x
    start = random.randint(0, length - crop_len)
    cropped = x[:, start : start + crop_len, :]
    # interpolate expects [B, C, T]
    resized = F.interpolate(cropped.transpose(1, 2), size=length, mode="linear", align_corners=False)
    return resized.transpose(1, 2)


def augment_window_batch(
    x: torch.Tensor,
    noise_std: float,
    mask_ratio: float,
    scale_std: float,
    min_crop_ratio: float,
) -> torch.Tensor:
    out = random_crop_resize(x, min_crop_ratio=min_crop_ratio)
    if scale_std > 0:
        scale = 1.0 + torch.randn(out.shape[0], 1, out.shape[2], device=out.device) * scale_std
        out = out * scale
    if noise_std > 0:
        out = out + torch.randn_like(out) * noise_std
    if mask_ratio > 0:
        mask = torch.rand(out.shape[0], out.shape[1], 1, device=out.device) < mask_ratio
        out = out.masked_fill(mask, 0.0)
    return out


def nt_xent_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float) -> torch.Tensor:
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    z = torch.cat([z1, z2], dim=0)
    sim = torch.matmul(z, z.T) / temperature
    batch = z1.shape[0]
    mask = torch.eye(2 * batch, device=z.device, dtype=torch.bool)
    sim = sim.masked_fill(mask, -1e9)
    targets = torch.arange(2 * batch, device=z.device)
    targets = (targets + batch) % (2 * batch)
    return F.cross_entropy(sim, targets)


def compute_train_stats(train_file: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(train_file) as data:
        x = data["X"].astype(np.float32)
    mean = x.mean(axis=(0, 1), keepdims=False)
    std = x.std(axis=(0, 1), keepdims=False).clip(min=1e-6)
    return mean.astype(np.float32), std.astype(np.float32)


def make_loader(path: Path, batch_size: int, shuffle: bool, mean: np.ndarray | None, std: np.ndarray | None) -> DataLoader:
    return DataLoader(
        WindowDataset(path, mean=mean, std=std),
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=shuffle,
        num_workers=0,
    )


def run_epoch(
    model: TS2VecStyleEncoder,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    temperature: float,
    noise_std: float,
    mask_ratio: float,
    scale_std: float,
    min_crop_ratio: float,
) -> float:
    training = optimizer is not None
    model.train(training)
    losses: list[float] = []

    for batch in loader:
        x = batch.to(device)
        view1 = augment_window_batch(
            x,
            noise_std=noise_std,
            mask_ratio=mask_ratio,
            scale_std=scale_std,
            min_crop_ratio=min_crop_ratio,
        )
        view2 = augment_window_batch(
            x,
            noise_std=noise_std,
            mask_ratio=mask_ratio,
            scale_std=scale_std,
            min_crop_ratio=min_crop_ratio,
        )
        _, z1 = model(view1)
        _, z2 = model(view2)
        loss = nt_xent_loss(z1, z2, temperature=temperature)

        if training:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

        losses.append(float(loss.detach().cpu()))

    return float(np.mean(losses)) if losses else float("nan")


@torch.no_grad()
def export_embeddings(model: TS2VecStyleEncoder, loader: DataLoader, device: torch.device) -> np.ndarray:
    model.eval()
    chunks: list[torch.Tensor] = []
    for batch in loader:
        emb = model.encode(batch.to(device))
        chunks.append(emb.detach().cpu())
    return torch.cat(chunks, dim=0).numpy().astype(np.float32)


def save_embeddings(path: Path, embeddings: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, embeddings=embeddings)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and evaluate a TS2Vec-style encoder on an RQ1 dataset split.")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="frs_clean_vnext_long",
        choices=["frs_clean_vnext_long", "frs_noisy_vnext_long"],
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--embedding_dim", type=int, default=128)
    parser.add_argument("--projection_dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--temperature", type=float, default=0.20)
    parser.add_argument("--noise_std", type=float, default=0.03)
    parser.add_argument("--mask_ratio", type=float, default=0.08)
    parser.add_argument("--scale_std", type=float, default=0.10)
    parser.add_argument("--min_crop_ratio", type=float, default=0.65)
    parser.add_argument("--normalization", choices=["train", "none"], default="train")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--python_exe", type=str, default=str(Path(__file__).resolve().parents[3] / "venv" / "Scripts" / "python.exe"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    root = Path(__file__).resolve().parents[1]
    repo_root = root.parents[1]
    split_dir = root / "artifacts" / "datasets" / args.dataset_name / "splits" / f"trajectory_seed{args.seed}"
    train_file = split_dir / "train_windows.npz"
    val_file = split_dir / "val_windows.npz"
    test_file = split_dir / "test_windows.npz"
    if not train_file.exists():
        raise FileNotFoundError(f"Missing split bundle at {split_dir}. Run the FRS pipeline first.")

    run_name = args.run_name or f"{args.dataset_name}_ts2vec_style"
    run_root = root / "artifacts" / "runs" / run_name
    checkpoint_dir = run_root / "ts2vec_style" / "training"
    embedding_dir = run_root / "ts2vec_style" / "embeddings"
    evaluation_dir = run_root / "ts2vec_style" / "evaluation"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    mean = std = None
    if args.normalization == "train":
        mean, std = compute_train_stats(train_file)

    train_loader = make_loader(train_file, args.batch_size, shuffle=True, mean=mean, std=std)
    val_loader = make_loader(val_file, args.batch_size, shuffle=False, mean=mean, std=std)
    train_export_loader = make_loader(train_file, args.batch_size, shuffle=False, mean=mean, std=std)
    test_loader = make_loader(test_file, args.batch_size, shuffle=False, mean=mean, std=std)

    with np.load(train_file) as data:
        input_dim = int(data["X"].shape[2])
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = TS2VecStyleEncoder(
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        embedding_dim=args.embedding_dim,
        projection_dim=args.projection_dim,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val = float("inf")
    history: list[dict[str, float]] = []
    best_path = checkpoint_dir / "best_model.pt"
    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(
            model,
            train_loader,
            optimizer,
            device,
            args.temperature,
            args.noise_std,
            args.mask_ratio,
            args.scale_std,
            args.min_crop_ratio,
        )
        val_loss = run_epoch(
            model,
            val_loader,
            None,
            device,
            args.temperature,
            args.noise_std,
            args.mask_ratio,
            args.scale_std,
            args.min_crop_ratio,
        )
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "args": vars(args),
                    "input_dim": input_dim,
                    "mean": mean,
                    "std": std,
                },
                best_path,
            )
        print(f"epoch={epoch:03d} train_loss={train_loss:.4f} val_loss={val_loss:.4f}")

    (checkpoint_dir / "training_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    save_embeddings(embedding_dir / "train_embeddings.npz", export_embeddings(model, train_export_loader, device))
    save_embeddings(embedding_dir / "val_embeddings.npz", export_embeddings(model, val_loader, device))
    save_embeddings(embedding_dir / "test_embeddings.npz", export_embeddings(model, test_loader, device))

    eval_command = [
        args.python_exe,
        str(root / "scripts" / "06_evaluate_recoverability.py"),
        "--train_embeddings",
        str(embedding_dir / "train_embeddings.npz"),
        "--val_embeddings",
        str(embedding_dir / "val_embeddings.npz"),
        "--test_embeddings",
        str(embedding_dir / "test_embeddings.npz"),
        "--train_split",
        str(train_file),
        "--val_split",
        str(val_file),
        "--test_split",
        str(test_file),
        "--output_dir",
        str(evaluation_dir),
    ]
    print("\nRunning:", " ".join(eval_command))
    subprocess.run(eval_command, check=True, cwd=repo_root)

    metadata = {
        "method": "ts2vec_style",
        "dataset_name": args.dataset_name,
        "seed": args.seed,
        "positive_pair_definition": "Two independently augmented views of the same unlabeled window.",
        "negative_pair_definition": "Other windows in the same batch.",
        "augmentation": {
            "random_crop_resize": True,
            "min_crop_ratio": args.min_crop_ratio,
            "noise_std": args.noise_std,
            "mask_ratio": args.mask_ratio,
            "scale_std": args.scale_std,
        },
        "normalization": args.normalization,
        "checkpoint": str(best_path),
        "evaluation": str(evaluation_dir / "recoverability_summary.json"),
    }
    (run_root / "ts2vec_style" / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print("\nTS2Vec-style recoverability run completed.")
    print(f"Checkpoint: {best_path}")
    print(f"Embeddings: {embedding_dir}")
    print(f"Evaluation: {evaluation_dir}")


if __name__ == "__main__":
    main()
