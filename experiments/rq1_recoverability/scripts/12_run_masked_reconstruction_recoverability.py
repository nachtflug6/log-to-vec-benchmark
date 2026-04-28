"""Train a masked time-series reconstruction encoder and evaluate recoverability.

This method represents the reconstruction family in RQ1/RQ2. It masks a subset
of temporal patches, reconstructs only the masked patches, and exports the
encoder representation as the downstream embedding.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import subprocess

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


class PatchEmbed(nn.Module):
    def __init__(self, input_dim: int, patch_size: int, hidden_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.patch_size = patch_size
        self.patch_dim = input_dim * patch_size
        self.proj = nn.Linear(self.patch_dim, hidden_dim)

    def patchify(self, x_btc: torch.Tensor) -> torch.Tensor:
        batch, length, channels = x_btc.shape
        if channels != self.input_dim:
            raise ValueError(f"Expected input_dim={self.input_dim}, got {channels}")
        if length % self.patch_size != 0:
            raise ValueError(
                f"Window length {length} must be divisible by patch_size {self.patch_size}. "
                "Use a compatible patch size for the active dataset."
            )
        num_patches = length // self.patch_size
        return x_btc.reshape(batch, num_patches, self.patch_dim)

    def forward(self, x_btc: torch.Tensor) -> torch.Tensor:
        patches = self.patchify(x_btc)
        return self.proj(patches)


class TransformerBlock(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, mlp_ratio: float, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(hidden_dim)
        mlp_hidden = int(hidden_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


class MaskedPatchReconstructionEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        window_length: int,
        patch_size: int,
        hidden_dim: int,
        embedding_dim: int,
        decoder_dim: int,
        encoder_depth: int,
        decoder_depth: int,
        num_heads: int,
        dropout: float,
    ):
        super().__init__()
        if window_length % patch_size != 0:
            raise ValueError(
                f"window_length={window_length} must be divisible by patch_size={patch_size}"
            )
        self.patch_embed = PatchEmbed(input_dim=input_dim, patch_size=patch_size, hidden_dim=hidden_dim)
        self.num_patches = window_length // patch_size
        self.patch_dim = input_dim * patch_size

        self.encoder_pos = nn.Parameter(torch.zeros(1, self.num_patches, hidden_dim))
        self.encoder_blocks = nn.ModuleList(
            [TransformerBlock(hidden_dim, num_heads, mlp_ratio=4.0, dropout=dropout) for _ in range(encoder_depth)]
        )
        self.encoder_norm = nn.LayerNorm(hidden_dim)
        self.embedding_head = nn.Sequential(
            nn.Linear(hidden_dim, embedding_dim),
            nn.GELU(),
            nn.LayerNorm(embedding_dim),
        )

        self.decoder_input = nn.Linear(hidden_dim, decoder_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))
        self.decoder_pos = nn.Parameter(torch.zeros(1, self.num_patches, decoder_dim))
        self.decoder_blocks = nn.ModuleList(
            [TransformerBlock(decoder_dim, num_heads, mlp_ratio=4.0, dropout=dropout) for _ in range(decoder_depth)]
        )
        self.decoder_norm = nn.LayerNorm(decoder_dim)
        self.reconstruction_head = nn.Linear(decoder_dim, self.patch_dim)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.trunc_normal_(self.encoder_pos, std=0.02)
        nn.init.trunc_normal_(self.decoder_pos, std=0.02)
        nn.init.trunc_normal_(self.mask_token, std=0.02)

    def sample_patch_mask(self, batch_size: int, mask_ratio: float, device: torch.device) -> torch.Tensor:
        num_mask = max(1, int(round(self.num_patches * mask_ratio)))
        num_mask = min(num_mask, self.num_patches - 1)
        noise = torch.rand(batch_size, self.num_patches, device=device)
        order = torch.argsort(noise, dim=1)
        mask = torch.zeros(batch_size, self.num_patches, dtype=torch.bool, device=device)
        mask.scatter_(1, order[:, :num_mask], True)
        return mask

    def encode_tokens(self, x_btc: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        patches = self.patch_embed.patchify(x_btc)
        tokens = self.patch_embed.proj(patches) + self.encoder_pos
        for block in self.encoder_blocks:
            tokens = block(tokens)
        tokens = self.encoder_norm(tokens)
        return patches, tokens

    def reconstruct_masked(self, x_btc: torch.Tensor, mask_ratio: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        patches, encoder_tokens = self.encode_tokens(x_btc)
        batch, _, _ = encoder_tokens.shape
        device = x_btc.device
        patch_mask = self.sample_patch_mask(batch, mask_ratio=mask_ratio, device=device)

        decoder_tokens = self.decoder_input(encoder_tokens)
        mask_tokens = self.mask_token.expand(batch, self.num_patches, -1)
        decoder_tokens = torch.where(patch_mask.unsqueeze(-1), mask_tokens, decoder_tokens)
        decoder_tokens = decoder_tokens + self.decoder_pos
        for block in self.decoder_blocks:
            decoder_tokens = block(decoder_tokens)
        decoder_tokens = self.decoder_norm(decoder_tokens)
        reconstructed = self.reconstruction_head(decoder_tokens)
        return reconstructed, patches, patch_mask

    def encode(self, x_btc: torch.Tensor) -> torch.Tensor:
        _, tokens = self.encode_tokens(x_btc)
        pooled = tokens.mean(dim=1)
        return self.embedding_head(pooled)


def masked_reconstruction_loss(
    reconstructed_patches: torch.Tensor,
    target_patches: torch.Tensor,
    patch_mask: torch.Tensor,
) -> torch.Tensor:
    squared_error = (reconstructed_patches - target_patches) ** 2
    masked_error = squared_error * patch_mask.unsqueeze(-1)
    denom = patch_mask.sum().clamp_min(1) * target_patches.shape[-1]
    return masked_error.sum() / denom


def run_epoch(
    model: MaskedPatchReconstructionEncoder,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    mask_ratio: float,
) -> float:
    training = optimizer is not None
    model.train(training)
    losses: list[float] = []

    for batch in loader:
        x = batch.to(device)
        reconstructed, target_patches, patch_mask = model.reconstruct_masked(x, mask_ratio=mask_ratio)
        loss = masked_reconstruction_loss(reconstructed, target_patches, patch_mask)

        if training:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

        losses.append(float(loss.detach().cpu()))

    return float(np.mean(losses)) if losses else float("nan")


@torch.no_grad()
def export_embeddings(model: MaskedPatchReconstructionEncoder, loader: DataLoader, device: torch.device) -> np.ndarray:
    model.eval()
    chunks: list[torch.Tensor] = []
    for batch in loader:
        chunks.append(model.encode(batch.to(device)).detach().cpu())
    return torch.cat(chunks, dim=0).numpy().astype(np.float32)


def save_embeddings(path: Path, embeddings: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, embeddings=embeddings)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and evaluate a masked reconstruction encoder on an RQ1 dataset split."
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="frs_clean_vnext_long",
        choices=["frs_clean_vnext_long", "frs_noisy_vnext_long"],
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--patch_size", type=int, default=4)
    parser.add_argument("--hidden_dim", type=int, default=192)
    parser.add_argument("--embedding_dim", type=int, default=192)
    parser.add_argument("--decoder_dim", type=int, default=128)
    parser.add_argument("--encoder_depth", type=int, default=4)
    parser.add_argument("--decoder_depth", type=int, default=2)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--mask_ratio", type=float, default=0.40)
    parser.add_argument("--normalization", choices=["train", "none"], default="train")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--python_exe",
        type=str,
        default=str(Path(__file__).resolve().parents[3] / "venv" / "Scripts" / "python.exe"),
    )
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

    run_name = args.run_name or f"{args.dataset_name}_masked_recon"
    run_root = root / "artifacts" / "runs" / run_name
    checkpoint_dir = run_root / "masked_reconstruction" / "training"
    embedding_dir = run_root / "masked_reconstruction" / "embeddings"
    evaluation_dir = run_root / "masked_reconstruction" / "evaluation"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    mean = std = None
    if args.normalization == "train":
        mean, std = compute_train_stats(train_file)

    train_loader = make_loader(train_file, args.batch_size, shuffle=True, mean=mean, std=std)
    val_loader = make_loader(val_file, args.batch_size, shuffle=False, mean=mean, std=std)
    train_export_loader = make_loader(train_file, args.batch_size, shuffle=False, mean=mean, std=std)
    test_loader = make_loader(test_file, args.batch_size, shuffle=False, mean=mean, std=std)

    with np.load(train_file) as data:
        window_length = int(data["X"].shape[1])
        input_dim = int(data["X"].shape[2])

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = MaskedPatchReconstructionEncoder(
        input_dim=input_dim,
        window_length=window_length,
        patch_size=args.patch_size,
        hidden_dim=args.hidden_dim,
        embedding_dim=args.embedding_dim,
        decoder_dim=args.decoder_dim,
        encoder_depth=args.encoder_depth,
        decoder_depth=args.decoder_depth,
        num_heads=args.num_heads,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val = float("inf")
    history: list[dict[str, float]] = []
    best_path = checkpoint_dir / "best_model.pt"
    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(model, train_loader, optimizer, device, args.mask_ratio)
        val_loss = run_epoch(model, val_loader, None, device, args.mask_ratio)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "args": vars(args),
                    "window_length": window_length,
                    "input_dim": input_dim,
                    "mean": mean,
                    "std": std,
                },
                best_path,
            )
        print(f"epoch={epoch:03d} train_loss={train_loss:.6f} val_loss={val_loss:.6f}")

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
        "method": "masked_reconstruction",
        "dataset_name": args.dataset_name,
        "seed": args.seed,
        "objective": "Reconstruct masked temporal patches only.",
        "mask_ratio": args.mask_ratio,
        "patch_size": args.patch_size,
        "normalization": args.normalization,
        "checkpoint": str(best_path),
        "evaluation": str(evaluation_dir / "recoverability_summary.json"),
    }
    (run_root / "masked_reconstruction" / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print("\nMasked reconstruction recoverability run completed.")
    print(f"Checkpoint: {best_path}")
    print(f"Embeddings: {embedding_dir}")
    print(f"Evaluation: {evaluation_dir}")


if __name__ == "__main__":
    main()
