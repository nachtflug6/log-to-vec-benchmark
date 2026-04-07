from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from version2.data.fsss_dataset import FSSSWindowDataset
from version2.evaluation.eval_v2 import (
    load_embeddings,
    load_split,
    run_full_evaluation_suite,
    save_json,
)
from version2.models.tcn_hybrid import TCNHybridEncoder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run version2 FSSS evaluation suite (memo-aligned, no OOD).")
    parser.add_argument("--train_embeddings", type=str, required=True)
    parser.add_argument("--val_embeddings", type=str, required=True)
    parser.add_argument("--test_embeddings", type=str, required=True)

    parser.add_argument("--train_split", type=str, required=True)
    parser.add_argument("--val_split", type=str, required=True)
    parser.add_argument("--test_split", type=str, required=True)

    parser.add_argument("--output_dir", type=str, required=True)

    # Optional for robustness evaluation
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--model_type", type=str, default="tcn_hybrid", choices=["tcn_hybrid"])
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda")

    return parser.parse_args()


def build_encoder_fn_from_checkpoint(
    checkpoint_path: str,
    device: torch.device,
):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)

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

    @torch.no_grad()
    def encoder_fn(X_np: np.ndarray) -> np.ndarray:
        x = torch.from_numpy(X_np).to(device=device, dtype=torch.float32)
        out = model(x)
        return out["embeddings"].detach().cpu().numpy().astype(np.float32)

    return encoder_fn


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_embeddings = load_embeddings(args.train_embeddings)
    val_embeddings = load_embeddings(args.val_embeddings)
    test_embeddings = load_embeddings(args.test_embeddings)

    train_split = load_split(args.train_split)
    val_split = load_split(args.val_split)
    test_split = load_split(args.test_split)

    encoder_fn = None
    if args.checkpoint is not None:
        device = torch.device(args.device if torch.cuda.is_available() else "cpu")
        encoder_fn = build_encoder_fn_from_checkpoint(args.checkpoint, device)

    summary = run_full_evaluation_suite(
        train_embeddings=train_embeddings,
        val_embeddings=val_embeddings,
        test_embeddings=test_embeddings,
        train_split=train_split,
        val_split=val_split,
        test_split=test_split,
        encoder_fn=encoder_fn,
        visualization_dir=output_dir / "figures",
    )

    save_json(output_dir / "evaluation_suite_summary.json", summary)
    print(f"Saved evaluation summary to: {output_dir / 'evaluation_suite_summary.json'}")
    print(f"Saved figures to: {output_dir / 'figures'}")


if __name__ == "__main__":
    main()