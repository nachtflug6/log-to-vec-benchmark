#!/usr/bin/env python3
"""Create and run a tiny contrastive-training smoke workload."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np


def build_sequences(
    num_sequences: int,
    sequence_length: int,
    num_features: int,
    seed: int,
) -> np.ndarray:
    """Build a deterministic toy sequence tensor with a learnable signal."""
    rng = np.random.default_rng(seed)
    time = np.linspace(0.0, 2.0 * np.pi, sequence_length, dtype=np.float32)
    sequences = np.empty((num_sequences, sequence_length, num_features), dtype=np.float32)

    for idx in range(num_sequences):
        phase = idx * 0.07
        freq = 1.0 + (idx % 4) * 0.15
        signal = np.sin(freq * time + phase)
        trend = np.linspace(0.0, idx / max(1, num_sequences - 1), sequence_length)
        noise = rng.normal(0.0, 0.015, size=(sequence_length, num_features))

        for feature_idx in range(num_features):
            scale = 1.0 + feature_idx * 0.2
            offset = (feature_idx - num_features / 2.0) * 0.05
            sequences[idx, :, feature_idx] = scale * signal + offset + 0.2 * trend

        sequences[idx] += noise.astype(np.float32)

    return sequences


def write_training_config(
    config_path: Path,
    npz_path: Path,
    checkpoint_dir: Path,
    num_epochs: int,
    batch_size: int,
    seed: int,
    device: str,
) -> Dict[str, Any]:
    """Write a JSON-as-YAML config consumed by train_contrastive_toy.py."""
    config: Dict[str, Any] = {
        "data": {
            "npz_file": str(npz_path),
            "batch_size": batch_size,
            "num_workers": 0,
            "pair_mode": "neighbor",
            "normalize": False,
            "return_meta": False,
            "train_ratio": 0.7,
            "val_ratio": 0.15,
            "shuffle_split": True,
        },
        "model": {
            "hidden_dim": 16,
            "num_layers": 1,
            "projection_dim": 8,
            "dropout": 0.0,
        },
        "training": {
            "seed": seed,
            "device": device,
            "num_epochs": num_epochs,
            "learning_rate": 0.001,
            "weight_decay": 0.0,
            "temperature": 0.1,
        },
        "logging": {
            "checkpoint_dir": str(checkpoint_dir),
        },
        "evaluation": {
            "eval_every_n_epochs": max(1, num_epochs),
            "uniformity_t": 2.0,
            "uniformity_max_samples": 128,
        },
    }
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    return config


def prepare_workload(args: argparse.Namespace) -> Dict[str, Any]:
    run_dir = args.output_dir.resolve()
    data_dir = run_dir / "data"
    checkpoint_dir = run_dir / "checkpoints"
    data_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    npz_path = data_dir / "smoke_sequences.npz"
    config_path = run_dir / "smoke_contrastive_config.yaml"
    sequences = build_sequences(
        num_sequences=args.num_sequences,
        sequence_length=args.sequence_length,
        num_features=args.num_features,
        seed=args.seed,
    )
    np.savez(
        npz_path,
        sequences=sequences,
        run_id=np.array(args.run_id),
        seed=np.array(args.seed),
    )
    config = write_training_config(
        config_path=config_path,
        npz_path=npz_path,
        checkpoint_dir=checkpoint_dir,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        seed=args.seed,
        device=args.device,
    )

    return {
        "run_id": args.run_id,
        "run_dir": str(run_dir),
        "npz_path": str(npz_path),
        "config_path": str(config_path),
        "checkpoint_dir": str(checkpoint_dir),
        "sequence_shape": list(sequences.shape),
        "config": config,
        "best_model_path": str(checkpoint_dir / "best_model.pt"),
        "embeddings_path": str(checkpoint_dir / "test_embeddings.npy"),
    }


def run_training(repo_root: Path, config_path: Path) -> None:
    command = [
        sys.executable,
        str(repo_root / "examples" / "train_contrastive_toy.py"),
        "--config",
        str(config_path),
    ]
    subprocess.run(command, cwd=str(repo_root), check=True)


def write_summary(summary_path: Path, payload: Dict[str, Any]) -> None:
    summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run an Alvis model-deploy smoke test.")
    parser.add_argument("--run-id", default="ALVIS-SMOKE")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--num-sequences", type=int, default=64)
    parser.add_argument("--sequence-length", type=int, default=12)
    parser.add_argument("--num-features", type=int, default=4)
    parser.add_argument("--num-epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Create the NPZ/config/summary but skip model training.",
    )
    args = parser.parse_args(argv)

    if args.num_sequences < 16:
        parser.error("--num-sequences must be at least 16 for train/val/test splits")
    if args.sequence_length < 2:
        parser.error("--sequence-length must be at least 2")
    if args.num_features < 1:
        parser.error("--num-features must be at least 1")
    if args.num_epochs < 1:
        parser.error("--num-epochs must be at least 1")

    repo_root = Path(args.repo_root).resolve()
    if not (repo_root / "examples" / "train_contrastive_toy.py").exists():
        parser.error(f"repo root does not contain examples/train_contrastive_toy.py: {repo_root}")

    workload = prepare_workload(args)
    summary_path = Path(workload["run_dir"]) / "smoke_summary.json"

    summary: Dict[str, Any] = {
        **{key: value for key, value in workload.items() if key != "config"},
        "repo_root": str(repo_root),
        "dry_run": args.dry_run,
        "training_command": [
            sys.executable,
            str(repo_root / "examples" / "train_contrastive_toy.py"),
            "--config",
            workload["config_path"],
        ],
        "artifacts": {
            "best_model_exists": False,
            "embeddings_exists": False,
        },
    }

    if not args.dry_run:
        run_training(repo_root=repo_root, config_path=Path(workload["config_path"]))
        best_model_path = Path(workload["best_model_path"])
        embeddings_path = Path(workload["embeddings_path"])
        summary["artifacts"] = {
            "best_model_exists": best_model_path.exists(),
            "embeddings_exists": embeddings_path.exists(),
        }
        write_summary(summary_path, summary)

        missing = [
            path
            for path in (best_model_path, embeddings_path)
            if not path.exists()
        ]
        if missing:
            missing_text = ", ".join(str(path) for path in missing)
            raise FileNotFoundError(f"smoke training did not create expected artifacts: {missing_text}")
    else:
        write_summary(summary_path, summary)

    print(f"Smoke summary written to {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
