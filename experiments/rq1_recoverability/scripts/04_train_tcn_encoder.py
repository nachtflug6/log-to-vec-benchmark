"""RQ1 step 4: train the optional legacy TCN encoder."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the TCN encoder for an RQ1 split bundle.")
    parser.add_argument("--train_file", type=str, required=True)
    parser.add_argument("--val_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--python_exe", type=str, default=str(Path(__file__).resolve().parents[3] / "venv" / "Scripts" / "python.exe"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    repo_root = Path(__file__).resolve().parents[3]
    train_script = root / "src" / "rq1" / "models" / "train_tcn_hybrid.py"

    if not train_script.exists():
        raise FileNotFoundError(
            f"Missing local TCN training script: {train_script}. "
            "If you need the legacy TCN baseline, copy the original train_tcn_hybrid.py here."
        )

    command = [
        args.python_exe,
        str(train_script),
        "--train_file",
        args.train_file,
        "--val_file",
        args.val_file,
        "--output_dir",
        args.output_dir,
        "--epochs",
        str(args.epochs),
        "--batch_size",
        str(args.batch_size),
        "--device",
        args.device,
        "--seed",
        str(args.seed),
    ]
    print("Running:", " ".join(command))
    subprocess.run(command, check=True, cwd=repo_root)


if __name__ == "__main__":
    main()
