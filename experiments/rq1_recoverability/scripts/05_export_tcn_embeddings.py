"""RQ1 step 5: export embeddings from the trained TCN encoder."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export TCN embeddings for an RQ1 split bundle.")
    parser.add_argument("--train_file", type=str, required=True)
    parser.add_argument("--val_file", type=str, required=True)
    parser.add_argument("--test_file", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--python_exe", type=str, default=str(Path(__file__).resolve().parents[3] / "venv" / "Scripts" / "python.exe"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[3]
    extract_script = repo_root / "examples" / "fsss" / "extract_tcn_embeddings.py"

    command = [
        args.python_exe,
        str(extract_script),
        "--train_file",
        args.train_file,
        "--val_file",
        args.val_file,
        "--test_file",
        args.test_file,
        "--checkpoint",
        args.checkpoint,
        "--output_dir",
        args.output_dir,
        "--batch_size",
        str(args.batch_size),
        "--device",
        args.device,
    ]
    print("Running:", " ".join(command))
    subprocess.run(command, check=True, cwd=repo_root)


if __name__ == "__main__":
    main()
