"""RQ1 step 3: compute baseline representations and baseline metrics."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run baseline probes for an RQ1 split bundle.")
    parser.add_argument("--train_file", type=str, required=True)
    parser.add_argument("--test_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument(
        "--feature_sets",
        nargs="+",
        default=["fft", "summary", "raw_flatten"],
        choices=["fft", "summary", "raw_flatten"],
    )
    parser.add_argument(
        "--probe_models",
        nargs="+",
        default=["linear", "rbf"],
        choices=["linear", "rbf"],
    )
    parser.add_argument("--python_exe", type=str, default=str(Path(__file__).resolve().parents[3] / "venv" / "Scripts" / "python.exe"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[3]
    baseline_script = repo_root / "examples" / "fsss" / "run_baseline_probes.py"

    command = [
        args.python_exe,
        str(baseline_script),
        "--train_file",
        args.train_file,
        "--test_file",
        args.test_file,
        "--output_dir",
        args.output_dir,
        "--feature_sets",
        *args.feature_sets,
        "--probe_models",
        *args.probe_models,
    ]
    print("Running:", " ".join(command))
    subprocess.run(command, check=True, cwd=repo_root)


if __name__ == "__main__":
    main()
