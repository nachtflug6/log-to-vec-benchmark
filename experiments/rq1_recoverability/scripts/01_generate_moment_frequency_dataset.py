"""RQ1 step 1: generate the moment frequency sanity-check dataset."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from rq1.dataset_registry import build_dataset, register_dataset_artifact


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate the moment frequency dataset artifact for RQ1.")
    parser.add_argument(
        "--artifact_root",
        type=str,
        default=str(Path(__file__).resolve().parents[1] / "artifacts" / "datasets"),
        help="Root directory for dataset artifacts.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_samples", type=int, default=1024)
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--noise_std", type=float, default=0.1)
    parser.add_argument("--num_bins", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    artifact_dir = build_dataset(
        dataset_name="moment_freq",
        artifact_root=Path(args.artifact_root),
        seed=args.seed,
        n_samples=args.n_samples,
        seq_len=args.seq_len,
        noise_std=args.noise_std,
        num_bins=args.num_bins,
    )
    register_dataset_artifact(
        manifests_dir=Path(__file__).resolve().parents[1] / "manifests",
        dataset_name="moment_freq",
        artifact_dir=artifact_dir,
    )
    print(f"Prepared dataset artifact: {artifact_dir}")


if __name__ == "__main__":
    main()
