"""RQ1 step 1: generate a formal FRS dataset artifact."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rq1.generation.dataset_registry import build_dataset, register_dataset_artifact


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate an FRS dataset artifact for formal RQ1 runs.")
    parser.add_argument(
        "--dataset_name",
        type=str,
        required=True,
        choices=["frs_clean_vnext_long", "frs_noisy_vnext_long"],
    )
    parser.add_argument(
        "--artifact_root",
        type=str,
        default=str(Path(__file__).resolve().parents[1] / "artifacts" / "datasets"),
        help="Root directory for dataset artifacts.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_trajectories", type=int, default=120)
    parser.add_argument("--trajectory_length", type=int, default=320)
    parser.add_argument("--latent_dim", type=int, default=6)
    parser.add_argument("--num_channels", type=int, default=4)
    parser.add_argument("--window_length", type=int, default=48)
    parser.add_argument("--stride", type=int, default=12)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    artifact_dir = build_dataset(
        dataset_name=args.dataset_name,
        artifact_root=Path(args.artifact_root),
        seed=args.seed,
        num_trajectories=args.num_trajectories,
        trajectory_length=args.trajectory_length,
        latent_dim=args.latent_dim,
        num_channels=args.num_channels,
        window_length=args.window_length,
        stride=args.stride,
    )
    register_dataset_artifact(
        manifests_dir=Path(__file__).resolve().parents[1] / "manifests",
        dataset_name=args.dataset_name,
        artifact_dir=artifact_dir,
    )
    print(f"Prepared dataset artifact: {artifact_dir}")


if __name__ == "__main__":
    main()
