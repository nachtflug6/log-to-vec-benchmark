"""RQ1 step 6: run the unified latent-factor recoverability evaluation."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rq1.evaluation.eval_v2 import (
    load_embeddings,
    load_split,
    run_full_evaluation_suite,
    save_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate precomputed embeddings with the unified RQ1 protocol.")
    parser.add_argument("--train_embeddings", type=str, required=True)
    parser.add_argument("--val_embeddings", type=str, required=True)
    parser.add_argument("--test_embeddings", type=str, required=True)
    parser.add_argument("--train_split", type=str, required=True)
    parser.add_argument("--val_split", type=str, required=True)
    parser.add_argument("--test_split", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument(
        "--skip_visualizations",
        action="store_true",
        help="Disable PCA/t-SNE embedding plots.",
    )
    parser.add_argument(
        "--visualization_max_points",
        type=int,
        default=1500,
        help="Maximum number of test samples used for embedding plots.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    visualization_dir = None if args.skip_visualizations else (output_dir / "visualizations")

    summary = run_full_evaluation_suite(
        train_embeddings=load_embeddings(args.train_embeddings),
        val_embeddings=load_embeddings(args.val_embeddings),
        test_embeddings=load_embeddings(args.test_embeddings),
        train_split=load_split(args.train_split),
        val_split=load_split(args.val_split),
        test_split=load_split(args.test_split),
        visualization_dir=visualization_dir,
        visualization_max_points=args.visualization_max_points,
    )
    save_json(output_dir / "recoverability_summary.json", summary)
    print(f"Saved evaluation summary to: {output_dir / 'recoverability_summary.json'}")
    if visualization_dir is not None:
        print(f"Saved embedding plots to: {visualization_dir}")


if __name__ == "__main__":
    main()
