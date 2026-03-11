"""
Evaluate extracted sequence embeddings from train / val / test splits.

This script performs simple representation-level checks, including:
1. Basic shape and summary statistics
2. Embedding norm statistics
3. Per-dimension variance statistics
4. Collapse-related checks
5. Distribution gap between different splits

"""

import argparse
import json
from pathlib import Path

import numpy as np


def load_embeddings(npz_path: str) -> np.ndarray:
    """
    Load embeddings from a saved .npz file.

    Expected format:
        embeddings: [N, D]
    """
    data = np.load(npz_path)

    if "embeddings" not in data:
        raise ValueError(f"'embeddings' not found in {npz_path}")

    embeddings = data["embeddings"].astype(np.float32)

    if embeddings.ndim != 2:
        raise ValueError(
            f"Expected embeddings to have shape [N, D], got {embeddings.shape}"
        )

    return embeddings


def compute_basic_stats(embeddings: np.ndarray) -> dict:
    """
    Compute basic statistics for one embedding matrix.

    Args:
        embeddings: [N, D]

    Returns:
        Dictionary of summary statistics.
    """
    norms = np.linalg.norm(embeddings, axis=1)
    dim_std = embeddings.std(axis=0)
    dim_var = embeddings.var(axis=0)

    stats = {
        "num_samples": int(embeddings.shape[0]),
        "embedding_dim": int(embeddings.shape[1]),
        "global_mean": float(embeddings.mean()),
        "global_std": float(embeddings.std()),
        "mean_l2_norm": float(norms.mean()),
        "std_l2_norm": float(norms.std()),
        "min_l2_norm": float(norms.min()),
        "max_l2_norm": float(norms.max()),
        "mean_dim_std": float(dim_std.mean()),
        "min_dim_std": float(dim_std.min()),
        "max_dim_std": float(dim_std.max()),
        "mean_dim_var": float(dim_var.mean()),
        "min_dim_var": float(dim_var.min()),
        "max_dim_var": float(dim_var.max()),
    }

    return stats


def compute_collapse_metrics(embeddings: np.ndarray, variance_threshold: float = 1e-4) -> dict:
    """
    Compute simple collapse-related diagnostics.

    Args:
        embeddings: [N, D]
        variance_threshold: threshold below which a dimension is considered near-collapsed

    Returns:
        Dictionary of collapse diagnostics.
    """
    dim_var = embeddings.var(axis=0)
    num_low_variance_dims = int((dim_var < variance_threshold).sum())
    frac_low_variance_dims = float(num_low_variance_dims / len(dim_var))

    sample_std = embeddings.std(axis=0).mean()
    overall_std = embeddings.std()

    return {
        "num_low_variance_dims": num_low_variance_dims,
        "frac_low_variance_dims": frac_low_variance_dims,
        "mean_feature_std": float(sample_std),
        "overall_std": float(overall_std),
        "is_likely_collapsed": bool(overall_std < variance_threshold),
    }


def compute_pairwise_split_gap(a: np.ndarray, b: np.ndarray) -> dict:
    """
    Compare the overall distribution of two embedding sets.

    This is not a formal statistical test. It gives simple checks:
    - mean vector distance
    - std vector distance
    - norm mean difference

    Args:
        a: [N1, D]
        b: [N2, D]

    Returns:
        Dictionary of split-gap statistics.
    """
    mean_a = a.mean(axis=0)
    mean_b = b.mean(axis=0)

    std_a = a.std(axis=0)
    std_b = b.std(axis=0)

    norm_a = np.linalg.norm(a, axis=1)
    norm_b = np.linalg.norm(b, axis=1)

    return {
        "mean_vector_l2_gap": float(np.linalg.norm(mean_a - mean_b)),
        "std_vector_l2_gap": float(np.linalg.norm(std_a - std_b)),
        "mean_norm_gap": float(abs(norm_a.mean() - norm_b.mean())),
        "std_norm_gap": float(abs(norm_a.std() - norm_b.std())),
    }


def pretty_print_stats(name: str, stats: dict) -> None:
    """
    Print stats in a readable format.
    """
    print("\n" + "=" * 70)
    print(name)
    print("=" * 70)

    for key, value in stats.items():
        if isinstance(value, float):
            print(f"{key:28s}: {value:.6f}")
        else:
            print(f"{key:28s}: {value}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate extracted embeddings for version1"
    )

    parser.add_argument(
        "--train_embeddings",
        type=str,
        default="data/embeddings/version1/contrastive/train_embeddings.npz"
    )
    parser.add_argument(
        "--val_embeddings",
        type=str,
        default="data/embeddings/version1/contrastive/val_embeddings.npz"
    )
    parser.add_argument(
        "--test_embeddings",
        type=str,
        default="data/embeddings/version1/contrastive/test_embeddings.npz"
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/version1/evaluation"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # Load embeddings
    # -------------------------------------------------------------------------
    print("Loading embedding files...")

    train_emb = load_embeddings(args.train_embeddings)
    val_emb = load_embeddings(args.val_embeddings)
    test_emb = load_embeddings(args.test_embeddings)

    print(f"Train embeddings: {train_emb.shape}")
    print(f"Val embeddings:   {val_emb.shape}")
    print(f"Test embeddings:  {test_emb.shape}")

    # -------------------------------------------------------------------------
    # Per-split statistics
    # -------------------------------------------------------------------------
    train_basic = compute_basic_stats(train_emb)
    val_basic = compute_basic_stats(val_emb)
    test_basic = compute_basic_stats(test_emb)

    train_collapse = compute_collapse_metrics(train_emb)
    val_collapse = compute_collapse_metrics(val_emb)
    test_collapse = compute_collapse_metrics(test_emb)

    pretty_print_stats("TRAIN BASIC STATS", train_basic)
    pretty_print_stats("VAL BASIC STATS", val_basic)
    pretty_print_stats("TEST BASIC STATS", test_basic)

    pretty_print_stats("TRAIN COLLAPSE CHECK", train_collapse)
    pretty_print_stats("VAL COLLAPSE CHECK", val_collapse)
    pretty_print_stats("TEST COLLAPSE CHECK", test_collapse)

    # -------------------------------------------------------------------------
    # Split-to-split comparisons
    # -------------------------------------------------------------------------
    train_val_gap = compute_pairwise_split_gap(train_emb, val_emb)
    val_test_gap = compute_pairwise_split_gap(val_emb, test_emb)
    train_test_gap = compute_pairwise_split_gap(train_emb, test_emb)

    pretty_print_stats("TRAIN vs VAL GAP", train_val_gap)
    pretty_print_stats("VAL vs TEST GAP", val_test_gap)
    pretty_print_stats("TRAIN vs TEST GAP", train_test_gap)

    # -------------------------------------------------------------------------
    # Save evaluation summary
    # -------------------------------------------------------------------------
    summary = {
        "train_basic": train_basic,
        "val_basic": val_basic,
        "test_basic": test_basic,
        "train_collapse": train_collapse,
        "val_collapse": val_collapse,
        "test_collapse": test_collapse,
        "train_val_gap": train_val_gap,
        "val_test_gap": val_test_gap,
        "train_test_gap": train_test_gap,
    }

    summary_path = output_dir / "embedding_evaluation_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\nSaved evaluation summary to:")
    print(summary_path)

    print("\nDone.")


if __name__ == "__main__":
    main()