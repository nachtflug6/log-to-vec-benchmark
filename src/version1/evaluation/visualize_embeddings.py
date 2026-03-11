"""
Visualize extracted sequence embeddings using PCA and t-SNE.

This script:
1. Loads saved train / val / test embedding files
2. Reduces embeddings to 2D with PCA and t-SNE
3. Creates separate plots for each split
4. Creates combined plots across splits
5. Saves all figures for later inspection and reporting
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


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


def reduce_with_pca(embeddings: np.ndarray, n_components: int = 2) -> np.ndarray:
    """
    Reduce embeddings with PCA.
    """
    pca = PCA(n_components=n_components)
    reduced = pca.fit_transform(embeddings)
    return reduced.astype(np.float32)


def reduce_with_tsne(
    embeddings: np.ndarray,
    n_components: int = 2,
    perplexity: float = 30.0,
    random_state: int = 42
) -> np.ndarray:
    """
    Reduce embeddings with t-SNE.
    """
    tsne = TSNE(
        n_components=n_components,
        perplexity=perplexity,
        random_state=random_state,
        init="pca",
        learning_rate="auto"
    )
    reduced = tsne.fit_transform(embeddings)
    return reduced.astype(np.float32)


def plot_2d_embeddings(
    points: np.ndarray,
    title: str,
    output_path: Path,
    labels: np.ndarray | None = None
) -> None:
    """
    Save a 2D scatter plot.

    Args:
        points: [N, 2]
        title: figure title
        output_path: image output path
        labels: optional integer labels for coloring
    """
    plt.figure(figsize=(8, 6))

    if labels is None:
        plt.scatter(points[:, 0], points[:, 1], s=10, alpha=0.7)
    else:
        unique_labels = np.unique(labels)
        for label in unique_labels:
            mask = labels == label
            plt.scatter(
                points[mask, 0],
                points[mask, 1],
                s=10,
                alpha=0.7,
                label=str(label)
            )
        plt.legend()

    plt.title(title)
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def build_split_labels(train_emb, val_emb, test_emb):
    """
    Build integer labels for combined visualization.

    0 -> train
    1 -> val
    2 -> test
    """
    train_labels = np.zeros(len(train_emb), dtype=np.int64)
    val_labels = np.ones(len(val_emb), dtype=np.int64)
    test_labels = np.full(len(test_emb), 2, dtype=np.int64)

    labels = np.concatenate([train_labels, val_labels, test_labels], axis=0)
    return labels


def main():
    parser = argparse.ArgumentParser(
        description="Visualize extracted embeddings with PCA and t-SNE"
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
        default="outputs/version1/visualization"
    )
    parser.add_argument(
        "--tsne_perplexity",
        type=float,
        default=30.0
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=42
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading embeddings...")

    train_emb = load_embeddings(args.train_embeddings)
    val_emb = load_embeddings(args.val_embeddings)
    test_emb = load_embeddings(args.test_embeddings)

    print(f"Train embeddings: {train_emb.shape}")
    print(f"Val embeddings:   {val_emb.shape}")
    print(f"Test embeddings:  {test_emb.shape}")

    # -------------------------------------------------------------------------
    # Separate PCA plots
    # -------------------------------------------------------------------------
    print("\nCreating PCA plots for each split...")

    train_pca = reduce_with_pca(train_emb)
    val_pca = reduce_with_pca(val_emb)
    test_pca = reduce_with_pca(test_emb)

    plot_2d_embeddings(
        train_pca,
        "Train Embeddings - PCA",
        output_dir / "train_pca.png"
    )
    plot_2d_embeddings(
        val_pca,
        "Val Embeddings - PCA",
        output_dir / "val_pca.png"
    )
    plot_2d_embeddings(
        test_pca,
        "Test Embeddings - PCA",
        output_dir / "test_pca.png"
    )

    # -------------------------------------------------------------------------
    # Separate t-SNE plots
    # -------------------------------------------------------------------------
    print("Creating t-SNE plots for each split...")

    train_tsne = reduce_with_tsne(
        train_emb,
        perplexity=args.tsne_perplexity,
        random_state=args.random_state
    )
    val_tsne = reduce_with_tsne(
        val_emb,
        perplexity=args.tsne_perplexity,
        random_state=args.random_state
    )
    test_tsne = reduce_with_tsne(
        test_emb,
        perplexity=args.tsne_perplexity,
        random_state=args.random_state
    )

    plot_2d_embeddings(
        train_tsne,
        "Train Embeddings - t-SNE",
        output_dir / "train_tsne.png"
    )
    plot_2d_embeddings(
        val_tsne,
        "Val Embeddings - t-SNE",
        output_dir / "val_tsne.png"
    )
    plot_2d_embeddings(
        test_tsne,
        "Test Embeddings - t-SNE",
        output_dir / "test_tsne.png"
    )

    # -------------------------------------------------------------------------
    # Combined PCA / t-SNE plots
    # -------------------------------------------------------------------------
    print("Creating combined plots across splits...")

    combined_emb = np.concatenate([train_emb, val_emb, test_emb], axis=0)
    split_labels = build_split_labels(train_emb, val_emb, test_emb)

    combined_pca = reduce_with_pca(combined_emb)
    plot_2d_embeddings(
        combined_pca,
        "Combined Embeddings - PCA (train/val/test)",
        output_dir / "combined_pca.png",
        labels=split_labels
    )

    combined_tsne = reduce_with_tsne(
        combined_emb,
        perplexity=args.tsne_perplexity,
        random_state=args.random_state
    )
    plot_2d_embeddings(
        combined_tsne,
        "Combined Embeddings - t-SNE (train/val/test)",
        output_dir / "combined_tsne.png",
        labels=split_labels
    )

    print("\nSaved visualization files to:")
    print(output_dir)
    print("\nDone.")


if __name__ == "__main__":
    main()
