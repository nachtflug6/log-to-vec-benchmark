"""
Visualize and sanity-check contrastive embeddings.

Outputs (saved under --out-dir):
- pca_2d.png
- tsne_2d.png
- pair_cosine_hist.png
- knn_report.txt

Usage:
python scripts/visualize_embeddings.py \
  --npz data/processed_features.npz \
  --emb checkpoints/test_embeddings.npy \
  --out-dir checkpoints/viz \
  --pair-mode neighbor \
  --max-points 2000 \
  --tsne-perplexity 30
"""

import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors


def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """L2 normalize rows of x."""
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    return x / (norm + eps)


def cosine_sim(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Cosine similarity per-row between a and b (same shape)."""
    a_n = a / (np.linalg.norm(a, axis=1, keepdims=True) + eps)
    b_n = b / (np.linalg.norm(b, axis=1, keepdims=True) + eps)
    return np.sum(a_n * b_n, axis=1)


def try_get_labels(npz: np.lib.npyio.NpzFile, n: int):
    """
    Try to auto-detect labels from NPZ keys.
    If no suitable labels found, return None.
    """
    # Common candidates you might store
    candidates = ["labels", "label", "y", "freq", "frequency", "class", "classes", "group", "groups"]
    for k in candidates:
        if k in npz.files:
            arr = npz[k]
            # Accept shape (N,) or (N,1)
            if arr.ndim == 2 and arr.shape[1] == 1:
                arr = arr.reshape(-1)
            if arr.ndim == 1 and len(arr) >= n:
                return arr[:n], k

    # If nothing found, return None
    return None, None


def downsample(x: np.ndarray, y: np.ndarray | None, max_points: int, seed: int = 42):
    """Downsample to at most max_points (for TSNE speed)."""
    n = x.shape[0]
    if n <= max_points:
        return x, y, np.arange(n)

    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=max_points, replace=False)
    idx.sort()
    x_ds = x[idx]
    y_ds = y[idx] if y is not None else None
    return x_ds, y_ds, idx



def plot_scatter_2d(points_2d: np.ndarray, labels: np.ndarray | None, title: str, out_path: Path):
    """
    Save a 2D scatter plot. If labels exist, color by labels.
    """
    plt.figure()
    if labels is None:
        plt.scatter(points_2d[:, 0], points_2d[:, 1], s=8, alpha=0.7)
    else:
        # Convert labels to categorical ids for coloring
        uniq = np.unique(labels)
        # Map to 0..K-1
        label_to_id = {v: i for i, v in enumerate(uniq)}
        ids = np.array([label_to_id[v] for v in labels], dtype=int)

        sc = plt.scatter(points_2d[:, 0], points_2d[:, 1], c=ids, s=8, alpha=0.7)
        plt.colorbar(sc)

    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_pair_cosine_hist(emb: np.ndarray, pair_mode: str, out_path: Path):
    """
    For neighbor mode:
      positive pairs are (i, i+1)
    For augment mode:
      positive pairs would be (view1, view2) which we don't have here,
      so we only plot neighbor-style by default.

    This histogram is a sanity check:
      - if contrastive is learning, cos(pos) tends to be higher.
      - if embeddings collapse, cos will be ~1 for all pairs (very narrow peak).
    """
    n = emb.shape[0]
    if n < 3:
        return

    # Use neighbor pairs as a consistent sanity-check.
    a = emb[:-1]
    b = emb[1:]
    pos_cos = cosine_sim(a, b)

    plt.figure()
    plt.hist(pos_cos, bins=50, alpha=0.9)
    plt.title(f"Cosine similarity of neighbor pairs (mode={pair_mode})")
    plt.xlabel("cosine(x_i, x_{i+1})")
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def write_knn_report(emb: np.ndarray, labels: np.ndarray | None, out_path: Path, k: int = 5, seed: int = 42):
    """
    Write a small kNN report for a few random anchors:
      - show nearest neighbor indices
      - if labels exist, show their labels
    """
    n = emb.shape[0]
    if n < k + 1:
        return

    emb_n = l2_normalize(emb)
    nn = NearestNeighbors(n_neighbors=k + 1, metric="cosine")
    nn.fit(emb_n)

    rng = np.random.default_rng(seed)
    anchors = rng.choice(n, size=min(10, n), replace=False)

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("kNN sanity-check report (cosine distance)\n")
        f.write(f"Total embeddings: {n}\n")
        f.write(f"k={k}\n\n")

        for a in anchors:
            dists, idxs = nn.kneighbors(emb_n[a:a + 1], return_distance=True)
            idxs = idxs[0]
            dists = dists[0]

            # idxs[0] should be itself (distance ~0)
            f.write(f"Anchor idx: {a}\n")
            if labels is not None:
                f.write(f"  Anchor label: {labels[a]}\n")

            for rank in range(1, len(idxs)):
                j = idxs[rank]
                f.write(f"  #{rank}: idx={j}, cosine_dist={dists[rank]:.6f}")
                if labels is not None:
                    f.write(f", label={labels[j]}")
                f.write("\n")

            f.write("\n")



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", type=str, required=True, help="Path to .npz containing sequences (and optional labels).")
    parser.add_argument("--emb", type=str, required=True, help="Path to embeddings .npy (e.g., test_embeddings.npy).")
    parser.add_argument("--out-dir", type=str, required=True, help="Output directory for figures and report.")
    parser.add_argument("--pair-mode", type=str, default="neighbor", choices=["neighbor", "augment"], help="For titles only.")
    parser.add_argument("--max-points", type=int, default=2000, help="Max points for t-SNE (downsample if larger).")
    parser.add_argument("--tsne-perplexity", type=int, default=30, help="t-SNE perplexity.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    npz = np.load(args.npz)
    emb = np.load(args.emb)

    if emb.ndim != 2:
        raise ValueError(f"Embeddings must be 2D (N, E). Got: {emb.shape}")

    n = emb.shape[0]
    labels, label_key = try_get_labels(npz, n)

    print(f"Loaded embeddings: {emb.shape}")
    if labels is None:
        print("No labels found in npz. Will plot without coloring.")
    else:
        print(f"Found labels from key '{label_key}', shape={labels.shape}")

    # PCA 2D
    pca = PCA(n_components=2, random_state=args.seed)
    pca_2d = pca.fit_transform(emb)
    plot_scatter_2d(
        pca_2d,
        labels,
        title=f"PCA 2D of embeddings (mode={args.pair_mode})",
        out_path=out_dir / "pca_2d.png",
    )

    # t-SNE 2D (downsample for speed)
    emb_ds, labels_ds, idx_ds = downsample(emb, labels, max_points=args.max_points, seed=args.seed)
    tsne = TSNE(
        n_components=2,
        perplexity=min(args.tsne_perplexity, max(5, (emb_ds.shape[0] - 1) // 3)),
        init="pca",
        learning_rate="auto",
        random_state=args.seed,
    )
    tsne_2d = tsne.fit_transform(emb_ds)
    plot_scatter_2d(
        tsne_2d,
        labels_ds,
        title=f"t-SNE 2D of embeddings (subset={emb_ds.shape[0]})",
        out_path=out_dir / "tsne_2d.png",
    )

    # Pair cosine histogram (neighbor sanity-check)
    plot_pair_cosine_hist(
        emb=emb,
        pair_mode=args.pair_mode,
        out_path=out_dir / "pair_cosine_hist.png",
    )

    # kNN report
    write_knn_report(
        emb=emb,
        labels=labels,
        out_path=out_dir / "knn_report.txt",
        k=5,
        seed=args.seed,
    )

    print(f"Saved outputs to: {out_dir}")


if __name__ == "__main__":
    main()