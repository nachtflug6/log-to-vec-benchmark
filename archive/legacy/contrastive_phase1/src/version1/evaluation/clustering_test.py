"""
Evaluate embedding quality with a clustering-based test.

This script:
1. Loads extracted embeddings
2. Loads the corresponding processed sequences
3. Builds simple sequence labels
4. Runs KMeans clustering on embeddings
5. Computes clustering metrics against reference labels
6. Saves clustering summary and cluster-label composition
"""

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
)


def load_embeddings(npz_path: str) -> np.ndarray:
    """
    Load extracted embeddings from .npz.

    Expected format:
        embeddings: [N, D]
    """
    data = np.load(npz_path)
    if "embeddings" not in data:
        raise ValueError(f"'embeddings' not found in {npz_path}")
    emb = data["embeddings"].astype(np.float32)
    if emb.ndim != 2:
        raise ValueError(f"Expected embeddings with shape [N, D], got {emb.shape}")
    return emb


def load_sequences(npz_path: str) -> np.ndarray:
    """
    Load processed sequences from .npz.

    Expected format:
        X: [N, L, D]
    """
    data = np.load(npz_path)
    if "X" not in data:
        raise ValueError(f"'X' not found in {npz_path}")
    X = data["X"].astype(np.float32)
    if X.ndim != 3:
        raise ValueError(f"Expected sequences with shape [N, L, D], got {X.shape}")
    return X


def build_labels_from_last_timestep_event_type(
    X: np.ndarray,
    event_type_dim: int = 0
) -> np.ndarray:
    """
    Build a simple sequence label from the last timestep's event_type_id.

    Args:
        X: [N, L, D]
        event_type_dim: feature dimension corresponding to event_type_id

    Returns:
        labels: [N]
    """
    labels = X[:, -1, event_type_dim].astype(np.int64)
    return labels


def compute_cluster_purity(cluster_ids: np.ndarray, labels: np.ndarray) -> float:
    """
    Compute clustering purity.

    For each cluster, find the dominant label and count how many samples
    in that cluster belong to that label. Sum over clusters and divide by N.
    """
    total_correct = 0
    n = len(labels)

    for cluster_id in np.unique(cluster_ids):
        cluster_mask = cluster_ids == cluster_id
        cluster_labels = labels[cluster_mask]

        if len(cluster_labels) == 0:
            continue

        values, counts = np.unique(cluster_labels, return_counts=True)
        total_correct += counts.max()

    return total_correct / n if n > 0 else 0.0


def compute_cluster_label_composition(cluster_ids: np.ndarray, labels: np.ndarray) -> dict:
    """
    Build a readable summary of label composition inside each cluster.
    """
    summary = {}

    for cluster_id in np.unique(cluster_ids):
        cluster_mask = cluster_ids == cluster_id
        cluster_labels = labels[cluster_mask]

        values, counts = np.unique(cluster_labels, return_counts=True)

        summary[int(cluster_id)] = {
            "size": int(len(cluster_labels)),
            "label_counts": {
                int(v): int(c) for v, c in zip(values, counts)
            }
        }

    return summary


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate embeddings with clustering test")

    parser.add_argument(
        "--embeddings_file",
        type=str,
        default="data/embeddings/version1/contrastive/test_embeddings.npz",
        help="Path to extracted embeddings"
    )
    parser.add_argument(
        "--processed_test_file",
        type=str,
        default="data/processed/version1/test.npz",
        help="Path to processed test sequences"
    )
    parser.add_argument(
        "--event_type_dim",
        type=int,
        default=0,
        help="Feature dimension corresponding to event_type_id"
    )
    parser.add_argument(
        "--num_clusters",
        type=int,
        default=None,
        help="Number of KMeans clusters. If None, use number of unique labels."
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=42
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/version1/clustering"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading embeddings...")
    embeddings = load_embeddings(args.embeddings_file)
    print(f"Embeddings shape: {embeddings.shape}")

    print("\nLoading processed sequences...")
    X = load_sequences(args.processed_test_file)
    print(f"Processed sequence shape: {X.shape}")

    if len(embeddings) != len(X):
        raise ValueError(
            f"Mismatch between embeddings ({len(embeddings)}) and sequences ({len(X)})"
        )

    print("\nBuilding reference labels...")
    labels = build_labels_from_last_timestep_event_type(
        X,
        event_type_dim=args.event_type_dim
    )

    unique_labels = np.unique(labels)
    print(f"Labels shape: {labels.shape}")
    print(f"Unique labels: {unique_labels}")

    num_clusters = args.num_clusters if args.num_clusters is not None else len(unique_labels)
    print(f"\nRunning KMeans with {num_clusters} clusters...")

    kmeans = KMeans(
        n_clusters=num_clusters,
        random_state=args.random_state,
        n_init=10
    )
    cluster_ids = kmeans.fit_predict(embeddings)

    print("\nComputing clustering metrics...")
    ari = adjusted_rand_score(labels, cluster_ids)
    nmi = normalized_mutual_info_score(labels, cluster_ids)
    purity = compute_cluster_purity(cluster_ids, labels)

    if len(np.unique(cluster_ids)) > 1 and len(embeddings) > len(np.unique(cluster_ids)):
        silhouette = silhouette_score(embeddings, cluster_ids)
    else:
        silhouette = float("nan")

    metrics = {
        "num_samples": int(len(embeddings)),
        "embedding_dim": int(embeddings.shape[1]),
        "num_reference_labels": int(len(unique_labels)),
        "num_clusters": int(num_clusters),
        "ari": float(ari),
        "nmi": float(nmi),
        "purity": float(purity),
        "silhouette": float(silhouette),
    }

    print("\nClustering metrics:")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.6f}")
        else:
            print(f"  {k}: {v}")

    composition = compute_cluster_label_composition(cluster_ids, labels)

    metrics_path = output_dir / "clustering_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nSaved metrics to {metrics_path}")

    composition_path = output_dir / "cluster_composition.json"
    with open(composition_path, "w", encoding="utf-8") as f:
        json.dump(composition, f, indent=2)
    print(f"Saved cluster composition to {composition_path}")

    assignments_path = output_dir / "cluster_assignments.npz"
    np.savez_compressed(
        assignments_path,
        cluster_ids=cluster_ids.astype(np.int64),
        labels=labels.astype(np.int64),
    )
    print(f"Saved assignments to {assignments_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()