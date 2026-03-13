"""
Evaluate embedding quality with a simple nearest-neighbor retrieval task.

This script:
1. Loads extracted test embeddings
2. Loads the corresponding processed test sequences
3. Builds simple sequence labels from raw sequence features
4. Computes nearest neighbors in embedding space
5. Reports retrieval metrics such as Recall@k
6. Saves example retrieval results for manual inspection
"""

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


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
    Load processed sequence data from .npz.

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


def build_labels_from_last_timestep_event_type(X: np.ndarray, event_type_dim: int = 0) -> np.ndarray:
    """
    Build a simple sequence label from the last timestep's event_type_id.

    Assumption:
        event_type_id is stored in feature dimension `event_type_dim`.

    Args:
        X: [N, L, D]
        event_type_dim: which feature dimension corresponds to event_type_id

    Returns:
        labels: [N]
    """
    labels = X[:, -1, event_type_dim].astype(np.int64)
    return labels


def compute_retrieval_neighbors(embeddings: np.ndarray) -> np.ndarray:
    """
    Compute cosine-similarity nearest-neighbor ranking for all samples.

    Returns:
        ranked_indices: [N, N-1], each row sorted from most similar to least similar,
                        excluding self.
    """
    sim = cosine_similarity(embeddings)  # [N, N]

    # Exclude self by setting diagonal to -inf
    np.fill_diagonal(sim, -np.inf)

    ranked_indices = np.argsort(-sim, axis=1)
    return ranked_indices


def recall_at_k(labels: np.ndarray, ranked_indices: np.ndarray, k: int) -> float:
    """
    Compute Recall@k:
    whether at least one of the top-k neighbors shares the same label.
    """
    hits = 0
    n = len(labels)

    for i in range(n):
        neighbors = ranked_indices[i, :k]
        if np.any(labels[neighbors] == labels[i]):
            hits += 1

    return hits / n if n > 0 else 0.0


def topk_match_fraction(labels: np.ndarray, ranked_indices: np.ndarray, k: int) -> float:
    """
    Compute average fraction of matching labels among top-k neighbors.
    """
    fractions = []
    n = len(labels)

    for i in range(n):
        neighbors = ranked_indices[i, :k]
        frac = np.mean(labels[neighbors] == labels[i])
        fractions.append(frac)

    return float(np.mean(fractions)) if n > 0 else 0.0


def save_example_neighbors(
    output_path: Path,
    labels: np.ndarray,
    ranked_indices: np.ndarray,
    k: int = 5,
    num_examples: int = 20
) -> None:
    """
    Save a small set of retrieval examples for manual inspection.
    """
    examples = []
    n = min(num_examples, len(labels))

    for i in range(n):
        neighbors = ranked_indices[i, :k].tolist()
        example = {
            "query_index": int(i),
            "query_label": int(labels[i]),
            "topk_neighbors": [
                {
                    "index": int(j),
                    "label": int(labels[j]),
                    "label_match": bool(labels[j] == labels[i]),
                }
                for j in neighbors
            ],
        }
        examples.append(example)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(examples, f, indent=2)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate embeddings with retrieval test")

    parser.add_argument(
        "--embeddings_file",
        type=str,
        default="data/embeddings/version1/contrastive/test_embeddings.npz",
        help="Path to extracted test embeddings"
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
        help="Feature dimension that stores event_type_id"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/version1/retrieval"
    )
    parser.add_argument(
        "--example_topk",
        type=int,
        default=5
    )

    return parser.parse_args()


def main():
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading embeddings...")
    embeddings = load_embeddings(args.embeddings_file)
    print(f"Embeddings shape: {embeddings.shape}")

    print("\nLoading processed test sequences...")
    X = load_sequences(args.processed_test_file)
    print(f"Processed test sequence shape: {X.shape}")

    if len(embeddings) != len(X):
        raise ValueError(
            f"Mismatch between embeddings ({len(embeddings)}) and sequences ({len(X)})"
        )

    print("\nBuilding sequence labels...")
    labels = build_labels_from_last_timestep_event_type(
        X,
        event_type_dim=args.event_type_dim
    )
    print(f"Labels shape: {labels.shape}")
    print(f"Unique labels: {np.unique(labels)}")

    print("\nComputing nearest neighbors...")
    ranked_indices = compute_retrieval_neighbors(embeddings)

    metrics = {
        "recall_at_1": recall_at_k(labels, ranked_indices, k=1),
        "recall_at_5": recall_at_k(labels, ranked_indices, k=5),
        "recall_at_10": recall_at_k(labels, ranked_indices, k=10),
        "top1_match_fraction": topk_match_fraction(labels, ranked_indices, k=1),
        "top5_match_fraction": topk_match_fraction(labels, ranked_indices, k=5),
        "top10_match_fraction": topk_match_fraction(labels, ranked_indices, k=10),
    }

    print("\nRetrieval metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.6f}")

    metrics_path = output_dir / "retrieval_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nSaved metrics to {metrics_path}")

    examples_path = output_dir / "retrieval_examples.json"
    save_example_neighbors(
        output_path=examples_path,
        labels=labels,
        ranked_indices=ranked_indices,
        k=args.example_topk,
        num_examples=20
    )
    print(f"Saved example neighbors to {examples_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()