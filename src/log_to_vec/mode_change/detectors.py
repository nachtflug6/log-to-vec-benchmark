"""Baseline unsupervised mode-change detection and segment clustering."""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
from sklearn.cluster import KMeans


def _window_mean(embeddings: np.ndarray, start: int, end: int) -> np.ndarray:
    """Return mean vector for a half-open interval [start, end)."""
    return embeddings[start:end].mean(axis=0)


def compute_change_scores(
    embeddings: np.ndarray,
    window_size: int = 5,
) -> np.ndarray:
    """Compute change scores using adjacent-window mean distance.

    Score at index i is the L2 distance between means of
    [i-window_size, i) and [i, i+window_size).
    """
    if embeddings.ndim != 2:
        raise ValueError("embeddings must be a 2D array")
    if window_size < 1:
        raise ValueError("window_size must be >= 1")

    n_samples = embeddings.shape[0]
    scores = np.zeros(n_samples, dtype=np.float32)

    for i in range(window_size, n_samples - window_size):
        left = _window_mean(embeddings, i - window_size, i)
        right = _window_mean(embeddings, i, i + window_size)
        scores[i] = float(np.linalg.norm(left - right))

    return scores


def detect_change_points(
    scores: np.ndarray,
    threshold_scale: float = 2.5,
    min_distance: int = 3,
) -> List[int]:
    """Detect change points by robust thresholding and local peak filtering."""
    if scores.ndim != 1:
        raise ValueError("scores must be 1D")
    if min_distance < 1:
        raise ValueError("min_distance must be >= 1")

    median = float(np.median(scores))
    mad = float(np.median(np.abs(scores - median))) + 1e-8
    threshold = median + threshold_scale * mad

    candidate_indices = np.where(scores > threshold)[0].tolist()
    if not candidate_indices:
        return []

    peaks: List[int] = []
    for idx in candidate_indices:
        if not peaks:
            peaks.append(idx)
            continue

        if idx - peaks[-1] < min_distance:
            if scores[idx] > scores[peaks[-1]]:
                peaks[-1] = idx
        else:
            peaks.append(idx)

    return peaks


def _segment_bounds(n_samples: int, change_points: List[int]) -> List[Tuple[int, int]]:
    """Convert change points to segment bounds as half-open intervals."""
    if n_samples <= 0:
        return []

    cps = sorted(cp for cp in change_points if 0 < cp < n_samples)
    bounds: List[Tuple[int, int]] = []
    start = 0
    for cp in cps:
        bounds.append((start, cp))
        start = cp
    bounds.append((start, n_samples))
    return bounds


def cluster_segments(
    embeddings: np.ndarray,
    change_points: List[int],
    num_clusters: int = 3,
    random_state: int = 42,
) -> Dict[str, np.ndarray]:
    """Cluster mode segments and return sample-level and segment-level labels."""
    if embeddings.ndim != 2:
        raise ValueError("embeddings must be a 2D array")

    bounds = _segment_bounds(len(embeddings), change_points)
    if not bounds:
        return {
            "segment_bounds": np.empty((0, 2), dtype=np.int32),
            "segment_labels": np.empty((0,), dtype=np.int32),
            "sample_labels": np.empty((0,), dtype=np.int32),
        }

    segment_vectors = np.stack(
        [embeddings[start:end].mean(axis=0) for start, end in bounds],
        axis=0,
    )

    n_clusters = min(num_clusters, len(segment_vectors))
    if n_clusters < 1:
        n_clusters = 1

    if len(segment_vectors) == 1:
        segment_labels = np.array([0], dtype=np.int32)
    else:
        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        segment_labels = kmeans.fit_predict(segment_vectors).astype(np.int32)

    sample_labels = np.empty((len(embeddings),), dtype=np.int32)
    for (start, end), seg_label in zip(bounds, segment_labels):
        sample_labels[start:end] = seg_label

    return {
        "segment_bounds": np.array(bounds, dtype=np.int32),
        "segment_labels": segment_labels,
        "sample_labels": sample_labels,
    }
