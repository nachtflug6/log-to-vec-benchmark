"""Trace-level metrics for RQ2 embedding evaluation.

All functions operate on a trace: an ordered sequence of embedding vectors
[N_windows, embed_dim] produced from a single trajectory, together with
per-window mode labels and mode change-point indices.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.decomposition import PCA


# ---------------------------------------------------------------------------
# DTW (simple quadratic, no external dependency)
# ---------------------------------------------------------------------------

def _dtw_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Compute DTW distance between two sequences [La, D] and [Lb, D]."""
    La, Lb = len(a), len(b)
    cost = np.full((La, Lb), np.inf)
    cost[0, 0] = float(np.linalg.norm(a[0] - b[0]))
    for i in range(1, La):
        cost[i, 0] = cost[i - 1, 0] + float(np.linalg.norm(a[i] - b[0]))
    for j in range(1, Lb):
        cost[0, j] = cost[0, j - 1] + float(np.linalg.norm(a[0] - b[j]))
    for i in range(1, La):
        for j in range(1, Lb):
            d = float(np.linalg.norm(a[i] - b[j]))
            cost[i, j] = d + min(cost[i - 1, j], cost[i, j - 1], cost[i - 1, j - 1])
    return float(cost[-1, -1])


# ---------------------------------------------------------------------------
# Per-mode segment extraction
# ---------------------------------------------------------------------------

def _extract_mode_segments(
    embeddings: np.ndarray,
    mode_ids: np.ndarray,
    change_points: List[int],
    window_stride: int,
) -> Dict[int, List[np.ndarray]]:
    """Return {mode_id: [segment_array, ...]} from a single trajectory's embeddings.

    Each segment array is a contiguous slice of embedding vectors for one
    uninterrupted mode run.
    """
    # Convert change_points (timestep-level) to window indices
    # change_points are timestep indices; window i covers [i*stride, i*stride+L)
    # So window index = timestep // stride (approximate)
    cp_window = sorted(set(int(cp // window_stride) for cp in change_points if cp > 0))
    boundaries = [0] + cp_window + [len(embeddings)]

    segments: Dict[int, List[np.ndarray]] = {}
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        seg = embeddings[start:end]
        if len(seg) == 0:
            continue
        seg_labels = mode_ids[start:end]
        unique, counts = np.unique(seg_labels, return_counts=True)
        dominant_mode = int(unique[counts.argmax()])
        segments.setdefault(dominant_mode, []).append(seg)

    return segments


# ---------------------------------------------------------------------------
# Metric 1: Mode Separability Index (MSI)
# ---------------------------------------------------------------------------

def mode_separability_index(
    embeddings: np.ndarray,
    mode_ids: np.ndarray,
) -> float:
    """MSI = mean pairwise inter-mode centroid L2 / mean intra-mode embedding std.

    Higher is better (modes are well-separated relative to their spread).
    """
    unique_modes = np.unique(mode_ids)
    if len(unique_modes) < 2:
        return 0.0

    centroids = {m: embeddings[mode_ids == m].mean(axis=0) for m in unique_modes}
    intra_stds = {m: embeddings[mode_ids == m].std(axis=0).mean() for m in unique_modes}

    inter_dists = []
    for i, m1 in enumerate(unique_modes):
        for m2 in unique_modes[i + 1 :]:
            inter_dists.append(float(np.linalg.norm(centroids[m1] - centroids[m2])))

    mean_intra = float(np.mean(list(intra_stds.values())))
    mean_inter = float(np.mean(inter_dists))

    if mean_intra < 1e-10:
        return float("inf")
    return mean_inter / mean_intra


# ---------------------------------------------------------------------------
# Metric 2: Loop Consistency (LC) via DTW
# ---------------------------------------------------------------------------

def loop_consistency(
    per_trajectory_segments: Dict[int, List[np.ndarray]],
    max_pairs: int = 10,
    pca_dim: int = 16,
) -> Dict[int, float]:
    """Mean DTW distance between repeated traces of the same mode across runs.

    Lower is better (same mode looks similar across different production runs).

    pca_dim: reduce to this many dims before DTW to keep cost manageable.
    Returns {mode_id: mean_dtw}.
    """
    results: Dict[int, float] = {}

    for mode_id, segs in per_trajectory_segments.items():
        if len(segs) < 2:
            results[mode_id] = float("nan")
            continue

        # Optional PCA reduction for speed
        all_seg = np.concatenate(segs, axis=0)
        if all_seg.shape[1] > pca_dim:
            pca = PCA(n_components=pca_dim, random_state=0)
            pca.fit(all_seg)
            segs_reduced = [pca.transform(s) for s in segs]
        else:
            segs_reduced = segs

        # Sample pairs
        n = len(segs_reduced)
        pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
        if len(pairs) > max_pairs:
            rng = np.random.default_rng(0)
            idx = rng.choice(len(pairs), size=max_pairs, replace=False)
            pairs = [pairs[k] for k in idx]

        dtw_vals = [_dtw_distance(segs_reduced[i], segs_reduced[j]) for i, j in pairs]
        results[mode_id] = float(np.mean(dtw_vals))

    return results


# ---------------------------------------------------------------------------
# Metric 3: Transition Sharpness (TS)
# ---------------------------------------------------------------------------

def transition_sharpness(
    embeddings: np.ndarray,
    mode_ids: np.ndarray,
    change_point_windows: List[int],
    search_radius: int = 20,
) -> float:
    """Mean windows needed to move halfway between pre- and post-change centroids.

    At each known change point, measure how many windows after the boundary
    the embedding trajectory needs to cross the midpoint between the pre-change
    and post-change mode centroids.

    Lower = sharper = better.
    """
    unique_modes = np.unique(mode_ids)
    centroids = {m: embeddings[mode_ids == m].mean(axis=0) for m in unique_modes}

    sharpness_vals = []
    N = len(embeddings)

    for cp in change_point_windows:
        pre_start = max(0, cp - search_radius)
        pre_end = max(0, cp)
        post_start = min(N, cp)
        post_end = min(N, cp + search_radius)

        if pre_end <= pre_start or post_end <= post_start:
            continue

        pre_labels = mode_ids[pre_start:pre_end]
        post_labels = mode_ids[post_start:post_end]

        pre_unique, pre_counts = np.unique(pre_labels, return_counts=True)
        post_unique, post_counts = np.unique(post_labels, return_counts=True)

        mode_before = int(pre_unique[pre_counts.argmax()])
        mode_after = int(post_unique[post_counts.argmax()])

        if mode_before == mode_after:
            continue

        c_before = centroids[mode_before]
        c_after = centroids[mode_after]
        midpoint = 0.5 * (c_before + c_after)

        # Project onto the line from c_before to c_after
        direction = c_after - c_before
        norm = np.linalg.norm(direction)
        if norm < 1e-10:
            continue
        direction /= norm

        # How many windows after cp until projection crosses 0.5?
        crossed = None
        for offset in range(1, search_radius + 1):
            idx = cp + offset
            if idx >= N:
                break
            proj = float(np.dot(embeddings[idx] - c_before, direction)) / float(
                np.dot(c_after - c_before, direction) + 1e-10
            )
            if proj >= 0.5:
                crossed = offset
                break

        if crossed is not None:
            sharpness_vals.append(crossed)

    if not sharpness_vals:
        return float("nan")
    return float(np.mean(sharpness_vals))


# ---------------------------------------------------------------------------
# Metric 4: PCA Loop Compactness
# ---------------------------------------------------------------------------

def pca_loop_compactness(
    embeddings: np.ndarray,
    mode_ids: np.ndarray,
) -> Dict[int, float]:
    """Convex hull area of each mode's trace in PC1–PC2 space, normalized by
    mean inter-mode centroid distance in the same 2D space.

    Lower = tighter loop = better.
    Requires scipy for convex hull area.
    """
    try:
        from scipy.spatial import ConvexHull
    except ImportError:
        return {}

    pca = PCA(n_components=2, random_state=0)
    emb_2d = pca.fit_transform(embeddings)

    unique_modes = np.unique(mode_ids)
    centroids_2d = {m: emb_2d[mode_ids == m].mean(axis=0) for m in unique_modes}

    inter_dists = []
    modes = list(unique_modes)
    for i in range(len(modes)):
        for j in range(i + 1, len(modes)):
            inter_dists.append(
                float(np.linalg.norm(centroids_2d[modes[i]] - centroids_2d[modes[j]]))
            )
    scale = float(np.mean(inter_dists)) if inter_dists else 1.0
    if scale < 1e-10:
        scale = 1.0

    compactness: Dict[int, float] = {}
    for m in unique_modes:
        pts = emb_2d[mode_ids == m]
        if len(pts) < 3:
            compactness[m] = float("nan")
            continue
        try:
            hull = ConvexHull(pts)
            area = float(hull.volume)  # scipy ConvexHull.volume = area in 2D
        except Exception:
            area = float("nan")
        compactness[m] = area / (scale ** 2) if not np.isnan(area) else float("nan")

    return compactness


# ---------------------------------------------------------------------------
# Metric 5: Centroid Stability
# ---------------------------------------------------------------------------

def centroid_stability(
    per_trajectory_embeddings: List[np.ndarray],
    per_trajectory_mode_ids: List[np.ndarray],
) -> Dict[int, float]:
    """Std of per-trajectory mode centroids across test trajectories.

    Lower = the mode has a stable location in embedding space across runs.
    Returns {mode_id: mean_std_across_dims}.
    """
    # Collect centroids per trajectory per mode
    mode_centroids: Dict[int, List[np.ndarray]] = {}
    for emb, labels in zip(per_trajectory_embeddings, per_trajectory_mode_ids):
        unique_modes = np.unique(labels)
        for m in unique_modes:
            c = emb[labels == m].mean(axis=0)
            mode_centroids.setdefault(int(m), []).append(c)

    result: Dict[int, float] = {}
    for m, cs in mode_centroids.items():
        if len(cs) < 2:
            result[m] = float("nan")
        else:
            stacked = np.stack(cs, axis=0)  # [K, D]
            result[m] = float(stacked.std(axis=0).mean())

    return result


# ---------------------------------------------------------------------------
# Aggregate runner
# ---------------------------------------------------------------------------

def compute_all_metrics(
    embeddings: np.ndarray,
    mode_ids: np.ndarray,
    change_point_windows: List[int],
    per_trajectory_embeddings: Optional[List[np.ndarray]] = None,
    per_trajectory_mode_ids: Optional[List[np.ndarray]] = None,
    per_trajectory_segments: Optional[Dict[int, List[np.ndarray]]] = None,
    window_stride: int = 12,
) -> dict:
    """Run all trace metrics and return a flat results dict."""
    results: dict = {}

    results["mode_separability_index"] = mode_separability_index(embeddings, mode_ids)

    if per_trajectory_segments is not None:
        lc = loop_consistency(per_trajectory_segments)
        results["loop_consistency_per_mode"] = {int(k): v for k, v in lc.items()}
        finite = [v for v in lc.values() if np.isfinite(v)]
        results["loop_consistency_mean"] = float(np.mean(finite)) if finite else float("nan")

    if change_point_windows:
        results["transition_sharpness"] = transition_sharpness(
            embeddings, mode_ids, change_point_windows
        )

    compactness = pca_loop_compactness(embeddings, mode_ids)
    results["pca_loop_compactness_per_mode"] = {int(k): v for k, v in compactness.items()}
    finite_c = [v for v in compactness.values() if np.isfinite(v)]
    results["pca_loop_compactness_mean"] = float(np.mean(finite_c)) if finite_c else float("nan")

    if per_trajectory_embeddings is not None and per_trajectory_mode_ids is not None:
        stab = centroid_stability(per_trajectory_embeddings, per_trajectory_mode_ids)
        results["centroid_stability_per_mode"] = {int(k): v for k, v in stab.items()}
        finite_s = [v for v in stab.values() if np.isfinite(v)]
        results["centroid_stability_mean"] = float(np.mean(finite_s)) if finite_s else float("nan")

    return results
