from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import numpy as np
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    adjusted_rand_score,
    balanced_accuracy_score,
    mean_absolute_error,
    mean_squared_error,
    normalized_mutual_info_score,
    r2_score,
    silhouette_score,
)
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR


CLASSIFICATION_TARGETS = ["mode_id", "spectral_id", "coupling_id", "is_transition_window"]
REGRESSION_TARGETS = ["mean_load"]


def load_embeddings(npz_path: str | Path) -> np.ndarray:
    data = np.load(npz_path)
    if "embeddings" not in data:
        raise ValueError(f"'embeddings' not found in {npz_path}")
    emb = data["embeddings"].astype(np.float32)
    if emb.ndim != 2:
        raise ValueError(f"Expected embeddings [N, D], got {emb.shape}")
    return emb


def load_split(npz_path: str | Path) -> Dict[str, np.ndarray]:
    with np.load(npz_path) as data:
        return {k: data[k] for k in data.files}


def check_alignment(embeddings: np.ndarray, split: Dict[str, np.ndarray], split_name: str) -> None:
    n = len(embeddings)
    split_n = len(split["X"])
    if n != split_n:
        raise ValueError(f"Embedding/split length mismatch for {split_name}: embeddings={n}, split={split_n}")


# -----------------------------------------------------------------------------
# Representation-level diagnostics
# -----------------------------------------------------------------------------

def compute_basic_stats(embeddings: np.ndarray) -> dict:
    norms = np.linalg.norm(embeddings, axis=1)
    dim_std = embeddings.std(axis=0)
    dim_var = embeddings.var(axis=0)
    return {
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


def compute_collapse_metrics(embeddings: np.ndarray, variance_threshold: float = 1e-4) -> dict:
    dim_var = embeddings.var(axis=0)
    num_low_variance_dims = int((dim_var < variance_threshold).sum())
    frac_low_variance_dims = float(num_low_variance_dims / len(dim_var))
    overall_std = embeddings.std()
    return {
        "num_low_variance_dims": num_low_variance_dims,
        "frac_low_variance_dims": frac_low_variance_dims,
        "overall_std": float(overall_std),
        "is_likely_collapsed": bool(overall_std < variance_threshold),
    }


def compute_pairwise_split_gap(a: np.ndarray, b: np.ndarray) -> dict:
    mean_a, mean_b = a.mean(axis=0), b.mean(axis=0)
    std_a, std_b = a.std(axis=0), b.std(axis=0)
    norm_a, norm_b = np.linalg.norm(a, axis=1), np.linalg.norm(b, axis=1)
    return {
        "mean_vector_l2_gap": float(np.linalg.norm(mean_a - mean_b)),
        "std_vector_l2_gap": float(np.linalg.norm(std_a - std_b)),
        "mean_norm_gap": float(abs(norm_a.mean() - norm_b.mean())),
        "std_norm_gap": float(abs(norm_a.std() - norm_b.std())),
    }


# -----------------------------------------------------------------------------
# Probes
# -----------------------------------------------------------------------------

def fit_classifier(kind: str):
    if kind == "linear":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=5000, solver="lbfgs", random_state=42)),
        ])
    if kind == "rbf":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(C=1.0, kernel="rbf", gamma="scale", random_state=42)),
        ])
    raise ValueError(f"Unknown classifier kind: {kind}")


def fit_regressor(kind: str):
    if kind == "linear":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("reg", Ridge(alpha=1.0)),
        ])
    if kind == "rbf":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("reg", SVR(C=1.0, kernel="rbf", gamma="scale")),
        ])
    raise ValueError(f"Unknown regressor kind: {kind}")


def run_probe_task(
    train_emb: np.ndarray,
    train_y: np.ndarray,
    test_emb: np.ndarray,
    test_y: np.ndarray,
    task_type: str,
    model_kind: str,
) -> Dict[str, float]:
    if task_type == "classification":
        model = fit_classifier(model_kind)
        model.fit(train_emb, train_y)
        pred = model.predict(test_emb)
        return {
            "accuracy": float(accuracy_score(test_y, pred)),
            "balanced_accuracy": float(balanced_accuracy_score(test_y, pred)),
        }

    if task_type == "regression":
        model = fit_regressor(model_kind)
        model.fit(train_emb, train_y)
        pred = model.predict(test_emb)
        return {
            "mae": float(mean_absolute_error(test_y, pred)),
            "rmse": float(np.sqrt(mean_squared_error(test_y, pred))),
            "r2": float(r2_score(test_y, pred)),
        }

    raise ValueError(f"Unknown task_type: {task_type}")


def run_probe_suite(
    train_emb: np.ndarray,
    train_split: Dict[str, np.ndarray],
    test_emb: np.ndarray,
    test_split: Dict[str, np.ndarray],
    model_kinds: Iterable[str] = ("linear", "rbf"),
) -> Dict[str, Any]:
    results: Dict[str, Any] = {}

    for target in CLASSIFICATION_TARGETS:
        if target not in train_split or target not in test_split:
            continue
        results[target] = {}
        train_y = train_split[target]
        test_y = test_split[target]
        for kind in model_kinds:
            results[target][kind] = run_probe_task(train_emb, train_y, test_emb, test_y, "classification", kind)

    for target in REGRESSION_TARGETS:
        if target not in train_split or target not in test_split:
            continue
        results[target] = {}
        train_y = train_split[target]
        test_y = test_split[target]
        for kind in model_kinds:
            results[target][kind] = run_probe_task(train_emb, train_y, test_emb, test_y, "regression", kind)

    return results


# -----------------------------------------------------------------------------
# Clustering
# -----------------------------------------------------------------------------

def compute_cluster_purity(cluster_ids: np.ndarray, labels: np.ndarray) -> float:
    total_correct = 0
    n = len(labels)
    for cluster_id in np.unique(cluster_ids):
        cluster_mask = cluster_ids == cluster_id
        cluster_labels = labels[cluster_mask]
        if len(cluster_labels) == 0:
            continue
        _, counts = np.unique(cluster_labels, return_counts=True)
        total_correct += counts.max()
    return total_correct / n if n > 0 else 0.0


def compute_cluster_label_composition(cluster_ids: np.ndarray, labels: np.ndarray) -> dict:
    summary = {}
    for cluster_id in np.unique(cluster_ids):
        mask = cluster_ids == cluster_id
        values, counts = np.unique(labels[mask], return_counts=True)
        summary[int(cluster_id)] = {
            "size": int(mask.sum()),
            "label_counts": {int(v): int(c) for v, c in zip(values, counts)},
        }
    return summary


def run_clustering_suite(test_emb: np.ndarray, test_split: Dict[str, np.ndarray]) -> Dict[str, Any]:
    results: Dict[str, Any] = {}
    for target in ["mode_id", "spectral_id", "coupling_id", "is_transition_window"]:
        if target not in test_split:
            continue
        labels = test_split[target]
        unique_labels = np.unique(labels)
        num_clusters = len(unique_labels)
        if num_clusters < 2:
            continue
        kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
        cluster_ids = kmeans.fit_predict(test_emb)
        silhouette = float("nan")
        if len(np.unique(cluster_ids)) > 1 and len(test_emb) > len(np.unique(cluster_ids)):
            silhouette = float(silhouette_score(test_emb, cluster_ids))
        results[target] = {
            "num_clusters": int(num_clusters),
            "ari": float(adjusted_rand_score(labels, cluster_ids)),
            "nmi": float(normalized_mutual_info_score(labels, cluster_ids)),
            "purity": float(compute_cluster_purity(cluster_ids, labels)),
            "silhouette": silhouette,
            "composition": compute_cluster_label_composition(cluster_ids, labels),
        }
    return results


# -----------------------------------------------------------------------------
# Retrieval
# -----------------------------------------------------------------------------

def compute_retrieval_neighbors(embeddings: np.ndarray) -> np.ndarray:
    sim = cosine_similarity(embeddings)
    np.fill_diagonal(sim, -np.inf)
    return np.argsort(-sim, axis=1)


def recall_at_k(labels: np.ndarray, ranked_indices: np.ndarray, k: int) -> float:
    hits = 0
    n = len(labels)
    for i in range(n):
        neighbors = ranked_indices[i, :k]
        if np.any(labels[neighbors] == labels[i]):
            hits += 1
    return hits / n if n > 0 else 0.0


def topk_match_fraction(labels: np.ndarray, ranked_indices: np.ndarray, k: int) -> float:
    fractions = []
    n = len(labels)
    for i in range(n):
        neighbors = ranked_indices[i, :k]
        fractions.append(np.mean(labels[neighbors] == labels[i]))
    return float(np.mean(fractions)) if n > 0 else 0.0


def retrieval_metrics_for_labels(labels: np.ndarray, ranked_indices: np.ndarray) -> Dict[str, float]:
    return {
        "recall_at_1": float(recall_at_k(labels, ranked_indices, 1)),
        "recall_at_5": float(recall_at_k(labels, ranked_indices, 5)),
        "recall_at_10": float(recall_at_k(labels, ranked_indices, 10)),
        "top1_match_fraction": float(topk_match_fraction(labels, ranked_indices, 1)),
        "top5_match_fraction": float(topk_match_fraction(labels, ranked_indices, 5)),
        "top10_match_fraction": float(topk_match_fraction(labels, ranked_indices, 10)),
    }


def run_retrieval_suite(test_emb: np.ndarray, test_split: Dict[str, np.ndarray]) -> Dict[str, Any]:
    ranked_indices = compute_retrieval_neighbors(test_emb)
    results: Dict[str, Any] = {"num_samples": int(len(test_emb))}
    for target in ["mode_id", "spectral_id", "coupling_id", "is_transition_window"]:
        if target in test_split:
            results[target] = retrieval_metrics_for_labels(test_split[target], ranked_indices)
    return results


# -----------------------------------------------------------------------------
# OOD tests
# -----------------------------------------------------------------------------

def _subset_indices(mask: np.ndarray) -> np.ndarray:
    return np.where(mask)[0]


def _safe_probe_subset(
    train_emb: np.ndarray,
    train_y: np.ndarray,
    test_emb: np.ndarray,
    test_y: np.ndarray,
    task_type: str,
    model_kind: str,
) -> Dict[str, float] | None:
    if len(test_emb) == 0 or len(np.unique(test_y)) < (2 if task_type == "classification" else 1):
        return None
    if task_type == "classification" and len(np.unique(train_y)) < 2:
        return None
    return run_probe_task(train_emb, train_y, test_emb, test_y, task_type, model_kind)


def run_ood_suite(
    train_emb: np.ndarray,
    train_split: Dict[str, np.ndarray],
    test_emb: np.ndarray,
    test_split: Dict[str, np.ndarray],
) -> Dict[str, Any]:
    results: Dict[str, Any] = {}

    # Seen vs unseen factor-combination generalization
    train_pairs = set(zip(train_split["spectral_id"].tolist(), train_split["coupling_id"].tolist()))
    test_pairs = list(zip(test_split["spectral_id"].tolist(), test_split["coupling_id"].tolist()))
    seen_mask = np.array([pair in train_pairs for pair in test_pairs], dtype=bool)
    unseen_mask = ~seen_mask

    pair_results = {
        "num_seen": int(seen_mask.sum()),
        "num_unseen": int(unseen_mask.sum()),
        "seen_pairs_in_train": sorted([list(p) for p in train_pairs]),
    }
    if unseen_mask.sum() > 0:
        pair_results["mode_probe_linear_unseen"] = _safe_probe_subset(
            train_emb, train_split["mode_id"], test_emb[unseen_mask], test_split["mode_id"][unseen_mask], "classification", "linear"
        )
        pair_results["coupling_probe_linear_unseen"] = _safe_probe_subset(
            train_emb, train_split["coupling_id"], test_emb[unseen_mask], test_split["coupling_id"][unseen_mask], "classification", "linear"
        )
    if seen_mask.sum() > 0:
        pair_results["mode_probe_linear_seen"] = _safe_probe_subset(
            train_emb, train_split["mode_id"], test_emb[seen_mask], test_split["mode_id"][seen_mask], "classification", "linear"
        )
    results["heldout_factor_combination"] = pair_results

    # Load-range extrapolation: outside train range
    train_min = float(train_split["mean_load"].min())
    train_max = float(train_split["mean_load"].max())
    outside_mask = (test_split["mean_load"] < train_min) | (test_split["mean_load"] > train_max)
    inside_mask = ~outside_mask
    load_results = {
        "train_load_min": train_min,
        "train_load_max": train_max,
        "num_inside": int(inside_mask.sum()),
        "num_outside": int(outside_mask.sum()),
    }
    if inside_mask.sum() > 0:
        load_results["load_probe_linear_inside"] = _safe_probe_subset(
            train_emb, train_split["mean_load"], test_emb[inside_mask], test_split["mean_load"][inside_mask], "regression", "linear"
        )
    if outside_mask.sum() > 0:
        load_results["load_probe_linear_outside"] = _safe_probe_subset(
            train_emb, train_split["mean_load"], test_emb[outside_mask], test_split["mean_load"][outside_mask], "regression", "linear"
        )
    results["heldout_load_range"] = load_results

    # Device OOD summary: only meaningful if test has devices absent from train
    train_devices = set(train_split["device_id"].tolist())
    unseen_device_mask = np.array([d not in train_devices for d in test_split["device_id"].tolist()], dtype=bool)
    results["heldout_device"] = {
        "num_unseen_device_windows": int(unseen_device_mask.sum()),
        "unseen_device_ids": sorted(set(map(int, test_split["device_id"][unseen_device_mask]))),
    }

    return results


# -----------------------------------------------------------------------------
# Transition-specific reporting
# -----------------------------------------------------------------------------

def run_transition_suite(
    train_emb: np.ndarray,
    train_split: Dict[str, np.ndarray],
    test_emb: np.ndarray,
    test_split: Dict[str, np.ndarray],
) -> Dict[str, Any]:
    transition_mask = test_split["is_transition_window"].astype(bool)
    clean_mask = ~transition_mask

    results: Dict[str, Any] = {
        "num_transition": int(transition_mask.sum()),
        "num_clean": int(clean_mask.sum()),
    }

    # Full-mode probe separately on clean vs transition windows
    for name, mask in [("clean", clean_mask), ("transition", transition_mask)]:
        if mask.sum() == 0:
            continue
        metrics = _safe_probe_subset(
            train_emb,
            train_split["mode_id"],
            test_emb[mask],
            test_split["mode_id"][mask],
            "classification",
            "linear",
        )
        results[f"mode_probe_linear_{name}"] = metrics

    # Retrieval separately on clean vs transition
    ranked = compute_retrieval_neighbors(test_emb)
    for name, mask in [("clean", clean_mask), ("transition", transition_mask)]:
        idx = _subset_indices(mask)
        if len(idx) == 0:
            continue
        labels = test_split["mode_id"]
        subset_hits_at1 = []
        subset_hits_at5 = []
        subset_top5_frac = []
        for i in idx:
            nn = ranked[i]
            subset_hits_at1.append(np.any(labels[nn[:1]] == labels[i]))
            subset_hits_at5.append(np.any(labels[nn[:5]] == labels[i]))
            subset_top5_frac.append(np.mean(labels[nn[:5]] == labels[i]))
        results[f"retrieval_mode_{name}"] = {
            "recall_at_1": float(np.mean(subset_hits_at1)),
            "recall_at_5": float(np.mean(subset_hits_at5)),
            "top5_match_fraction": float(np.mean(subset_top5_frac)),
        }

    # Distance-to-boundary buckets for transition windows
    if transition_mask.sum() > 0:
        dist = test_split["distance_to_boundary"][transition_mask]
        bins = {
            "near": dist <= 4,
            "mid": (dist > 4) & (dist <= 12),
            "far": dist > 12,
        }
        transition_emb = test_emb[transition_mask]
        transition_mode = test_split["mode_id"][transition_mask]
        bucket_results = {}
        for name, bmask in bins.items():
            if bmask.sum() == 0:
                continue
            metrics = _safe_probe_subset(
                train_emb,
                train_split["mode_id"],
                transition_emb[bmask],
                transition_mode[bmask],
                "classification",
                "linear",
            )
            bucket_results[name] = {
                "num_samples": int(bmask.sum()),
                "mode_probe_linear": metrics,
            }
        results["transition_distance_buckets"] = bucket_results

    return results


# -----------------------------------------------------------------------------
# Top-level suite
# -----------------------------------------------------------------------------

def run_full_evaluation_suite(
    train_embeddings: np.ndarray,
    val_embeddings: np.ndarray,
    test_embeddings: np.ndarray,
    train_split: Dict[str, np.ndarray],
    val_split: Dict[str, np.ndarray],
    test_split: Dict[str, np.ndarray],
) -> Dict[str, Any]:
    check_alignment(train_embeddings, train_split, "train")
    check_alignment(val_embeddings, val_split, "val")
    check_alignment(test_embeddings, test_split, "test")

    summary = {
        "embedding_stats": {
            "train_basic": compute_basic_stats(train_embeddings),
            "val_basic": compute_basic_stats(val_embeddings),
            "test_basic": compute_basic_stats(test_embeddings),
            "train_collapse": compute_collapse_metrics(train_embeddings),
            "val_collapse": compute_collapse_metrics(val_embeddings),
            "test_collapse": compute_collapse_metrics(test_embeddings),
            "train_val_gap": compute_pairwise_split_gap(train_embeddings, val_embeddings),
            "val_test_gap": compute_pairwise_split_gap(val_embeddings, test_embeddings),
            "train_test_gap": compute_pairwise_split_gap(train_embeddings, test_embeddings),
        },
        "probes": run_probe_suite(train_embeddings, train_split, test_embeddings, test_split),
        "clustering": run_clustering_suite(test_embeddings, test_split),
        "retrieval": run_retrieval_suite(test_embeddings, test_split),
        "ood": run_ood_suite(train_embeddings, train_split, test_embeddings, test_split),
        "transition": run_transition_suite(train_embeddings, train_split, test_embeddings, test_split),
    }
    return summary


def save_json(path: str | Path, payload: Dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
