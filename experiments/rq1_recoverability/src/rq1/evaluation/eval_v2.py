from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.manifold import TSNE
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

import matplotlib.pyplot as plt


CLASSIFICATION_TARGETS = ["mode_id", "spectral_id", "coupling_id", "is_transition_window"]
REGRESSION_TARGETS = ["mean_load"]


# -----------------------------------------------------------------------------
# IO
# -----------------------------------------------------------------------------

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


def save_json(path: str | Path, payload: Dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


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
    results: Dict[str, Any] = {
        "primary_model": "linear",
        "secondary_model": "rbf",
    }

    for target in CLASSIFICATION_TARGETS:
        if target not in train_split or target not in test_split:
            continue

        train_y = train_split[target]
        test_y = test_split[target]

        train_classes = np.unique(train_y)
        test_classes = np.unique(test_y)

        results[target] = {}

        if len(train_classes) < 2:
            results[target]["skipped"] = True
            results[target]["task_type"] = "classification"
            results[target]["skipped_reason"] = (
                f"train split has only one class: {train_classes.tolist()}"
            )
            continue

        if len(test_classes) < 2:
            results[target]["skipped"] = True
            results[target]["task_type"] = "classification"
            results[target]["skipped_reason"] = (
                f"test split has only one class: {test_classes.tolist()}"
            )
            continue

        results[target]["skipped"] = False
        for kind in model_kinds:
            results[target][kind] = run_probe_task(
                train_emb, train_y, test_emb, test_y, "classification", kind
            )

    for target in REGRESSION_TARGETS:
        if target not in train_split or target not in test_split:
            continue

        train_y = train_split[target]
        test_y = test_split[target]

        results[target] = {
            "skipped": False,
            "task_type": "regression",
        }

        for kind in model_kinds:
            results[target][kind] = run_probe_task(
                train_emb, train_y, test_emb, test_y, "regression", kind
            )

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


def run_single_clustering(embeddings: np.ndarray, labels: np.ndarray, random_state: int) -> Dict[str, Any]:
    unique_labels = np.unique(labels)
    num_clusters = len(unique_labels)
    kmeans = KMeans(n_clusters=num_clusters, random_state=random_state, n_init=10)
    cluster_ids = kmeans.fit_predict(embeddings)

    silhouette = float("nan")
    if len(np.unique(cluster_ids)) > 1 and len(embeddings) > len(np.unique(cluster_ids)):
        silhouette = float(silhouette_score(embeddings, cluster_ids))

    return {
        "num_clusters": int(num_clusters),
        "ari": float(adjusted_rand_score(labels, cluster_ids)),
        "nmi": float(normalized_mutual_info_score(labels, cluster_ids)),
        "purity": float(compute_cluster_purity(cluster_ids, labels)),
        "silhouette": silhouette,
        "composition": compute_cluster_label_composition(cluster_ids, labels),
    }


def summarize_metric_over_runs(metrics: List[Dict[str, Any]], keys: List[str]) -> Dict[str, Any]:
    out = {}
    for key in keys:
        vals = [m[key] for m in metrics if key in m and not np.isnan(m[key])]
        if len(vals) == 0:
            out[key] = {"mean": None, "std": None}
        else:
            out[key] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)),
            }
    return out


def run_clustering_suite(
    test_emb: np.ndarray,
    test_split: Dict[str, np.ndarray],
    seeds: Iterable[int] = (0, 1, 2, 3, 4),
) -> Dict[str, Any]:
    results: Dict[str, Any] = {}
    for target in ["mode_id", "spectral_id", "coupling_id", "is_transition_window"]:
        if target not in test_split:
            continue

        labels = test_split[target]
        unique_labels = np.unique(labels)
        if len(unique_labels) < 2:
            continue

        runs = []
        for seed in seeds:
            runs.append(run_single_clustering(test_emb, labels, random_state=int(seed)))

        results[target] = {
            "stability": summarize_metric_over_runs(runs, ["ari", "nmi", "purity", "silhouette"]),
            "reference_run": runs[0],
        }

    return results


# -----------------------------------------------------------------------------
# Retrieval
# -----------------------------------------------------------------------------

def compute_retrieval_neighbors(embeddings: np.ndarray) -> np.ndarray:
    sim = cosine_similarity(embeddings)
    np.fill_diagonal(sim, -np.inf)
    return np.argsort(-sim, axis=1)


def precision_at_k(binary_relevance: np.ndarray, k: int) -> float:
    return float(binary_relevance[:k].mean()) if len(binary_relevance) > 0 else 0.0


def recall_at_k_query(binary_relevance: np.ndarray, num_relevant_total: int, k: int) -> float:
    if num_relevant_total <= 0:
        return 0.0
    return float(binary_relevance[:k].sum() / num_relevant_total)


def average_precision_at_k(binary_relevance: np.ndarray, num_relevant_total: int, k: int) -> float:
    if num_relevant_total <= 0:
        return 0.0
    rel = binary_relevance[:k]
    if rel.sum() == 0:
        return 0.0
    precisions = []
    hits = 0
    for i, r in enumerate(rel, start=1):
        if r:
            hits += 1
            precisions.append(hits / i)
    return float(np.sum(precisions) / min(num_relevant_total, k))


def ndcg_at_k(binary_relevance: np.ndarray, k: int) -> float:
    rel = binary_relevance[:k].astype(np.float32)
    if rel.sum() == 0:
        return 0.0
    discounts = 1.0 / np.log2(np.arange(2, len(rel) + 2))
    dcg = float((rel * discounts).sum())
    ideal = np.sort(rel)[::-1]
    idcg = float((ideal * discounts).sum())
    return 0.0 if idcg == 0 else dcg / idcg


def retrieval_metrics_for_labels(
    labels: np.ndarray,
    ranked_indices: np.ndarray,
    ks: Iterable[int] = (1, 5, 10),
) -> Dict[str, float]:
    n = len(labels)
    out: Dict[str, float] = {}

    precision_scores = {k: [] for k in ks}
    recall_scores = {k: [] for k in ks}
    ap_scores = {k: [] for k in ks}
    ndcg_scores = {k: [] for k in ks}

    for i in range(n):
        neighbor_idx = ranked_indices[i]
        relevant = (labels[neighbor_idx] == labels[i]).astype(np.int32)
        num_relevant_total = int((labels == labels[i]).sum() - 1)

        for k in ks:
            precision_scores[k].append(precision_at_k(relevant, k))
            recall_scores[k].append(recall_at_k_query(relevant, num_relevant_total, k))
            ap_scores[k].append(average_precision_at_k(relevant, num_relevant_total, k))
            ndcg_scores[k].append(ndcg_at_k(relevant, k))

    for k in ks:
        out[f"precision_at_{k}"] = float(np.mean(precision_scores[k]))
        out[f"recall_at_{k}"] = float(np.mean(recall_scores[k]))
        out[f"map_at_{k}"] = float(np.mean(ap_scores[k]))
        out[f"ndcg_at_{k}"] = float(np.mean(ndcg_scores[k]))

    return out


def run_retrieval_suite(test_emb: np.ndarray, test_split: Dict[str, np.ndarray]) -> Dict[str, Any]:
    ranked_indices = compute_retrieval_neighbors(test_emb)
    results: Dict[str, Any] = {"num_samples": int(len(test_emb))}
    for target in ["mode_id", "spectral_id", "coupling_id", "is_transition_window"]:
        if target in test_split:
            results[target] = retrieval_metrics_for_labels(test_split[target], ranked_indices)
    return results


# -----------------------------------------------------------------------------
# Robustness
# -----------------------------------------------------------------------------

def apply_gaussian_noise(X: np.ndarray, noise_std: float = 0.02) -> np.ndarray:
    return (X + np.random.randn(*X.shape).astype(np.float32) * noise_std).astype(np.float32)


def apply_time_jitter(X: np.ndarray, max_shift: int = 1) -> np.ndarray:
    out = np.empty_like(X)
    for i in range(X.shape[0]):
        shift = np.random.randint(-max_shift, max_shift + 1)
        out[i] = np.roll(X[i], shift=shift, axis=0)
    return out.astype(np.float32)


def apply_amplitude_scaling(X: np.ndarray, scale_std: float = 0.02) -> np.ndarray:
    scales = (1.0 + np.random.randn(X.shape[0], 1, 1).astype(np.float32) * scale_std)
    return (X * scales).astype(np.float32)


def apply_random_point_masking(X: np.ndarray, mask_ratio: float = 0.10) -> np.ndarray:
    out = X.copy()
    mask = np.random.rand(*X.shape) < mask_ratio
    out[mask] = 0.0
    return out.astype(np.float32)


def apply_random_channel_masking(X: np.ndarray, channel_mask_ratio: float = 0.25) -> np.ndarray:
    out = X.copy()
    N, _, C = X.shape
    for i in range(N):
        num_mask = max(1, int(round(C * channel_mask_ratio)))
        idx = np.random.choice(C, size=num_mask, replace=False)
        out[i, :, idx] = 0.0
    return out.astype(np.float32)


def compute_embedding_drift(a: np.ndarray, b: np.ndarray) -> Dict[str, float]:
    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
    cosine = (a_norm * b_norm).sum(axis=1)
    l2 = np.linalg.norm(a - b, axis=1)
    return {
        "mean_cosine_similarity": float(np.mean(cosine)),
        "std_cosine_similarity": float(np.std(cosine)),
        "mean_l2_distance": float(np.mean(l2)),
        "std_l2_distance": float(np.std(l2)),
    }


def compute_neighbor_overlap(a: np.ndarray, b: np.ndarray, k: int = 10) -> float:
    neigh_a = compute_retrieval_neighbors(a)[:, :k]
    neigh_b = compute_retrieval_neighbors(b)[:, :k]
    overlaps = []
    for i in range(a.shape[0]):
        sa = set(neigh_a[i].tolist())
        sb = set(neigh_b[i].tolist())
        overlaps.append(len(sa.intersection(sb)) / k)
    return float(np.mean(overlaps))


def extract_embeddings_fn(
    encoder_fn,
    X: np.ndarray,
    batch_size: int = 256,
) -> np.ndarray:
    outs = []
    for start in range(0, len(X), batch_size):
        batch = X[start:start + batch_size]
        outs.append(encoder_fn(batch))
    return np.concatenate(outs, axis=0).astype(np.float32)


def run_robustness_suite(
    encoder_fn,
    test_embeddings: np.ndarray,
    test_split: Dict[str, np.ndarray],
    batch_size: int = 256,
) -> Dict[str, Any]:
    X = test_split["X"].astype(np.float32)

    perturbations = {
        "gaussian_noise": apply_gaussian_noise(X, noise_std=0.02),
        "time_jitter": apply_time_jitter(X, max_shift=1),
        "amplitude_scaling": apply_amplitude_scaling(X, scale_std=0.02),
        "point_masking": apply_random_point_masking(X, mask_ratio=0.10),
        "channel_masking": apply_random_channel_masking(X, channel_mask_ratio=0.25),
    }

    results = {}
    base_ranked = compute_retrieval_neighbors(test_embeddings)

    for name, X_pert in perturbations.items():
        emb_pert = extract_embeddings_fn(encoder_fn, X_pert, batch_size=batch_size)

        drift = compute_embedding_drift(test_embeddings, emb_pert)
        overlap = compute_neighbor_overlap(test_embeddings, emb_pert, k=10)

        ranked_pert = compute_retrieval_neighbors(emb_pert)
        spectral_retrieval = retrieval_metrics_for_labels(test_split["spectral_id"], ranked_pert)
        mode_retrieval = retrieval_metrics_for_labels(test_split["mode_id"], ranked_pert)

        results[name] = {
            "embedding_drift": drift,
            "neighbor_overlap_at_10": overlap,
            "retrieval_mode": mode_retrieval,
            "retrieval_spectral": spectral_retrieval,
        }

    return results


# -----------------------------------------------------------------------------
# Transition-specific reporting
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

    for name, mask in [("clean", clean_mask), ("transition", transition_mask)]:
        if mask.sum() == 0:
            continue

        results[f"mode_probe_linear_{name}"] = _safe_probe_subset(
            train_emb, train_split["mode_id"], test_emb[mask], test_split["mode_id"][mask], "classification", "linear"
        )
        results[f"spectral_probe_linear_{name}"] = _safe_probe_subset(
            train_emb, train_split["spectral_id"], test_emb[mask], test_split["spectral_id"][mask], "classification", "linear"
        )
        results[f"coupling_probe_linear_{name}"] = _safe_probe_subset(
            train_emb, train_split["coupling_id"], test_emb[mask], test_split["coupling_id"][mask], "classification", "linear"
        )

    ranked = compute_retrieval_neighbors(test_emb)
    for target in ["mode_id", "spectral_id", "coupling_id"]:
        labels = test_split[target]
        for name, mask in [("clean", clean_mask), ("transition", transition_mask)]:
            idx = _subset_indices(mask)
            if len(idx) == 0:
                continue

            precision1, recall5, map10 = [], [], []
            transition_neighbor_fraction = []

            for i in idx:
                nn = ranked[i]
                rel = (labels[nn] == labels[i]).astype(np.int32)
                num_relevant_total = int((labels == labels[i]).sum() - 1)

                precision1.append(precision_at_k(rel, 1))
                recall5.append(recall_at_k_query(rel, num_relevant_total, 5))
                map10.append(average_precision_at_k(rel, num_relevant_total, 10))

                trans_labels = test_split["is_transition_window"][nn[:10]].astype(np.int32)
                transition_neighbor_fraction.append(float(trans_labels.mean()))

            results[f"retrieval_{target}_{name}"] = {
                "precision_at_1": float(np.mean(precision1)),
                "recall_at_5": float(np.mean(recall5)),
                "map_at_10": float(np.mean(map10)),
                "mean_transition_neighbor_fraction_at_10": float(np.mean(transition_neighbor_fraction)),
            }

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
# Visualization
# -----------------------------------------------------------------------------

def _safe_label_values(arr: np.ndarray, max_points: int = 1500) -> np.ndarray:
    if len(arr) <= max_points:
        return np.arange(len(arr))
    rng = np.random.default_rng(42)
    return np.sort(rng.choice(len(arr), size=max_points, replace=False))


def make_embedding_visualizations(
    embeddings: np.ndarray,
    split: Dict[str, np.ndarray],
    output_dir: str | Path,
    prefix: str = "test",
    max_points: int = 1500,
) -> Dict[str, str]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    idx = _safe_label_values(embeddings, max_points=max_points)
    emb = embeddings[idx]

    pca_2d = PCA(n_components=2, random_state=42).fit_transform(emb)

    tsne_2d = TSNE(
        n_components=2,
        random_state=42,
        init="pca",
        learning_rate="auto",
        perplexity=min(30, max(5, len(emb) // 10)),
    ).fit_transform(emb)

    outputs = {}

    for method_name, coords in [("pca", pca_2d), ("tsne", tsne_2d)]:
        for target in ["mode_id", "spectral_id", "coupling_id", "is_transition_window", "device_id"]:
            if target not in split:
                continue
            y = split[target][idx]

            plt.figure(figsize=(7, 6))
            plt.scatter(coords[:, 0], coords[:, 1], c=y, s=10, alpha=0.8)
            plt.title(f"{prefix} embeddings | {method_name.upper()} | color={target}")
            plt.xlabel("dim 1")
            plt.ylabel("dim 2")
            plt.tight_layout()

            out_path = output_dir / f"{prefix}_{method_name}_{target}.png"
            plt.savefig(out_path, dpi=160, bbox_inches="tight")
            plt.close()
            outputs[f"{method_name}_{target}"] = str(out_path)

        if "mean_load" in split:
            y = split["mean_load"][idx]
            plt.figure(figsize=(7, 6))
            plt.scatter(coords[:, 0], coords[:, 1], c=y, s=10, alpha=0.8)
            plt.title(f"{prefix} embeddings | {method_name.upper()} | color=mean_load")
            plt.xlabel("dim 1")
            plt.ylabel("dim 2")
            plt.tight_layout()

            out_path = output_dir / f"{prefix}_{method_name}_mean_load.png"
            plt.savefig(out_path, dpi=160, bbox_inches="tight")
            plt.close()
            outputs[f"{method_name}_mean_load"] = str(out_path)

    return outputs


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
    encoder_fn=None,
    visualization_dir: str | Path | None = None,
    visualization_max_points: int = 1500,
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
        "transition": run_transition_suite(train_embeddings, train_split, test_embeddings, test_split),
    }

    if encoder_fn is not None:
        summary["robustness"] = run_robustness_suite(
            encoder_fn=encoder_fn,
            test_embeddings=test_embeddings,
            test_split=test_split,
            batch_size=256,
        )

    if visualization_dir is not None:
        summary["visualizations"] = make_embedding_visualizations(
            embeddings=test_embeddings,
            split=test_split,
            output_dir=visualization_dir,
            prefix="test",
            max_points=visualization_max_points,
        )

    return summary
