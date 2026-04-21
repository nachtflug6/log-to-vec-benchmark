from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

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


def save_json(path: str | Path, payload: Dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def check_alignment(embeddings: np.ndarray, split: Dict[str, np.ndarray], split_name: str) -> None:
    if len(embeddings) != len(split["X"]):
        raise ValueError(
            f"Embedding/split length mismatch for {split_name}: embeddings={len(embeddings)}, split={len(split['X'])}"
        )


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
    low = dim_var < variance_threshold
    return {
        "num_low_variance_dims": int(low.sum()),
        "frac_low_variance_dims": float(low.mean()),
        "overall_std": float(embeddings.std()),
        "is_likely_collapsed": bool(embeddings.std() < variance_threshold),
    }


def compute_pairwise_split_gap(a: np.ndarray, b: np.ndarray) -> dict:
    norm_a = np.linalg.norm(a, axis=1)
    norm_b = np.linalg.norm(b, axis=1)
    return {
        "mean_vector_l2_gap": float(np.linalg.norm(a.mean(axis=0) - b.mean(axis=0))),
        "std_vector_l2_gap": float(np.linalg.norm(a.std(axis=0) - b.std(axis=0))),
        "mean_norm_gap": float(abs(norm_a.mean() - norm_b.mean())),
        "std_norm_gap": float(abs(norm_a.std() - norm_b.std())),
    }


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
        return Pipeline([("scaler", StandardScaler()), ("reg", Ridge(alpha=1.0))])
    if kind == "rbf":
        return Pipeline([("scaler", StandardScaler()), ("reg", SVR(C=1.0, kernel="rbf", gamma="scale"))])
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
    model = fit_regressor(model_kind)
    model.fit(train_emb, train_y)
    pred = model.predict(test_emb)
    return {
        "mae": float(mean_absolute_error(test_y, pred)),
        "rmse": float(np.sqrt(mean_squared_error(test_y, pred))),
        "r2": float(r2_score(test_y, pred)),
    }


def run_probe_suite(
    train_emb: np.ndarray,
    train_split: Dict[str, np.ndarray],
    test_emb: np.ndarray,
    test_split: Dict[str, np.ndarray],
    model_kinds: Iterable[str] = ("linear", "rbf"),
) -> Dict[str, Any]:
    results: Dict[str, Any] = {"primary_model": "linear", "secondary_model": "rbf"}
    for target in CLASSIFICATION_TARGETS:
        if target not in train_split or target not in test_split:
            continue
        train_y = train_split[target]
        test_y = test_split[target]
        results[target] = {}
        if len(np.unique(train_y)) < 2 or len(np.unique(test_y)) < 2:
            results[target] = {
                "skipped": True,
                "task_type": "classification",
                "skipped_reason": "train or test split has fewer than two classes",
            }
            continue
        results[target]["skipped"] = False
        for kind in model_kinds:
            results[target][kind] = run_probe_task(train_emb, train_y, test_emb, test_y, "classification", kind)

    for target in REGRESSION_TARGETS:
        if target not in train_split or target not in test_split:
            continue
        train_y = train_split[target]
        test_y = test_split[target]
        results[target] = {"skipped": False, "task_type": "regression"}
        for kind in model_kinds:
            results[target][kind] = run_probe_task(train_emb, train_y, test_emb, test_y, "regression", kind)
    return results


def compute_cluster_purity(cluster_ids: np.ndarray, labels: np.ndarray) -> float:
    correct = 0
    for cluster_id in np.unique(cluster_ids):
        values, counts = np.unique(labels[cluster_ids == cluster_id], return_counts=True)
        if len(values):
            correct += int(counts.max())
    return float(correct / len(labels)) if len(labels) else 0.0


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
    n_clusters = len(np.unique(labels))
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    clusters = kmeans.fit_predict(embeddings)
    silhouette = float("nan")
    if len(np.unique(clusters)) > 1 and len(embeddings) > len(np.unique(clusters)):
        silhouette = float(silhouette_score(embeddings, clusters))
    return {
        "num_clusters": int(n_clusters),
        "ari": float(adjusted_rand_score(labels, clusters)),
        "nmi": float(normalized_mutual_info_score(labels, clusters)),
        "purity": compute_cluster_purity(clusters, labels),
        "silhouette": silhouette,
        "composition": compute_cluster_label_composition(clusters, labels),
    }


def summarize_metric_over_runs(metrics: List[Dict[str, Any]], keys: List[str]) -> Dict[str, Any]:
    out = {}
    for key in keys:
        vals = [m[key] for m in metrics if key in m and not np.isnan(m[key])]
        out[key] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))} if vals else {"mean": None, "std": None}
    return out


def run_clustering_suite(test_emb: np.ndarray, test_split: Dict[str, np.ndarray]) -> Dict[str, Any]:
    results: Dict[str, Any] = {}
    for target in CLASSIFICATION_TARGETS:
        if target not in test_split:
            continue
        labels = test_split[target]
        if len(np.unique(labels)) < 2:
            continue
        runs = [run_single_clustering(test_emb, labels, seed) for seed in (0, 1, 2, 3, 4)]
        results[target] = {
            "stability": summarize_metric_over_runs(runs, ["ari", "nmi", "purity", "silhouette"]),
            "reference_run": runs[0],
        }
    return results


def compute_retrieval_neighbors(embeddings: np.ndarray) -> np.ndarray:
    sim = cosine_similarity(embeddings)
    np.fill_diagonal(sim, -np.inf)
    return np.argsort(-sim, axis=1)


def retrieval_metrics_for_labels(labels: np.ndarray, ranked_indices: np.ndarray, ks: Iterable[int] = (1, 5, 10)) -> Dict[str, float]:
    out: Dict[str, float] = {}
    precision_scores = {k: [] for k in ks}
    recall_scores = {k: [] for k in ks}
    ap_scores = {k: [] for k in ks}
    ndcg_scores = {k: [] for k in ks}
    for i in range(len(labels)):
        neighbors = ranked_indices[i]
        relevance = (labels[neighbors] == labels[i]).astype(np.float32)
        total_relevant = int((labels == labels[i]).sum() - 1)
        for k in ks:
            rel = relevance[:k]
            precision_scores[k].append(float(rel.mean()))
            recall_scores[k].append(float(rel.sum() / total_relevant) if total_relevant > 0 else 0.0)
            if rel.sum() == 0 or total_relevant <= 0:
                ap_scores[k].append(0.0)
                ndcg_scores[k].append(0.0)
                continue
            hits = 0
            precisions = []
            for rank, is_rel in enumerate(rel, start=1):
                if is_rel:
                    hits += 1
                    precisions.append(hits / rank)
            ap_scores[k].append(float(np.sum(precisions) / min(total_relevant, k)))
            discounts = 1.0 / np.log2(np.arange(2, len(rel) + 2))
            dcg = float((rel * discounts).sum())
            ideal = np.sort(rel)[::-1]
            idcg = float((ideal * discounts).sum())
            ndcg_scores[k].append(0.0 if idcg == 0 else dcg / idcg)
    for k in ks:
        out[f"precision_at_{k}"] = float(np.mean(precision_scores[k]))
        out[f"recall_at_{k}"] = float(np.mean(recall_scores[k]))
        out[f"map_at_{k}"] = float(np.mean(ap_scores[k]))
        out[f"ndcg_at_{k}"] = float(np.mean(ndcg_scores[k]))
    return out


def run_retrieval_suite(test_emb: np.ndarray, test_split: Dict[str, np.ndarray]) -> Dict[str, Any]:
    ranked = compute_retrieval_neighbors(test_emb)
    results: Dict[str, Any] = {"num_samples": int(len(test_emb))}
    for target in CLASSIFICATION_TARGETS:
        if target in test_split and len(np.unique(test_split[target])) >= 2:
            results[target] = retrieval_metrics_for_labels(test_split[target], ranked)
    return results


def run_transition_suite(test_emb: np.ndarray, test_split: Dict[str, np.ndarray]) -> Dict[str, Any]:
    if "is_transition_window" not in test_split:
        return {}
    flags = test_split["is_transition_window"].astype(bool)
    ranked = compute_retrieval_neighbors(test_emb)
    results: Dict[str, Any] = {
        "num_transition": int(flags.sum()),
        "num_clean": int((~flags).sum()),
    }
    for name, mask in [("clean", ~flags), ("transition", flags)]:
        if int(mask.sum()) < 2:
            continue
        emb = test_emb[mask]
        for target in ["mode_id", "spectral_id", "coupling_id"]:
            if target not in test_split:
                continue
            y = test_split[target][mask]
            if len(np.unique(y)) < 2:
                results[f"{target.replace('_id', '')}_probe_linear_{name}"] = None
                continue
            probe = run_probe_task(emb, y, emb, y, "classification", "linear")
            results[f"{target.replace('_id', '')}_probe_linear_{name}"] = probe
            local_ranked = ranked[mask][:, :10]
            transition_neighbor_fraction = flags[local_ranked].mean(axis=1)
            rel = (test_split[target][local_ranked[:, 0]] == test_split[target][mask]).astype(np.float32)
            results[f"retrieval_{target}_{name}"] = {
                "precision_at_1": float(rel.mean()),
                "mean_transition_neighbor_fraction_at_10": float(transition_neighbor_fraction.mean()),
            }
    return results


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
    return {
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
        "transition": run_transition_suite(test_embeddings, test_split),
    }
