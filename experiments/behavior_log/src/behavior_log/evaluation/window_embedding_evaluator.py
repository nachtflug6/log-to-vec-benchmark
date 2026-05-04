"""Evaluate behavior recoverability from window embeddings."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    adjusted_rand_score,
    confusion_matrix,
    f1_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE


@dataclass
class EmbeddingEvaluationResult:
    """Simple container for evaluation metrics."""

    metrics: dict
    artifacts: dict


class WindowEmbeddingEvaluator:
    """Compute simple downstream recoverability metrics."""

    def __init__(
        self,
        knn_k: int = 3,
        logistic_max_iter: int = 2000,
        retrieval_k: int = 5,
        retrieval_ks: list[int] | None = None,
    ) -> None:
        self.knn_k = knn_k
        self.logistic_max_iter = logistic_max_iter
        self.retrieval_k = retrieval_k
        self.retrieval_ks = retrieval_ks or [1, 5, 10, 20]

    def evaluate(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
        split: np.ndarray,
        is_transition_window: np.ndarray | None = None,
    ) -> EmbeddingEvaluationResult:
        train_mask = split == "train"
        val_mask = split == "val"
        test_mask = split == "test"
        if is_transition_window is None:
            is_transition_window = np.zeros(len(labels), dtype=np.int64)

        X_train = embeddings[train_mask]
        y_train = labels[train_mask]
        X_val = embeddings[val_mask]
        y_val = labels[val_mask]
        X_test = embeddings[test_mask]
        y_test = labels[test_mask]

        logistic = LogisticRegression(max_iter=self.logistic_max_iter)
        logistic.fit(X_train, y_train)
        val_pred = logistic.predict(X_val)
        test_pred = logistic.predict(X_test)

        knn = KNeighborsClassifier(n_neighbors=self.knn_k)
        knn.fit(X_train, y_train)
        class_labels = np.unique(labels.astype(np.int64))

        metrics = {
            "n_windows": int(len(embeddings)),
            "n_train_windows": int(np.sum(train_mask)),
            "n_val_windows": int(np.sum(val_mask)),
            "n_test_windows": int(np.sum(test_mask)),
            "class_labels": [int(x) for x in class_labels.tolist()],
            "linear_probe_val_accuracy": float(accuracy_score(y_val, val_pred)),
            "linear_probe_test_accuracy": float(accuracy_score(y_test, test_pred)),
            "linear_probe_val_macro_f1": float(f1_score(y_val, val_pred, average="macro")),
            "linear_probe_test_macro_f1": float(f1_score(y_test, test_pred, average="macro")),
            "linear_probe_val_confusion_matrix": confusion_matrix(y_val, val_pred, labels=class_labels).tolist(),
            "linear_probe_test_confusion_matrix": confusion_matrix(y_test, test_pred, labels=class_labels).tolist(),
            "knn_val_accuracy": float(accuracy_score(y_val, knn.predict(X_val))),
            "knn_test_accuracy": float(accuracy_score(y_test, knn.predict(X_test))),
        }
        metrics.update(self._retrieval_metric_block(X_val, y_val, prefix="val"))
        metrics.update(self._retrieval_metric_block(X_test, y_test, prefix="test"))
        metrics["per_class_retrieval_p_at_5_test"] = self._retrieval_precision_at_k_per_class(X_test, y_test, 5)
        metrics["per_class_retrieval_p_at_5_val"] = self._retrieval_precision_at_k_per_class(X_val, y_val, 5)
        metrics.update(
            self._window_type_metric_block(
                X_test=X_test,
                y_test=y_test,
                test_pred=test_pred,
                knn_pred=knn.predict(X_test),
                is_transition_test=is_transition_window[test_mask],
            )
        )

        artifacts = {
            "pca_2d": self._project_pca_2d(embeddings),
            "tsne_2d": self._project_tsne_2d(embeddings),
        }

        n_classes = len(np.unique(labels))
        if n_classes > 1 and len(embeddings) > n_classes:
            kmeans = KMeans(n_clusters=n_classes, n_init=10, random_state=0)
            cluster_pred = kmeans.fit_predict(embeddings)
            metrics["clustering_kmeans_ari"] = float(adjusted_rand_score(labels, cluster_pred))
            metrics["clustering_kmeans_nmi"] = float(normalized_mutual_info_score(labels, cluster_pred))
            metrics["clustering_silhouette_score"] = float(silhouette_score(embeddings, labels))
            artifacts["cluster_assignments"] = cluster_pred.astype(np.int64)
        else:
            metrics["clustering_kmeans_ari"] = None
            metrics["clustering_kmeans_nmi"] = None
            metrics["clustering_silhouette_score"] = None
            artifacts["cluster_assignments"] = None

        return EmbeddingEvaluationResult(metrics=metrics, artifacts=artifacts)

    def _retrieval_metric_block(self, embeddings: np.ndarray, labels: np.ndarray, prefix: str) -> dict:
        metrics: dict[str, float | dict] = {}
        for k in self.retrieval_ks:
            metrics[f"retrieval_p_at_{k}_{prefix}"] = self._retrieval_precision_at_k(embeddings, labels, k)
        metrics[f"retrieval_p_at_k_{prefix}"] = self._retrieval_precision_at_k(embeddings, labels, self.retrieval_k)
        return metrics

    def _window_type_metric_block(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        test_pred: np.ndarray,
        knn_pred: np.ndarray,
        is_transition_test: np.ndarray,
    ) -> dict:
        metrics: dict[str, float | None] = {}
        window_type_masks = {
            "clean": is_transition_test.astype(np.int64) == 0,
            "transition": is_transition_test.astype(np.int64) > 0,
        }
        for window_type, mask in window_type_masks.items():
            count = int(np.sum(mask))
            metrics[f"{window_type}_test_n_windows"] = count
            if count == 0:
                metrics[f"{window_type}_linear_probe_test_accuracy"] = None
                metrics[f"{window_type}_linear_probe_test_macro_f1"] = None
                metrics[f"{window_type}_knn_test_accuracy"] = None
                metrics[f"{window_type}_retrieval_p_at_5_test"] = None
                continue
            metrics[f"{window_type}_linear_probe_test_accuracy"] = float(accuracy_score(y_test[mask], test_pred[mask]))
            metrics[f"{window_type}_linear_probe_test_macro_f1"] = float(f1_score(y_test[mask], test_pred[mask], average="macro"))
            metrics[f"{window_type}_knn_test_accuracy"] = float(accuracy_score(y_test[mask], knn_pred[mask]))
            metrics[f"{window_type}_retrieval_p_at_5_test"] = self._retrieval_precision_at_k(X_test[mask], y_test[mask], 5)
        return metrics

    @staticmethod
    def _retrieval_precision_at_k(embeddings: np.ndarray, labels: np.ndarray, k: int) -> float:
        if len(embeddings) <= 1:
            return 0.0
        k = max(1, min(k, len(embeddings) - 1))
        distances = np.linalg.norm(embeddings[:, None, :] - embeddings[None, :, :], axis=2)
        np.fill_diagonal(distances, np.inf)
        neighbor_indices = np.argsort(distances, axis=1)[:, :k]
        neighbor_labels = labels[neighbor_indices]
        hits = (neighbor_labels == labels[:, None]).astype(np.float32)
        return float(np.mean(np.mean(hits, axis=1)))

    @classmethod
    def _retrieval_precision_at_k_per_class(cls, embeddings: np.ndarray, labels: np.ndarray, k: int) -> dict[str, float]:
        results: dict[str, float] = {}
        if len(embeddings) <= 1:
            return {str(int(class_id)): 0.0 for class_id in np.unique(labels.astype(np.int64))}
        k = max(1, min(k, len(embeddings) - 1))
        distances = np.linalg.norm(embeddings[:, None, :] - embeddings[None, :, :], axis=2)
        np.fill_diagonal(distances, np.inf)
        neighbor_indices = np.argsort(distances, axis=1)[:, :k]
        neighbor_labels = labels[neighbor_indices]
        hits = (neighbor_labels == labels[:, None]).astype(np.float32)
        for class_id in np.unique(labels.astype(np.int64)):
            mask = labels.astype(np.int64) == class_id
            results[str(int(class_id))] = float(np.mean(np.mean(hits[mask], axis=1))) if np.any(mask) else 0.0
        return results

    @staticmethod
    def _project_pca_2d(embeddings: np.ndarray) -> np.ndarray:
        if len(embeddings) < 2:
            return np.zeros((len(embeddings), 2), dtype=np.float32)
        return PCA(n_components=2, random_state=0).fit_transform(embeddings).astype(np.float32)

    @staticmethod
    def _project_tsne_2d(embeddings: np.ndarray) -> np.ndarray:
        if len(embeddings) < 3:
            return np.zeros((len(embeddings), 2), dtype=np.float32)
        perplexity = max(2, min(30, len(embeddings) - 1))
        return TSNE(n_components=2, random_state=0, init="pca", learning_rate="auto", perplexity=perplexity).fit_transform(embeddings).astype(np.float32)
