"""Evaluate trace-level embeddings for HDFS anomaly detection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


@dataclass
class TraceEmbeddingEvaluationResult:
    metrics: dict[str, Any]
    artifacts: dict[str, np.ndarray]


class HDFSTraceEmbeddingEvaluator:
    """Evaluate frozen block-level embeddings with fixed splits and labels."""

    def __init__(
        self,
        *,
        retrieval_ks: list[int] | None = None,
        structure_k: int = 5,
        logistic_max_iter: int = 2000,
        logistic_class_weight: str | None = "balanced",
        standardize_probe_features: bool = True,
        ngram_min: int = 1,
        ngram_max: int = 3,
    ) -> None:
        self.retrieval_ks = retrieval_ks or [5, 10]
        self.structure_k = structure_k
        self.logistic_max_iter = logistic_max_iter
        self.logistic_class_weight = logistic_class_weight
        self.standardize_probe_features = standardize_probe_features
        self.ngram_min = ngram_min
        self.ngram_max = ngram_max

    def evaluate(
        self,
        *,
        embeddings: np.ndarray,
        sample_id: np.ndarray,
        labels: np.ndarray,
        split: np.ndarray,
        traces_df: pd.DataFrame,
        occurrence_df: pd.DataFrame,
    ) -> TraceEmbeddingEvaluationResult:
        labels = labels.astype(str)
        split = split.astype(str)
        sample_id = sample_id.astype(str)
        y = self._binary_labels(labels)

        train_mask = split == "train"
        val_mask = split == "val"
        test_mask = split == "test"
        if not np.any(train_mask) or not np.any(test_mask):
            raise ValueError("Evaluation requires non-empty train and test splits.")

        metrics: dict[str, Any] = {
            "n_samples": int(len(embeddings)),
            "embedding_dim": int(embeddings.shape[1]),
            "split_counts": pd.Series(split).value_counts().sort_index().to_dict(),
            "label_counts": pd.Series(labels).value_counts().sort_index().to_dict(),
            "positive_label": "Anomaly",
        }
        metrics.update(
            self._linear_probe_metrics(
                embeddings=embeddings,
                y=y,
                split=split,
                labels=labels,
            )
        )
        retrieval_metrics, neighbor_artifacts = self._retrieval_metrics(
            embeddings=embeddings,
            labels=labels,
            split=split,
        )
        metrics.update(retrieval_metrics)
        metrics.update(
            self._structure_preservation_metrics(
                query_sample_ids=sample_id[test_mask],
                neighbor_sample_ids=sample_id[test_mask][neighbor_artifacts[f"neighbor_indices_at_{self.structure_k}"]],
                traces_df=traces_df,
                occurrence_df=occurrence_df,
            )
        )
        return TraceEmbeddingEvaluationResult(metrics=metrics, artifacts=neighbor_artifacts)

    @staticmethod
    def _binary_labels(labels: np.ndarray) -> np.ndarray:
        normalized = np.char.lower(labels.astype(str))
        if not np.any(normalized == "anomaly"):
            raise ValueError("Expected at least one Anomaly label for HDFS evaluation.")
        return (normalized == "anomaly").astype(np.int64)

    def _linear_probe_metrics(
        self,
        *,
        embeddings: np.ndarray,
        y: np.ndarray,
        split: np.ndarray,
        labels: np.ndarray,
    ) -> dict[str, Any]:
        train_mask = split == "train"
        val_mask = split == "val"
        test_mask = split == "test"

        X_train = embeddings[train_mask]
        X_val = embeddings[val_mask]
        X_test = embeddings[test_mask]
        if self.standardize_probe_features:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_val = scaler.transform(X_val) if np.any(val_mask) else X_val
            X_test = scaler.transform(X_test)

        classifier = LogisticRegression(
            max_iter=self.logistic_max_iter,
            class_weight=self.logistic_class_weight,
            random_state=0,
        )
        classifier.fit(X_train, y[train_mask])

        metrics: dict[str, Any] = {}
        if np.any(val_mask):
            metrics.update(
                self._probe_split_metrics(
                    prefix="val",
                    y_true=y[val_mask],
                    y_score=classifier.predict_proba(X_val)[:, 1],
                    y_pred=classifier.predict(X_val),
                )
            )
        metrics.update(
            self._probe_split_metrics(
                prefix="test",
                y_true=y[test_mask],
                y_score=classifier.predict_proba(X_test)[:, 1],
                y_pred=classifier.predict(X_test),
            )
        )
        metrics["linear_probe_class_weight"] = self.logistic_class_weight
        metrics["linear_probe_standardized"] = self.standardize_probe_features
        metrics["linear_probe_classes"] = ["Normal", "Anomaly"]
        metrics["test_label_counts"] = pd.Series(labels[test_mask]).value_counts().sort_index().to_dict()
        return metrics

    @staticmethod
    def _probe_split_metrics(
        *,
        prefix: str,
        y_true: np.ndarray,
        y_score: np.ndarray,
        y_pred: np.ndarray,
    ) -> dict[str, Any]:
        if len(np.unique(y_true)) < 2:
            auroc: float | None = None
            auprc: float | None = None
        else:
            auroc = float(roc_auc_score(y_true, y_score))
            auprc = float(average_precision_score(y_true, y_score))
        return {
            f"linear_probe_{prefix}_macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
            f"linear_probe_{prefix}_auroc": auroc,
            f"linear_probe_{prefix}_auprc": auprc,
            f"linear_probe_{prefix}_anomaly_recall": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
            f"linear_probe_{prefix}_anomaly_precision": float(
                precision_score(y_true, y_pred, pos_label=1, zero_division=0)
            ),
            f"linear_probe_{prefix}_confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
        }

    def _retrieval_metrics(
        self,
        *,
        embeddings: np.ndarray,
        labels: np.ndarray,
        split: np.ndarray,
    ) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
        test_mask = split == "test"
        test_embeddings = embeddings[test_mask]
        test_labels = labels[test_mask]
        max_k = max(max(self.retrieval_ks), self.structure_k)
        neighbor_indices, _ = self._topk_neighbor_indices(test_embeddings, max_k)

        metrics: dict[str, Any] = {}
        for k in self.retrieval_ks:
            k_neighbors = neighbor_indices[:, :k]
            hits = test_labels[k_neighbors] == test_labels[:, None]
            metrics[f"retrieval_overall_p_at_{k}_test"] = float(np.mean(hits))
            for label_name, metric_prefix in (("Anomaly", "anomaly"), ("Normal", "normal")):
                query_mask = test_labels == label_name
                metrics[f"retrieval_{metric_prefix}_p_at_{k}_test"] = (
                    float(np.mean(hits[query_mask])) if np.any(query_mask) else None
                )

        artifacts = {
            f"neighbor_indices_at_{self.structure_k}": neighbor_indices[:, : self.structure_k].astype(np.int64),
        }
        return metrics, artifacts

    @staticmethod
    def _topk_neighbor_indices(embeddings: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        if len(embeddings) <= 1:
            return np.zeros((len(embeddings), 0), dtype=np.int64), np.zeros((len(embeddings), 0), dtype=np.float32)
        k = max(1, min(k, len(embeddings) - 1))
        model = NearestNeighbors(n_neighbors=k + 1, metric="euclidean")
        model.fit(embeddings)
        distances, indices = model.kneighbors(embeddings, return_distance=True)
        return indices[:, 1 : k + 1], distances[:, 1 : k + 1]

    def _structure_preservation_metrics(
        self,
        *,
        query_sample_ids: np.ndarray,
        neighbor_sample_ids: np.ndarray,
        traces_df: pd.DataFrame,
        occurrence_df: pd.DataFrame,
    ) -> dict[str, Any]:
        traces_by_id = traces_df.set_index("sample_id")
        occurrence_by_id = occurrence_df.set_index("sample_id")

        missing_traces = set(query_sample_ids).difference(traces_by_id.index)
        missing_occurrence = set(query_sample_ids).difference(occurrence_by_id.index)
        if missing_traces:
            raise ValueError(f"Missing traces for {len(missing_traces)} evaluated sample ids.")
        if missing_occurrence:
            raise ValueError(f"Missing occurrence rows for {len(missing_occurrence)} evaluated sample ids.")

        event_count_features = self._occurrence_matrix(occurrence_by_id)
        event_count_cosine = self._neighbor_cosine_from_matrix(
            matrix=event_count_features,
            row_index=occurrence_by_id.index.astype(str).to_numpy(),
            query_sample_ids=query_sample_ids,
            neighbor_sample_ids=neighbor_sample_ids,
        )

        tfidf = TfidfVectorizer(
            analyzer="word",
            token_pattern=r"[^ ]+",
            lowercase=False,
            ngram_range=(self.ngram_min, self.ngram_max),
        )
        tfidf_matrix = tfidf.fit_transform(traces_by_id["sequence"].fillna("").astype(str))
        ngram_cosine = self._neighbor_cosine_from_matrix(
            matrix=tfidf_matrix,
            row_index=traces_by_id.index.astype(str).to_numpy(),
            query_sample_ids=query_sample_ids,
            neighbor_sample_ids=neighbor_sample_ids,
        )

        return {
            f"structure_event_count_cosine_at_{self.structure_k}_test": float(np.mean(event_count_cosine)),
            f"structure_ngram_cosine_at_{self.structure_k}_test": float(np.mean(ngram_cosine)),
            "structure_ngram_range": [self.ngram_min, self.ngram_max],
        }

    @staticmethod
    def _occurrence_matrix(occurrence_by_id: pd.DataFrame) -> sparse.csr_matrix:
        feature_columns = [
            column for column in occurrence_by_id.columns if column not in {"label", "split"}
        ]
        return sparse.csr_matrix(occurrence_by_id[feature_columns].to_numpy(dtype=np.float32))

    @staticmethod
    def _neighbor_cosine_from_matrix(
        *,
        matrix: sparse.spmatrix,
        row_index: np.ndarray,
        query_sample_ids: np.ndarray,
        neighbor_sample_ids: np.ndarray,
    ) -> np.ndarray:
        id_to_row = {sample_id: row_idx for row_idx, sample_id in enumerate(row_index.astype(str))}
        query_rows = np.array([id_to_row[sample_id] for sample_id in query_sample_ids.astype(str)], dtype=np.int64)
        neighbor_rows = np.vectorize(id_to_row.__getitem__)(neighbor_sample_ids.astype(str))

        normalized = matrix.tocsr().astype(np.float32)
        norms = np.sqrt(normalized.multiply(normalized).sum(axis=1)).A1
        inv_norms = np.divide(1.0, norms, out=np.zeros_like(norms, dtype=np.float32), where=norms > 0)
        normalized = sparse.diags(inv_norms).dot(normalized).tocsr()

        scores = np.zeros(neighbor_rows.shape, dtype=np.float32)
        for row_idx, query_row in enumerate(query_rows):
            query_vector = normalized[query_row]
            neighbor_matrix = normalized[neighbor_rows[row_idx]]
            scores[row_idx] = neighbor_matrix.dot(query_vector.T).toarray().ravel()
        return scores
