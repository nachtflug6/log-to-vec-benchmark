"""Evaluate subprocess embeddings for structure recovery."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    adjusted_rand_score,
    f1_score,
    mean_absolute_error,
    normalized_mutual_info_score,
    pairwise_distances,
    r2_score,
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


@dataclass
class SubprocessEmbeddingEvaluationResult:
    metrics: dict[str, Any]
    artifacts: dict[str, np.ndarray]


class SubprocessEmbeddingEvaluator:
    """Evaluate whether subprocess embeddings preserve useful structure."""

    def __init__(
        self,
        *,
        retrieval_ks: list[int] | None = None,
        logistic_max_iter: int = 2000,
        logistic_class_weight: str | None = "balanced",
        standardize_probe_features: bool = True,
        distance_gap_max_samples: int = 1000,
        random_state: int = 0,
    ) -> None:
        self.retrieval_ks = retrieval_ks or [5, 10]
        self.logistic_max_iter = logistic_max_iter
        self.logistic_class_weight = logistic_class_weight
        self.standardize_probe_features = standardize_probe_features
        self.distance_gap_max_samples = distance_gap_max_samples
        self.random_state = random_state

    def evaluate(
        self,
        *,
        embeddings: np.ndarray,
        sample_id: np.ndarray,
        split: np.ndarray,
        metadata_df: pd.DataFrame,
    ) -> SubprocessEmbeddingEvaluationResult:
        sample_id = sample_id.astype(str)
        split = split.astype(str)
        metadata = self._align_metadata(sample_id, metadata_df)
        metadata["template_id"] = metadata["sub_activities"].apply(self._template_id)

        train_mask = split == "train"
        val_mask = split == "val"
        test_mask = split == "test"
        if not np.any(train_mask) or not np.any(test_mask):
            raise ValueError("Subprocess evaluation requires non-empty train and test splits.")

        metrics: dict[str, Any] = {
            "n_samples": int(len(embeddings)),
            "embedding_dim": int(embeddings.shape[1]),
            "split_counts": pd.Series(split).value_counts().sort_index().to_dict(),
            "main_activity_counts": metadata["main_activity"].value_counts().sort_index().to_dict(),
            "template_count": int(metadata["template_id"].nunique()),
            "resource_counts": metadata["resource"].value_counts().sort_index().to_dict(),
        }
        artifacts: dict[str, np.ndarray] = {}

        for target in ("main_activity", "template_id"):
            metrics.update(
                self._retrieval_metrics(
                    embeddings=embeddings,
                    labels=metadata[target].to_numpy(dtype=str),
                    split=split,
                    prefix=target,
                )
            )
            metrics.update(
                self._distance_gap_metrics(
                    embeddings=embeddings[test_mask],
                    labels=metadata.loc[test_mask, target].to_numpy(dtype=str),
                    prefix=target,
                )
            )
            metrics.update(
                self._clustering_metrics(
                    embeddings=embeddings[test_mask],
                    labels=metadata.loc[test_mask, target].to_numpy(dtype=str),
                    prefix=target,
                )
            )

        for target in ("main_activity", "resource"):
            metrics.update(
                self._classification_probe_metrics(
                    embeddings=embeddings,
                    labels=metadata[target].to_numpy(dtype=str),
                    split=split,
                    prefix=target,
                )
            )

        for target in ("duration_seconds", "sub_event_count"):
            metrics.update(
                self._regression_probe_metrics(
                    embeddings=embeddings,
                    values=metadata[target].to_numpy(dtype=np.float64),
                    split=split,
                    prefix=target,
                )
            )
        metrics.update(
            self._sequence_similarity_preservation_metrics(
                embeddings=embeddings[test_mask],
                sub_activities=metadata.loc[test_mask, "sub_activities"].tolist(),
            )
        )

        max_k = max(self.retrieval_ks)
        neighbor_indices, neighbor_distances = self._topk_neighbor_indices(embeddings[test_mask], max_k)
        artifacts[f"test_neighbor_indices_at_{max_k}"] = neighbor_indices.astype(np.int64)
        artifacts[f"test_neighbor_distances_at_{max_k}"] = neighbor_distances.astype(np.float32)
        artifacts["test_sample_id"] = sample_id[test_mask]
        artifacts["test_main_activity"] = metadata.loc[test_mask, "main_activity"].to_numpy(dtype=object)
        artifacts["test_template_id"] = metadata.loc[test_mask, "template_id"].to_numpy(dtype=object)

        return SubprocessEmbeddingEvaluationResult(metrics=metrics, artifacts=artifacts)

    @staticmethod
    def _align_metadata(sample_id: np.ndarray, metadata_df: pd.DataFrame) -> pd.DataFrame:
        required = {
            "subprocess_id",
            "main_activity",
            "resource",
            "sub_activities",
            "duration_seconds",
            "sub_event_count",
        }
        missing = required.difference(metadata_df.columns)
        if missing:
            raise ValueError(f"Metadata is missing columns: {', '.join(sorted(missing))}")
        metadata = metadata_df.copy()
        metadata["subprocess_id"] = metadata["subprocess_id"].astype(str)
        by_id = metadata.set_index("subprocess_id", drop=False)
        missing_ids = set(sample_id).difference(by_id.index)
        if missing_ids:
            raise ValueError(f"Metadata is missing {len(missing_ids)} embedding sample ids.")
        return by_id.loc[sample_id].reset_index(drop=True)

    @staticmethod
    def _template_id(sub_activities: Any) -> str:
        if isinstance(sub_activities, str):
            try:
                parsed = json.loads(sub_activities)
            except json.JSONDecodeError:
                parsed = [token for token in sub_activities.split(" ") if token]
        else:
            parsed = sub_activities
        if not isinstance(parsed, list):
            parsed = [parsed]
        normalized = [str(token) for token in parsed]
        payload = json.dumps(normalized, ensure_ascii=False, separators=(",", ":"))
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]

    def _retrieval_metrics(
        self,
        *,
        embeddings: np.ndarray,
        labels: np.ndarray,
        split: np.ndarray,
        prefix: str,
    ) -> dict[str, Any]:
        test_mask = split == "test"
        test_embeddings = embeddings[test_mask]
        test_labels = labels[test_mask]
        neighbor_indices, _ = self._topk_neighbor_indices(test_embeddings, max(self.retrieval_ks))

        metrics: dict[str, Any] = {}
        for k in self.retrieval_ks:
            k_neighbors = neighbor_indices[:, :k]
            if k_neighbors.shape[1] == 0:
                metrics[f"{prefix}_retrieval_p_at_{k}_test"] = None
                continue
            hits = test_labels[k_neighbors] == test_labels[:, None]
            metrics[f"{prefix}_retrieval_p_at_{k}_test"] = float(np.mean(hits))
        return metrics

    def _distance_gap_metrics(self, *, embeddings: np.ndarray, labels: np.ndarray, prefix: str) -> dict[str, Any]:
        if len(embeddings) < 2 or len(np.unique(labels)) < 2:
            return {
                f"{prefix}_same_distance_mean_test": None,
                f"{prefix}_different_distance_mean_test": None,
                f"{prefix}_distance_gap_test": None,
            }
        embeddings, labels = self._subsample_for_gap(embeddings, labels)
        distances = pairwise_distances(embeddings, metric="euclidean")
        upper = np.triu_indices(len(embeddings), k=1)
        pair_distances = distances[upper]
        same_mask = labels[upper[0]] == labels[upper[1]]
        same_distances = pair_distances[same_mask]
        different_distances = pair_distances[~same_mask]
        same_mean = float(np.mean(same_distances)) if len(same_distances) else None
        different_mean = float(np.mean(different_distances)) if len(different_distances) else None
        gap = None if same_mean is None or different_mean is None else float(different_mean - same_mean)
        return {
            f"{prefix}_same_distance_mean_test": same_mean,
            f"{prefix}_different_distance_mean_test": different_mean,
            f"{prefix}_distance_gap_test": gap,
        }

    def _subsample_for_gap(self, embeddings: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if len(embeddings) <= self.distance_gap_max_samples:
            return embeddings, labels
        rng = np.random.default_rng(self.random_state)
        indices = rng.choice(len(embeddings), size=self.distance_gap_max_samples, replace=False)
        return embeddings[indices], labels[indices]

    def _clustering_metrics(self, *, embeddings: np.ndarray, labels: np.ndarray, prefix: str) -> dict[str, Any]:
        unique_labels = np.unique(labels)
        if len(unique_labels) < 2 or len(unique_labels) >= len(labels):
            return {
                f"{prefix}_clustering_ari_test": None,
                f"{prefix}_clustering_nmi_test": None,
                f"{prefix}_clustering_n_clusters_test": int(len(unique_labels)),
            }
        kmeans = KMeans(n_clusters=len(unique_labels), random_state=self.random_state, n_init=10)
        clusters = kmeans.fit_predict(embeddings)
        return {
            f"{prefix}_clustering_ari_test": float(adjusted_rand_score(labels, clusters)),
            f"{prefix}_clustering_nmi_test": float(normalized_mutual_info_score(labels, clusters)),
            f"{prefix}_clustering_n_clusters_test": int(len(unique_labels)),
        }

    def _classification_probe_metrics(
        self,
        *,
        embeddings: np.ndarray,
        labels: np.ndarray,
        split: np.ndarray,
        prefix: str,
    ) -> dict[str, Any]:
        train_mask = split == "train"
        val_mask = split == "val"
        test_mask = split == "test"
        if len(np.unique(labels[train_mask])) < 2:
            return {f"{prefix}_linear_probe_test_macro_f1": None}

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
            random_state=self.random_state,
        )
        classifier.fit(X_train, labels[train_mask])

        metrics: dict[str, Any] = {
            f"{prefix}_linear_probe_classes": classifier.classes_.tolist(),
            f"{prefix}_linear_probe_class_weight": self.logistic_class_weight,
            f"{prefix}_linear_probe_standardized": self.standardize_probe_features,
        }
        if np.any(val_mask):
            metrics[f"{prefix}_linear_probe_val_macro_f1"] = float(
                f1_score(labels[val_mask], classifier.predict(X_val), average="macro", zero_division=0)
            )
        metrics[f"{prefix}_linear_probe_test_macro_f1"] = float(
            f1_score(labels[test_mask], classifier.predict(X_test), average="macro", zero_division=0)
        )
        return metrics

    def _regression_probe_metrics(
        self,
        *,
        embeddings: np.ndarray,
        values: np.ndarray,
        split: np.ndarray,
        prefix: str,
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

        regressor = Ridge(alpha=1.0)
        regressor.fit(X_train, values[train_mask])
        metrics: dict[str, Any] = {}
        if np.any(val_mask):
            metrics.update(self._regression_split_metrics(prefix=f"{prefix}_linear_probe_val", y=values[val_mask], pred=regressor.predict(X_val)))
        metrics.update(self._regression_split_metrics(prefix=f"{prefix}_linear_probe_test", y=values[test_mask], pred=regressor.predict(X_test)))
        return metrics

    @staticmethod
    def _regression_split_metrics(*, prefix: str, y: np.ndarray, pred: np.ndarray) -> dict[str, Any]:
        spearman = spearmanr(y, pred).statistic if len(np.unique(y)) > 1 and len(np.unique(pred)) > 1 else None
        return {
            f"{prefix}_r2": float(r2_score(y, pred)),
            f"{prefix}_mae": float(mean_absolute_error(y, pred)),
            f"{prefix}_spearman": None if spearman is None or np.isnan(spearman) else float(spearman),
        }

    @staticmethod
    def _sequence_similarity_preservation_metrics(
        *,
        embeddings: np.ndarray,
        sub_activities: list[Any],
    ) -> dict[str, Any]:
        if len(embeddings) < 3:
            return {
                "sequence_tfidf_embedding_spearman_test": None,
                "sequence_tfidf_embedding_mae_test": None,
            }
        docs = []
        for sequence in sub_activities:
            if isinstance(sequence, str):
                try:
                    tokens = json.loads(sequence)
                except json.JSONDecodeError:
                    tokens = [token for token in sequence.split(" ") if token]
            else:
                tokens = sequence
            if not isinstance(tokens, list):
                tokens = [tokens]
            docs.append(" ".join(SubprocessEmbeddingEvaluator._encode_token(str(token)) for token in tokens))

        tfidf = TfidfVectorizer(
            analyzer="word",
            token_pattern=r"[^ ]+",
            lowercase=False,
            ngram_range=(1, 3),
        )
        sequence_matrix = tfidf.fit_transform(docs)
        sequence_similarity = (sequence_matrix @ sequence_matrix.T).toarray()

        normalized_embeddings = embeddings.astype(np.float64)
        norms = np.linalg.norm(normalized_embeddings, axis=1, keepdims=True)
        normalized_embeddings = normalized_embeddings / np.maximum(norms, 1e-12)
        embedding_similarity = normalized_embeddings @ normalized_embeddings.T

        upper = np.triu_indices(len(embeddings), k=1)
        sequence_values = sequence_similarity[upper]
        embedding_values = embedding_similarity[upper]
        spearman = spearmanr(sequence_values, embedding_values).statistic
        return {
            "sequence_tfidf_embedding_spearman_test": None if np.isnan(spearman) else float(spearman),
            "sequence_tfidf_embedding_mae_test": float(np.mean(np.abs(sequence_values - embedding_values))),
        }

    @staticmethod
    def _encode_token(token: str) -> str:
        return token.replace("\\", "\\\\").replace(" ", "\\s")

    @staticmethod
    def _topk_neighbor_indices(embeddings: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        if len(embeddings) <= 1:
            return np.zeros((len(embeddings), 0), dtype=np.int64), np.zeros((len(embeddings), 0), dtype=np.float32)
        k = max(1, min(k, len(embeddings) - 1))
        model = NearestNeighbors(n_neighbors=k + 1, metric="euclidean")
        model.fit(embeddings)
        distances, indices = model.kneighbors(embeddings, return_distance=True)
        return indices[:, 1 : k + 1], distances[:, 1 : k + 1]
