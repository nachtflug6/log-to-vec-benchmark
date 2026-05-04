"""Compute simple baseline representations for behavior-log windows."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from behavior_log.data.windowing import WindowBundle


@dataclass
class WindowRepresentationResult:
    """Output of one baseline representation pipeline."""

    embeddings: np.ndarray
    summary: dict


class WindowRepresentationBuilder:
    """Build simple baseline window embeddings."""

    def __init__(self, method: str, n_components: int = 8, whiten: bool = False) -> None:
        self.method = method
        self.n_components = n_components
        self.whiten = whiten
        self.scaler: StandardScaler | None = None
        self.pca: PCA | None = None
        self.tfidf_vectorizer: TfidfVectorizer | None = None
        self.count_vectorizer: CountVectorizer | None = None
        self.svd: TruncatedSVD | None = None

    def fit_transform(self, windows: WindowBundle) -> WindowRepresentationResult:
        """Fit train-time statistics if needed and embed every window."""

        train_mask = windows.split == "train"
        if self.method == "pca":
            features = self._raw_flatten(windows)
            self.scaler = StandardScaler()
            train_scaled = self.scaler.fit_transform(features[train_mask])
            n_components = min(self.n_components, train_scaled.shape[0], train_scaled.shape[1])
            self.pca = PCA(n_components=n_components, whiten=self.whiten, random_state=0)
            self.pca.fit(train_scaled)
            embeddings = self.pca.transform(self.scaler.transform(features)).astype(np.float32)
            summary = {
                "method": self.method,
                "input_dim": int(features.shape[1]),
                "embedding_dim": int(embeddings.shape[1]),
                "explained_variance_ratio": self.pca.explained_variance_ratio_.tolist(),
                "total_explained_variance_ratio": float(np.sum(self.pca.explained_variance_ratio_)),
            }
            return WindowRepresentationResult(embeddings=embeddings, summary=summary)

        if self.method == "tfidf_ngram":
            docs = self._window_docs(windows)
            self.tfidf_vectorizer = TfidfVectorizer(
                analyzer="word",
                token_pattern=r"[^ ]+",
                lowercase=False,
                ngram_range=(1, 3),
            )
            train_matrix = self.tfidf_vectorizer.fit_transform(docs[train_mask])
            all_matrix = self.tfidf_vectorizer.transform(docs)
            svd_dim = min(self.n_components, train_matrix.shape[0], train_matrix.shape[1])
            self.svd = TruncatedSVD(n_components=svd_dim, random_state=0)
            self.svd.fit(train_matrix)
            embeddings = self.svd.transform(all_matrix).astype(np.float32)
            summary = {
                "method": self.method,
                "input_dim": int(all_matrix.shape[1]),
                "embedding_dim": int(embeddings.shape[1]),
                "ngram_range": [1, 3],
                "vocab_size": int(len(self.tfidf_vectorizer.vocabulary_)),
                "explained_variance_ratio": self.svd.explained_variance_ratio_.tolist(),
                "total_explained_variance_ratio": float(np.sum(self.svd.explained_variance_ratio_)),
            }
            return WindowRepresentationResult(embeddings=embeddings, summary=summary)

        if self.method == "component_aware_ngram":
            docs = self._window_docs(windows)
            self.count_vectorizer = CountVectorizer(
                analyzer="word",
                token_pattern=r"[^ ]+",
                lowercase=False,
                ngram_range=(2, 3),
            )
            train_matrix = self.count_vectorizer.fit_transform(docs[train_mask])
            all_matrix = self.count_vectorizer.transform(docs)
            svd_dim = min(self.n_components, train_matrix.shape[0], train_matrix.shape[1])
            self.svd = TruncatedSVD(n_components=svd_dim, random_state=0)
            self.svd.fit(train_matrix)
            embeddings = self.svd.transform(all_matrix).astype(np.float32)
            summary = {
                "method": self.method,
                "input_dim": int(all_matrix.shape[1]),
                "embedding_dim": int(embeddings.shape[1]),
                "ngram_range": [2, 3],
                "vocab_size": int(len(self.count_vectorizer.vocabulary_)),
                "explained_variance_ratio": self.svd.explained_variance_ratio_.tolist(),
                "total_explained_variance_ratio": float(np.sum(self.svd.explained_variance_ratio_)),
            }
            return WindowRepresentationResult(embeddings=embeddings, summary=summary)

        features = self._compute_base_features(windows)
        embeddings = features.astype(np.float32)
        summary = {
            "method": self.method,
            "input_dim": int(features.shape[1]),
            "embedding_dim": int(features.shape[1]),
        }
        return WindowRepresentationResult(embeddings=embeddings, summary=summary)

    def save(self, path: str | Path) -> None:
        """Persist train-time state when needed."""

        payload = {
            "method": self.method,
            "n_components": self.n_components,
            "whiten": self.whiten,
            "scaler_mean": None if self.scaler is None else self.scaler.mean_.tolist(),
            "scaler_scale": None if self.scaler is None else self.scaler.scale_.tolist(),
            "pca_components": None if self.pca is None else self.pca.components_.tolist(),
            "pca_mean": None if self.pca is None else self.pca.mean_.tolist(),
            "pca_explained_variance_ratio": None if self.pca is None else self.pca.explained_variance_ratio_.tolist(),
            "tfidf_vocab_size": None if self.tfidf_vectorizer is None else int(len(self.tfidf_vectorizer.vocabulary_)),
            "count_vocab_size": None if self.count_vectorizer is None else int(len(self.count_vectorizer.vocabulary_)),
            "svd_explained_variance_ratio": None if self.svd is None else self.svd.explained_variance_ratio_.tolist(),
        }
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _compute_base_features(self, windows: WindowBundle) -> np.ndarray:
        if self.method == "raw_flatten":
            return self._raw_flatten(windows)
        if self.method == "event_count":
            return self._event_count(windows)
        if self.method == "bigram_count":
            return self._bigram_count(windows)
        if self.method == "summary_stats":
            return self._summary_stats(windows)
        if self.method == "tfidf_ngram":
            raise RuntimeError("tfidf_ngram should be handled in fit_transform().")
        if self.method == "component_aware_ngram":
            raise RuntimeError("component_aware_ngram should be handled in fit_transform().")
        if self.method == "pca":
            return self._raw_flatten(windows)
        raise ValueError(f"Unknown representation method: {self.method}")

    @staticmethod
    def _window_docs(windows: WindowBundle) -> np.ndarray:
        docs: list[str] = []
        mask = windows.mask.astype(bool)
        for row_idx in range(windows.X.shape[0]):
            valid = windows.X[row_idx][mask[row_idx]]
            tokens = [
                f"e{int(event[0])}|c{int(event[1])}|sev{int(event[2])}|st{int(event[3])}"
                for event in valid
            ]
            docs.append(" ".join(tokens))
        return np.array(docs, dtype=object)

    @staticmethod
    def _raw_flatten(windows: WindowBundle) -> np.ndarray:
        flat = windows.X * windows.mask[..., None]
        return flat.reshape(flat.shape[0], -1).astype(np.float32)

    @staticmethod
    def _event_count(windows: WindowBundle) -> np.ndarray:
        event_type_ids = windows.X[:, :, 0].astype(np.int64)
        mask = windows.mask.astype(bool)
        num_event_types = int(np.max(event_type_ids)) + 1
        counts = np.zeros((windows.X.shape[0], num_event_types), dtype=np.float32)
        for row_idx in range(windows.X.shape[0]):
            valid_ids = event_type_ids[row_idx][mask[row_idx]]
            bincount = np.bincount(valid_ids, minlength=num_event_types)
            counts[row_idx] = bincount.astype(np.float32)
        return counts

    @staticmethod
    def _bigram_count(windows: WindowBundle) -> np.ndarray:
        event_type_ids = windows.X[:, :, 0].astype(np.int64)
        mask = windows.mask.astype(bool)
        num_event_types = int(np.max(event_type_ids)) + 1
        counts = np.zeros((windows.X.shape[0], num_event_types * num_event_types), dtype=np.float32)
        for row_idx in range(windows.X.shape[0]):
            valid_ids = event_type_ids[row_idx][mask[row_idx]]
            for left, right in zip(valid_ids[:-1], valid_ids[1:]):
                counts[row_idx, left * num_event_types + right] += 1.0
        return counts

    @staticmethod
    def _summary_stats(windows: WindowBundle) -> np.ndarray:
        features = []
        mask = windows.mask.astype(bool)
        for row_idx in range(windows.X.shape[0]):
            valid = windows.X[row_idx][mask[row_idx]]
            event_ids = valid[:, 0]
            component_ids = valid[:, 1]
            severity_ids = valid[:, 2]
            state_ids = valid[:, 3]
            sensor = valid[:, 4]
            control = valid[:, 5]
            row = [
                float(len(valid)),
                float(np.mean(event_ids)),
                float(np.std(event_ids)),
                float(np.mean(component_ids)),
                float(np.mean(severity_ids)),
                float(np.mean(state_ids)),
                float(np.mean(sensor)),
                float(np.max(sensor)),
                float(np.std(sensor)),
                float(np.mean(control)),
                float(np.max(control)),
                float(np.std(control)),
                float(np.count_nonzero(sensor)),
                float(np.count_nonzero(control)),
            ]
            features.append(row)
        return np.array(features, dtype=np.float32)
