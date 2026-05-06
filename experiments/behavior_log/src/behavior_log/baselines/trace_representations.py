"""Baseline embeddings for trace-level prepared datasets."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import StandardScaler


@dataclass
class TraceRepresentationResult:
    embeddings: np.ndarray
    summary: dict


class TraceRepresentationBuilder:
    def __init__(
        self,
        *,
        method: str,
        n_components: int = 32,
        whiten: bool = False,
        ngram_min: int = 1,
        ngram_max: int = 3,
    ) -> None:
        self.method = method
        self.n_components = n_components
        self.whiten = whiten
        self.ngram_min = ngram_min
        self.ngram_max = ngram_max
        self.scaler: StandardScaler | None = None
        self.pca: PCA | None = None
        self.tfidf_vectorizer: TfidfVectorizer | None = None
        self.count_vectorizer: CountVectorizer | None = None
        self.svd: TruncatedSVD | None = None

    def fit_transform(
        self,
        *,
        traces_df: pd.DataFrame,
        occurrence_df: pd.DataFrame,
        split: np.ndarray,
    ) -> TraceRepresentationResult:
        train_mask = split == "train"

        if self.method == "occurrence_identity":
            features = self._occurrence_features(occurrence_df)
            return TraceRepresentationResult(
                embeddings=features.astype(np.float32),
                summary={
                    "method": self.method,
                    "input_dim": int(features.shape[1]),
                    "embedding_dim": int(features.shape[1]),
                },
            )

        if self.method == "occurrence_pca":
            features = self._occurrence_features(occurrence_df)
            self.scaler = StandardScaler()
            train_scaled = self.scaler.fit_transform(features[train_mask])
            n_components = min(self.n_components, train_scaled.shape[0], train_scaled.shape[1])
            self.pca = PCA(n_components=n_components, whiten=self.whiten, random_state=0)
            self.pca.fit(train_scaled)
            embeddings = self.pca.transform(self.scaler.transform(features)).astype(np.float32)
            return TraceRepresentationResult(
                embeddings=embeddings,
                summary={
                    "method": self.method,
                    "input_dim": int(features.shape[1]),
                    "embedding_dim": int(embeddings.shape[1]),
                    "explained_variance_ratio": self.pca.explained_variance_ratio_.tolist(),
                    "total_explained_variance_ratio": float(np.sum(self.pca.explained_variance_ratio_)),
                },
            )

        docs = traces_df["sequence"].fillna("").astype(str).to_numpy()

        if self.method == "tfidf_svd":
            self.tfidf_vectorizer = TfidfVectorizer(
                analyzer="word",
                token_pattern=r"[^ ]+",
                lowercase=False,
                ngram_range=(self.ngram_min, self.ngram_max),
            )
            train_matrix = self.tfidf_vectorizer.fit_transform(docs[train_mask])
            all_matrix = self.tfidf_vectorizer.transform(docs)
            svd_dim = min(self.n_components, train_matrix.shape[0], train_matrix.shape[1])
            self.svd = TruncatedSVD(n_components=svd_dim, random_state=0)
            self.svd.fit(train_matrix)
            embeddings = self.svd.transform(all_matrix).astype(np.float32)
            return TraceRepresentationResult(
                embeddings=embeddings,
                summary={
                    "method": self.method,
                    "input_dim": int(all_matrix.shape[1]),
                    "embedding_dim": int(embeddings.shape[1]),
                    "ngram_range": [self.ngram_min, self.ngram_max],
                    "vocab_size": int(len(self.tfidf_vectorizer.vocabulary_)),
                    "explained_variance_ratio": self.svd.explained_variance_ratio_.tolist(),
                    "total_explained_variance_ratio": float(np.sum(self.svd.explained_variance_ratio_)),
                },
            )

        if self.method == "count_svd":
            self.count_vectorizer = CountVectorizer(
                analyzer="word",
                token_pattern=r"[^ ]+",
                lowercase=False,
                ngram_range=(self.ngram_min, self.ngram_max),
            )
            train_matrix = self.count_vectorizer.fit_transform(docs[train_mask])
            all_matrix = self.count_vectorizer.transform(docs)
            svd_dim = min(self.n_components, train_matrix.shape[0], train_matrix.shape[1])
            self.svd = TruncatedSVD(n_components=svd_dim, random_state=0)
            self.svd.fit(train_matrix)
            embeddings = self.svd.transform(all_matrix).astype(np.float32)
            return TraceRepresentationResult(
                embeddings=embeddings,
                summary={
                    "method": self.method,
                    "input_dim": int(all_matrix.shape[1]),
                    "embedding_dim": int(embeddings.shape[1]),
                    "ngram_range": [self.ngram_min, self.ngram_max],
                    "vocab_size": int(len(self.count_vectorizer.vocabulary_)),
                    "explained_variance_ratio": self.svd.explained_variance_ratio_.tolist(),
                    "total_explained_variance_ratio": float(np.sum(self.svd.explained_variance_ratio_)),
                },
            )

        raise ValueError(f"Unknown baseline representation method: {self.method}")

    def save(self, path: str | Path) -> None:
        payload = {
            "method": self.method,
            "n_components": self.n_components,
            "whiten": self.whiten,
            "ngram_min": self.ngram_min,
            "ngram_max": self.ngram_max,
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

    @staticmethod
    def _occurrence_features(occurrence_df: pd.DataFrame) -> np.ndarray:
        feature_columns = [column for column in occurrence_df.columns if column not in {"sample_id", "label", "split"}]
        return occurrence_df[feature_columns].to_numpy(dtype=np.float32)
