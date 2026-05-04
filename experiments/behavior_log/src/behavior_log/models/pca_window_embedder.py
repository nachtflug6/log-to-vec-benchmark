"""Trainable PCA baseline for window embeddings."""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


@dataclass
class PCAWindowEmbedder:
    """Scale flattened windows and project them with PCA."""

    n_components: int = 8
    whiten: bool = False

    def __post_init__(self) -> None:
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=self.n_components, whiten=self.whiten, random_state=0)
        self.fitted = False

    def fit(self, X_windows: np.ndarray, mask: np.ndarray | None = None) -> "PCAWindowEmbedder":
        flat = self._flatten_windows(X_windows, mask)
        scaled = self.scaler.fit_transform(flat)
        self.pca.fit(scaled)
        self.fitted = True
        return self

    def transform(self, X_windows: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("Model must be fitted before transform().")
        flat = self._flatten_windows(X_windows, mask)
        scaled = self.scaler.transform(flat)
        return self.pca.transform(scaled).astype(np.float32)

    def fit_transform(self, X_windows: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
        return self.fit(X_windows, mask).transform(X_windows, mask)

    def training_summary(self) -> dict:
        if not self.fitted:
            raise RuntimeError("Model must be fitted before requesting summary.")
        return {
            "n_components": int(self.pca.n_components_),
            "explained_variance_ratio": self.pca.explained_variance_ratio_.tolist(),
            "total_explained_variance_ratio": float(np.sum(self.pca.explained_variance_ratio_)),
        }

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str | Path) -> "PCAWindowEmbedder":
        with Path(path).open("rb") as f:
            obj = pickle.load(f)
        return obj

    @staticmethod
    def _flatten_windows(X_windows: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
        flat = X_windows.copy()
        if mask is not None:
            flat = flat * mask[..., None]
        return flat.reshape(flat.shape[0], -1)
