"""Baseline feature extraction helpers for RQ1."""

from __future__ import annotations

import numpy as np


def flatten_features(x: np.ndarray) -> np.ndarray:
    """Flatten windows [N, T, C] into [N, T*C]."""
    if x.ndim != 3:
        raise ValueError(f"Expected X with shape [N, T, C], got {x.shape}")
    return x.reshape(x.shape[0], -1).astype(np.float32)


def summary_features(x: np.ndarray) -> np.ndarray:
    """Simple per-channel summary statistics and temporal change features."""
    if x.ndim != 3:
        raise ValueError(f"Expected X with shape [N, T, C], got {x.shape}")

    mean = x.mean(axis=1)
    std = x.std(axis=1)
    minimum = x.min(axis=1)
    maximum = x.max(axis=1)
    first = x[:, 0, :]
    last = x[:, -1, :]
    slope = last - first
    return np.concatenate([mean, std, minimum, maximum, slope], axis=1).astype(np.float32)


def fft_features(x: np.ndarray, keep_bins: int = 8) -> np.ndarray:
    """FFT magnitude features using the first non-DC frequency bins."""
    if x.ndim != 3:
        raise ValueError(f"Expected X with shape [N, T, C], got {x.shape}")
    spectrum = np.abs(np.fft.rfft(x, axis=1))
    spectrum = spectrum[:, 1 : keep_bins + 1, :]
    return spectrum.reshape(x.shape[0], -1).astype(np.float32)


def build_feature_set(x: np.ndarray, feature_set: str) -> np.ndarray:
    if feature_set == "summary":
        return summary_features(x)
    if feature_set == "fft":
        return fft_features(x)
    if feature_set == "raw_flatten":
        return flatten_features(x)
    raise ValueError(f"Unsupported feature set: {feature_set}")
