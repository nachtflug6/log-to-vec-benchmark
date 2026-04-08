from __future__ import annotations

import numpy as np


def flatten_features(X: np.ndarray) -> np.ndarray:
    """Flatten windows [N, L, C] -> [N, L*C]."""
    if X.ndim != 3:
        raise ValueError(f"Expected X with shape [N, L, C], got {X.shape}")
    return X.reshape(X.shape[0], -1).astype(np.float32)


def summary_stat_features(X: np.ndarray) -> np.ndarray:
    """Simple moments + temporal change features per channel."""
    if X.ndim != 3:
        raise ValueError(f"Expected X with shape [N, L, C], got {X.shape}")

    mean = X.mean(axis=1)
    std = X.std(axis=1)
    min_ = X.min(axis=1)
    max_ = X.max(axis=1)
    ptp = max_ - min_
    q25 = np.percentile(X, 25, axis=1)
    q50 = np.percentile(X, 50, axis=1)
    q75 = np.percentile(X, 75, axis=1)

    dx = np.diff(X, axis=1)
    if dx.shape[1] == 0:
        dx_mean = np.zeros_like(mean)
        dx_std = np.zeros_like(std)
        energy_diff = np.zeros_like(mean)
    else:
        dx_mean = dx.mean(axis=1)
        dx_std = dx.std(axis=1)
        energy_diff = (dx ** 2).mean(axis=1)

    energy = (X ** 2).mean(axis=1)
    zcr = _zero_crossing_rate(X)
    ac_lag1 = _autocorr_lag(X, lag=1)
    ac_lag2 = _autocorr_lag(X, lag=2)
    ac_lag4 = _autocorr_lag(X, lag=4)

    feats = np.concatenate(
        [
            mean, std, min_, max_, ptp,
            q25, q50, q75,
            dx_mean, dx_std,
            energy, energy_diff,
            zcr,
            ac_lag1, ac_lag2, ac_lag4,
        ],
        axis=1,
    )
    return feats.astype(np.float32)


def fft_features(X: np.ndarray, keep_bins: int = 8) -> np.ndarray:
    """Lightweight FFT magnitude features per channel.

    Keeps the first few non-DC bins and a few summary ratios.
    """
    if X.ndim != 3:
        raise ValueError(f"Expected X with shape [N, L, C], got {X.shape}")

    Xc = X - X.mean(axis=1, keepdims=True)
    fft_mag = np.abs(np.fft.rfft(Xc, axis=1)).astype(np.float32)  # [N, F, C]

    # Remove DC bin if present
    if fft_mag.shape[1] > 1:
        fft_nodc = fft_mag[:, 1:, :]
    else:
        fft_nodc = fft_mag

    k = min(keep_bins, fft_nodc.shape[1])
    first_bins = fft_nodc[:, :k, :]  # [N, k, C]
    first_bins = np.transpose(first_bins, (0, 2, 1)).reshape(X.shape[0], -1)

    total_energy = fft_nodc.sum(axis=1) + 1e-8
    peak_energy = fft_nodc.max(axis=1)
    peak_ratio = peak_energy / total_energy
    peak_bin = np.argmax(fft_nodc, axis=1).astype(np.float32)

    spectral_centroid = _spectral_centroid(fft_nodc)
    spectral_bandwidth = _spectral_bandwidth(fft_nodc, spectral_centroid)

    feats = np.concatenate(
        [
            first_bins,
            total_energy,
            peak_energy,
            peak_ratio,
            peak_bin,
            spectral_centroid,
            spectral_bandwidth,
        ],
        axis=1,
    )
    return feats.astype(np.float32)


def build_feature_set(X: np.ndarray, feature_type: str) -> np.ndarray:
    if feature_type == "raw_flatten":
        return flatten_features(X)
    if feature_type == "summary":
        return summary_stat_features(X)
    if feature_type == "fft":
        return fft_features(X)
    raise ValueError(f"Unknown feature_type: {feature_type}")


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def _zero_crossing_rate(X: np.ndarray) -> np.ndarray:
    signs = np.sign(X)
    zc = (signs[:, 1:, :] * signs[:, :-1, :] < 0).mean(axis=1)
    return zc.astype(np.float32)


def _autocorr_lag(X: np.ndarray, lag: int) -> np.ndarray:
    N, L, C = X.shape
    if lag >= L:
        return np.zeros((N, C), dtype=np.float32)

    a = X[:, :-lag, :]
    b = X[:, lag:, :]
    a = a - a.mean(axis=1, keepdims=True)
    b = b - b.mean(axis=1, keepdims=True)

    num = (a * b).sum(axis=1)
    den = np.sqrt((a * a).sum(axis=1) * (b * b).sum(axis=1)) + 1e-8
    return (num / den).astype(np.float32)


def _spectral_centroid(fft_mag: np.ndarray) -> np.ndarray:
    bins = np.arange(fft_mag.shape[1], dtype=np.float32)[None, :, None]
    num = (fft_mag * bins).sum(axis=1)
    den = fft_mag.sum(axis=1) + 1e-8
    return (num / den).astype(np.float32)


def _spectral_bandwidth(fft_mag: np.ndarray, centroid: np.ndarray) -> np.ndarray:
    bins = np.arange(fft_mag.shape[1], dtype=np.float32)[None, :, None]
    diff2 = (bins - centroid[:, None, :]) ** 2
    num = (fft_mag * diff2).sum(axis=1)
    den = fft_mag.sum(axis=1) + 1e-8
    return np.sqrt(num / den).astype(np.float32)
