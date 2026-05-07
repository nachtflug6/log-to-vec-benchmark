"""Standalone ROCKET implementation using numba directly (no sktime dependency).

Implements the random-convolutional-kernel transform from Dempster et al. (2020).
Each kernel: length-9, weights ∈ {−1, 2} centered, random dilation, bias from
training-data quantile.  Feature = PPV (proportion of positive values in output).

Requires numba (available in the codeserver-PyTorch container).
"""
from __future__ import annotations

import numpy as np

try:
    import numba as nb
    _HAVE_NUMBA = True
except ImportError:
    _HAVE_NUMBA = False


# ---------------------------------------------------------------------------
# Numba-accelerated kernel application
# ---------------------------------------------------------------------------

def _build_numba_kernel():
    import numba as nb

    @nb.njit(fastmath=True, parallel=True)
    def _ppv_batch(X_ch, weights, dilation, bias):
        """PPV of one dilated kernel applied to X_ch [N, L].  Returns [N] float32."""
        N, L = X_ch.shape
        klen = weights.shape[0]
        out_len = L - (klen - 1) * dilation
        if out_len <= 0:
            return np.zeros(N, dtype=nb.float32)
        result = np.zeros(N, dtype=nb.float32)
        for n in nb.prange(N):
            n_pos = 0
            for t in range(out_len):
                v = nb.float32(0.0)
                for j in range(klen):
                    v += weights[j] * X_ch[n, t + j * dilation]
                if v > bias:
                    n_pos += 1
            result[n] = nb.float32(n_pos) / nb.float32(out_len)
        return result

    return _ppv_batch


# ---------------------------------------------------------------------------
# Pure-numpy fallback (slower but always works)
# ---------------------------------------------------------------------------

def _ppv_numpy(X_ch: np.ndarray, weights: np.ndarray, dilation: int, bias: float) -> np.ndarray:
    N, L = X_ch.shape
    klen = len(weights)
    out_len = L - (klen - 1) * dilation
    if out_len <= 0:
        return np.zeros(N, dtype=np.float32)
    patches = np.stack([X_ch[:, j * dilation: j * dilation + out_len] for j in range(klen)], axis=-1)
    conv = (patches * weights[np.newaxis, np.newaxis, :]).sum(axis=-1)  # [N, out_len]
    return (conv > bias).mean(axis=1).astype(np.float32)


# ---------------------------------------------------------------------------
# Main ROCKET feature extractor
# ---------------------------------------------------------------------------

def rocket_features(
    X_train: np.ndarray,
    X_all: np.ndarray,
    num_kernels: int = 10_000,
    random_state: int = 42,
) -> np.ndarray:
    """Compute ROCKET PPV features.

    X_train: [N_train, L, C] — used only to sample bias quantiles
    X_all:   [N_all,  L, C] — features extracted for all samples
    Returns: [N_all, num_kernels] float32
    """
    N_train, L, C = X_train.shape
    N_all = X_all.shape[0]
    rng = np.random.default_rng(random_state)

    if _HAVE_NUMBA:
        _ppv = _build_numba_kernel()
    else:
        _ppv = None

    # Pre-compute valid dilations for this window length
    klen = 9
    max_exp = max(0, int(np.floor(np.log2(max(1, (L - 1) // (klen - 1))))))
    dilations = [2 ** i for i in range(max_exp + 1)]

    features = np.empty((N_all, num_kernels), dtype=np.float32)

    for k in range(num_kernels):
        dilation = int(dilations[rng.integers(len(dilations))])
        channel  = int(rng.integers(C))

        # Kernel weights: random subset of {-1, 2}^9 centred
        w = rng.choice(np.array([-1.0, 2.0], dtype=np.float32), size=klen)
        w -= w.mean()
        w = w.astype(np.float32)

        # Bias from random quantile of training data
        q = float(rng.uniform(0.0, 1.0))
        X_ch_train = X_train[:, :, channel].astype(np.float32)
        out_len_train = L - (klen - 1) * dilation
        if out_len_train < 1:
            features[:, k] = 0.0
            continue
        # Compute one conv output on training set to sample quantile
        patches_train = np.stack(
            [X_ch_train[:, j * dilation: j * dilation + out_len_train] for j in range(klen)],
            axis=-1,
        )
        conv_train = (patches_train * w[np.newaxis, np.newaxis, :]).sum(axis=-1)
        bias = float(np.quantile(conv_train.ravel(), q))

        X_ch_all = X_all[:, :, channel].astype(np.float32)

        if _HAVE_NUMBA and _ppv is not None:
            features[:, k] = _ppv(X_ch_all, w, dilation, np.float32(bias))
        else:
            features[:, k] = _ppv_numpy(X_ch_all, w, dilation, bias)

    return features
