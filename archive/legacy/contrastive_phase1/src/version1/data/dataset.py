"""
Load .npz -> read X -> convert to tensor
"""


import numpy as np
import torch
from torch.utils.data import Dataset


class SequenceDataset(Dataset):
    """
    Sequence dataset.

    This dataset is used for:
    - autoencoder training
    - transformer training
    - embedding extraction
    - evaluation
    - visualization

    Expected .npz format:
        X: [N, L, D]
    """

    def __init__(self, npz_path: str):
        data = np.load(npz_path)

        if "X" not in data:
            raise ValueError(f"'X' not found in {npz_path}")

        self.X = data["X"].astype(np.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = torch.from_numpy(self.X[idx])

        return {
            "x": x,          # shape: [L, D]
            "index": idx
        }


class ContrastiveDataset(Dataset):
    """
    Unified contrastive dataset supporting multiple modes.

    Modes:
    - "identity": same sample (x_i == x_j)
    - "augment": two augmented views

    Expected .npz:
        X: [N, L, D]
    """

    def __init__(
        self,
        npz_path: str,
        mode: str = "augment",
        jitter_std: float = 0.02,
        scaling_range=(0.9, 1.1),
        feature_dropout_rate: float = 0.05,
        time_mask_rate: float = 0.10,
        apply_augment: bool = True,
        seed: int = 42,
    ):
        data = np.load(npz_path)

        if "X" not in data:
            raise ValueError(f"'X' not found in {npz_path}")

        self.X = data["X"].astype(np.float32)

        self.mode = mode
        self.apply_augment = apply_augment

        self.jitter_std = float(jitter_std)
        self.scaling_range = scaling_range
        self.feature_dropout_rate = float(feature_dropout_rate)
        self.time_mask_rate = float(time_mask_rate)

        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.X)

    # ----------------------------
    # augmentations
    # ----------------------------
    def _jitter(self, x):
        if self.jitter_std <= 0:
            return x
        noise = self.rng.normal(0, self.jitter_std, size=x.shape).astype(np.float32)
        return x + noise

    def _scaling(self, x):
        low, high = self.scaling_range
        if low == 1.0 and high == 1.0:
            return x
        scale = np.float32(self.rng.uniform(low, high))
        return x * scale

    def _feature_dropout(self, x):
        if self.feature_dropout_rate <= 0:
            return x
        mask = self.rng.random(x.shape) > self.feature_dropout_rate
        return x * mask.astype(np.float32)

    def _time_mask(self, x):
        if self.time_mask_rate <= 0:
            return x
        L = x.shape[0]
        keep = self.rng.random(L) > self.time_mask_rate
        keep = keep.astype(np.float32)[:, None]
        return x * keep

    def _augment(self, x):
        x = x.copy()
        x = self._jitter(x)
        x = self._scaling(x)
        x = self._feature_dropout(x)
        x = self._time_mask(x)
        return x.astype(np.float32)

    # ----------------------------
    # main logic
    # ----------------------------
    def __getitem__(self, idx):
        x = self.X[idx]

        # -------- baseline --------
        if self.mode == "identity":
            x_i = x.copy()
            x_j = x.copy()

        # -------- augment --------
        elif self.mode == "augment":
            if self.apply_augment:
                x_i = self._augment(x)
                x_j = self._augment(x)
            else:
                x_i = x.copy()
                x_j = x.copy()

        # -------- others (future) --------
        elif self.mode == "other":
            # TODO: future extension
            pass

        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        return {
            "x_i": torch.from_numpy(x_i),
            "x_j": torch.from_numpy(x_j),
            "index": idx
        }