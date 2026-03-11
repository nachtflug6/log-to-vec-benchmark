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
    Contrastive dataset.

    This returns two identical views of the same sequence.

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
            "x_i": x.clone(),   # first view
            "x_j": x.clone(),   # second view
            "index": idx
        }