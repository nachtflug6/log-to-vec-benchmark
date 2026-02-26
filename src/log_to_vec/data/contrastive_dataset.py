"""
Dataset implementation for contrastive learning on sequence data.

This dataset is designed for NPZ files that contain a 'sequences' array
with shape (N, T, D), where:

    N = number of sequences
    T = sequence length
    D = feature dimension

It supports different positive-pair construction strategies:
    - "neighbor": use consecutive sequences (i, i+1)
    - "augment": apply a transformation to create a second view

"""

import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Optional, Callable, Dict, Any

class ContrastiveDataset(Dataset):
    """
    Dataset that returns pairs of sequences for contrastive learning.

    Each item is a dictionary:
        {
            "x1": Tensor (T, D),
            "x2": Tensor (T, D),
            "meta": Optional metadata
        }
    """

    def __init__(
        self,
        npz_path: str,
        pair_mode: str = "neighbor",
        transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        normalize: bool = False,
        return_meta: bool = False,
        dtype: torch.dtype = torch.float32):
        """
        Args:
            npz_path: Path to the .npz file containing 'sequences'
            pair_mode: Strategy for generating positive pairs.
                       Options:
                           - "neighbor"
                           - "augment"
            transform: Transformation function for augmentation (required if pair_mode="augment")
            normalize: Whether to apply global z-score normalization
            return_meta: Whether to return metadata if available
            dtype: Tensor dtype
        """
        super().__init__()

        data = np.load(npz_path)
        sequences = data["sequences"]

        self.sequences = torch.tensor(sequences, dtype=dtype)
        self.N = self.sequences.shape[0]
        self.pair_mode = pair_mode
        self.transform = transform
        self.return_meta = return_meta

        if self.pair_mode not in ("neighbor", "augment"):
            raise ValueError(
                f"Unsupported pair_mode: {self.pair_mode}. "
                f"Use 'neighbor' or 'augment'."
            )

        if self.pair_mode == "augment" and self.transform is None:
            raise ValueError("pair_mode='augment' requires a transform function.")

        # Optional normalization
        if normalize:
            mean, std = self.compute_norm_stats(indices=None)
            self.apply_norm_stats(mean=mean, std=std)

        # Optional metadata
        self.meta = {}
        for key in data.files:
            if key != "sequences":
                self.meta[key] = data[key]

    def compute_norm_stats(self, indices=None):
        """
        Compute z-score normalization stats.
        If indices is None -> compute over all sequences.
        If indices is a list/np array/torch tensor -> compute using only those sequence indices.
        Returns:
            mean: Tensor shape (1, 1, D)
            std:  Tensor shape (1, 1, D)
        """
        if indices is None:
            x = self.sequences  # (N, T, D)
        else:
            if isinstance(indices, torch.Tensor):
                idx = indices.long()
            else:
                idx = torch.tensor(indices, dtype=torch.long)
            x = self.sequences.index_select(0, idx)

        mean = x.mean(dim=(0, 1), keepdim=True)
        std = x.std(dim=(0, 1), keepdim=True) + 1e-8
        return mean, std

    def apply_norm_stats(self, mean: torch.Tensor, std: torch.Tensor):
        """
        Apply precomputed normalization stats to all sequences in the dataset.
        Args:
            mean: Tensor shape (1, 1, D)
            std:  Tensor shape (1, 1, D)
        """
        self.sequences = (self.sequences - mean) / std


    def __len__(self) -> int:
        """
        Return number of available training pairs.
        """
        if self.pair_mode == "neighbor":
            return self.N - 1
        return self.N

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Return a pair of sequences for contrastive learning.
        """
        x1 = self.sequences[idx]

        if self.pair_mode == "neighbor":
            x2 = self.sequences[idx + 1]

        elif self.pair_mode == "augment":
            x2 = self.transform(x1.clone())

        output = {
            "x1": x1,
            "x2": x2,
        }

        if self.return_meta and self.meta:
            meta_dict = {}
            for key, value in self.meta.items():
                if len(value) == self.N:
                    meta_dict[key] = value[idx]
            output["meta"] = meta_dict

        return output