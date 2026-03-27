from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


class FSSSWindowDataset(Dataset):
    """
    Dataset for processed FSSS window splits.

    Expected .npz keys:
        X
        trajectory_id
        device_id
        window_start
        mode_id
        spectral_id
        coupling_id
        mean_load
        is_transition_window
        distance_to_boundary
        left_mode_id
        right_mode_id
    """

    def __init__(self, npz_path: str):
        data = np.load(npz_path)
        required = [
            "X",
            "trajectory_id",
            "device_id",
            "window_start",
            "mode_id",
            "spectral_id",
            "coupling_id",
            "mean_load",
            "is_transition_window",
            "distance_to_boundary",
            "left_mode_id",
            "right_mode_id",
        ]
        missing = [k for k in required if k not in data]
        if missing:
            raise ValueError(f"Missing keys in {npz_path}: {missing}")

        self.X = data["X"].astype(np.float32)
        self.trajectory_id = data["trajectory_id"].astype(np.int64)
        self.device_id = data["device_id"].astype(np.int64)
        self.window_start = data["window_start"].astype(np.int64)
        self.mode_id = data["mode_id"].astype(np.int64)
        self.spectral_id = data["spectral_id"].astype(np.int64)
        self.coupling_id = data["coupling_id"].astype(np.int64)
        self.mean_load = data["mean_load"].astype(np.float32)
        self.is_transition_window = data["is_transition_window"].astype(np.bool_)
        self.distance_to_boundary = data["distance_to_boundary"].astype(np.int64)
        self.left_mode_id = data["left_mode_id"].astype(np.int64)
        self.right_mode_id = data["right_mode_id"].astype(np.int64)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        return {
            "x": torch.from_numpy(self.X[idx]),
            "index": idx,
            "trajectory_id": int(self.trajectory_id[idx]),
            "device_id": int(self.device_id[idx]),
            "window_start": int(self.window_start[idx]),
            "mode_id": int(self.mode_id[idx]),
            "spectral_id": int(self.spectral_id[idx]),
            "coupling_id": int(self.coupling_id[idx]),
            "mean_load": float(self.mean_load[idx]),
            "is_transition_window": bool(self.is_transition_window[idx]),
            "distance_to_boundary": int(self.distance_to_boundary[idx]),
            "left_mode_id": int(self.left_mode_id[idx]),
            "right_mode_id": int(self.right_mode_id[idx]),
        }
