"""Convert event-level features into trace windows and splits."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
from sklearn.model_selection import train_test_split

from behavior_log.preprocessing.event_preprocessor import EventFeatureBundle


@dataclass
class WindowBundle:
    """Container for window features and labels."""

    X: np.ndarray
    mask: np.ndarray
    window_behavior_id: np.ndarray
    trajectory_id: np.ndarray
    is_transition_window: np.ndarray
    window_start: np.ndarray
    split: np.ndarray
    feature_names: np.ndarray


class WindowDatasetBuilder:
    """Create fixed-length windows with trace-level splits."""

    def __init__(self, window_length: int, stride: int = 1, pad_short_traces: bool = True, seed: int = 42) -> None:
        if window_length <= 0:
            raise ValueError("window_length must be positive.")
        if stride <= 0:
            raise ValueError("stride must be positive.")
        self.window_length = window_length
        self.stride = stride
        self.pad_short_traces = pad_short_traces
        self.seed = seed

    def build_windows(self, bundle: EventFeatureBundle) -> WindowBundle:
        """Create windows from event-level features."""

        X_list: List[np.ndarray] = []
        mask_list: List[np.ndarray] = []
        behavior_ids: List[int] = []
        trajectory_ids: List[int] = []
        transition_flags: List[int] = []
        window_starts: List[int] = []

        unique_trajectory_ids = np.unique(bundle.trajectory_id)
        for trajectory_id in unique_trajectory_ids:
            indices = np.where(bundle.trajectory_id == trajectory_id)[0]
            trace_X = bundle.X[indices]
            trace_length = len(indices)

            if trace_length < self.window_length and not self.pad_short_traces:
                continue

            if trace_length < self.window_length:
                padded = np.zeros((self.window_length, trace_X.shape[1]), dtype=np.float32)
                padded[:trace_length] = trace_X
                mask = np.zeros(self.window_length, dtype=np.float32)
                mask[:trace_length] = 1.0
                X_list.append(padded)
                mask_list.append(mask)
                behavior_ids.append(self._majority_label(bundle.behavior_id[indices]))
                trajectory_ids.append(int(trajectory_id))
                transition_flags.append(int(self._is_transition_window(bundle.behavior_id[indices], bundle.is_transition[indices])))
                window_starts.append(0)
                continue

            for start in range(0, trace_length - self.window_length + 1, self.stride):
                end = start + self.window_length
                X_list.append(trace_X[start:end])
                mask_list.append(np.ones(self.window_length, dtype=np.float32))
                behavior_ids.append(self._majority_label(bundle.behavior_id[indices][start:end]))
                trajectory_ids.append(int(trajectory_id))
                transition_flags.append(
                    int(
                        self._is_transition_window(
                            bundle.behavior_id[indices][start:end],
                            bundle.is_transition[indices][start:end],
                        )
                    )
                )
                window_starts.append(start)

        split = np.array(["unassigned"] * len(X_list), dtype=object)
        return WindowBundle(
            X=np.stack(X_list).astype(np.float32),
            mask=np.stack(mask_list).astype(np.float32),
            window_behavior_id=np.array(behavior_ids, dtype=np.int64),
            trajectory_id=np.array(trajectory_ids, dtype=np.int64),
            is_transition_window=np.array(transition_flags, dtype=np.int64),
            window_start=np.array(window_starts, dtype=np.int64),
            split=split,
            feature_names=bundle.feature_names,
        )

    def assign_trace_splits(
        self,
        windows: WindowBundle,
        train_ratio: float,
        val_ratio: float,
        test_ratio: float,
    ) -> tuple[WindowBundle, Dict[str, List[int]]]:
        """Assign train/val/test based on unique trace ids."""

        total = train_ratio + val_ratio + test_ratio
        if not np.isclose(total, 1.0):
            raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0.")

        unique_trajectory_ids = np.unique(windows.trajectory_id)
        unique_behaviors = np.array(
            [
                self._majority_label(windows.window_behavior_id[np.where(windows.trajectory_id == trajectory_id)[0]])
                for trajectory_id in unique_trajectory_ids
            ]
        )

        train_ids, temp_ids, train_y, temp_y = train_test_split(
            unique_trajectory_ids,
            unique_behaviors,
            test_size=(1.0 - train_ratio),
            random_state=self.seed,
            stratify=self._safe_stratify_labels(unique_behaviors),
        )

        val_portion = val_ratio / (val_ratio + test_ratio)
        val_ids, test_ids = train_test_split(
            temp_ids,
            test_size=(1.0 - val_portion),
            random_state=self.seed,
            stratify=self._safe_stratify_labels(temp_y),
        )

        split = windows.split.copy()
        split[np.isin(windows.trajectory_id, train_ids)] = "train"
        split[np.isin(windows.trajectory_id, val_ids)] = "val"
        split[np.isin(windows.trajectory_id, test_ids)] = "test"

        assigned = WindowBundle(
            X=windows.X,
            mask=windows.mask,
            window_behavior_id=windows.window_behavior_id,
            trajectory_id=windows.trajectory_id,
            is_transition_window=windows.is_transition_window,
            window_start=windows.window_start,
            split=split,
            feature_names=windows.feature_names,
        )
        manifest = {
            "train_trajectory_ids": [int(x) for x in sorted(train_ids.tolist())],
            "val_trajectory_ids": [int(x) for x in sorted(val_ids.tolist())],
            "test_trajectory_ids": [int(x) for x in sorted(test_ids.tolist())],
        }
        return assigned, manifest

    @staticmethod
    def save_bundle(bundle: WindowBundle, path: str | Path) -> None:
        """Save windows to NPZ."""

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            X=bundle.X,
            mask=bundle.mask,
            window_behavior_id=bundle.window_behavior_id,
            trajectory_id=bundle.trajectory_id,
            is_transition_window=bundle.is_transition_window,
            window_start=bundle.window_start,
            split=bundle.split,
            feature_names=bundle.feature_names,
        )

    @staticmethod
    def load_bundle(path: str | Path) -> WindowBundle:
        """Load windows from NPZ."""

        data = np.load(Path(path), allow_pickle=True)
        return WindowBundle(
            X=data["X"],
            mask=data["mask"],
            window_behavior_id=data["window_behavior_id"],
            trajectory_id=data["trajectory_id"],
            is_transition_window=data["is_transition_window"],
            window_start=data["window_start"],
            split=data["split"],
            feature_names=data["feature_names"],
        )

    @staticmethod
    def save_split_manifest(manifest: Dict[str, List[int]], path: str | Path) -> None:
        """Save split ids to JSON."""

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    @staticmethod
    def _safe_stratify_labels(labels: np.ndarray) -> np.ndarray | None:
        """Disable stratification when a tiny sample would make it invalid."""

        if len(labels) < 2:
            return None
        _, counts = np.unique(labels, return_counts=True)
        if np.min(counts) < 2:
            return None
        return labels

    @staticmethod
    def _majority_label(labels: np.ndarray) -> int:
        values, counts = np.unique(labels.astype(np.int64), return_counts=True)
        return int(values[np.argmax(counts)])

    @staticmethod
    def _is_transition_window(labels: np.ndarray, flags: np.ndarray) -> bool:
        return bool(len(np.unique(labels.astype(np.int64))) > 1 or np.any(flags > 0))
