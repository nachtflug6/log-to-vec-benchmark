from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


REQUIRED_WINDOW_KEYS = {
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
}


@dataclass
class SplitConfig:
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    split_by: str = "trajectory"  # trajectory | device
    seed: int = 42


class FSSSDataError(Exception):
    pass


# -----------------------------------------------------------------------------
# Loading
# -----------------------------------------------------------------------------

def load_fsss_dataset(raw_dir: str | Path) -> Dict[str, object]:
    raw_dir = Path(raw_dir)
    traj_path = raw_dir / "trajectories.csv"
    windows_path = raw_dir / "windows.npz"
    meta_path = raw_dir / "metadata.json"

    if not traj_path.exists():
        raise FSSSDataError(f"Missing file: {traj_path}")
    if not windows_path.exists():
        raise FSSSDataError(f"Missing file: {windows_path}")
    if not meta_path.exists():
        raise FSSSDataError(f"Missing file: {meta_path}")

    trajectory_df = pd.read_csv(traj_path)
    with np.load(windows_path) as data:
        windows = {k: data[k] for k in data.files}
    with open(meta_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    missing = REQUIRED_WINDOW_KEYS - set(windows.keys())
    if missing:
        raise FSSSDataError(f"windows.npz is missing keys: {sorted(missing)}")

    if windows["X"].ndim != 3:
        raise FSSSDataError(f"Expected X to be 3D, got {windows['X'].shape}")

    n = len(windows["X"])
    for k, v in windows.items():
        if len(v) != n:
            raise FSSSDataError(f"All window arrays must have the same length. Key {k} has len={len(v)}, expected {n}")

    return {
        "trajectory_df": trajectory_df,
        "windows": windows,
        "metadata": metadata,
        "raw_dir": str(raw_dir),
    }


# -----------------------------------------------------------------------------
# Split logic
# -----------------------------------------------------------------------------

def _validate_ratios(cfg: SplitConfig) -> None:
    total = cfg.train_ratio + cfg.val_ratio + cfg.test_ratio
    if not np.isclose(total, 1.0):
        raise ValueError(f"Split ratios must sum to 1.0, got {total}")
    if cfg.split_by not in {"trajectory", "device"}:
        raise ValueError("split_by must be 'trajectory' or 'device'")



def _split_ids(unique_ids: np.ndarray, cfg: SplitConfig) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(cfg.seed)
    ids = np.array(unique_ids)
    rng.shuffle(ids)

    n = len(ids)
    n_train = int(n * cfg.train_ratio)
    n_val = int(n * cfg.val_ratio)

    train_ids = ids[:n_train]
    val_ids = ids[n_train:n_train + n_val]
    test_ids = ids[n_train + n_val:]
    return train_ids, val_ids, test_ids



def split_fsss_windows(dataset: Dict[str, object], cfg: SplitConfig) -> Dict[str, Dict[str, np.ndarray]]:
    _validate_ratios(cfg)
    windows = dataset["windows"]

    group_key = "trajectory_id" if cfg.split_by == "trajectory" else "device_id"
    unique_ids = np.unique(windows[group_key])

    train_ids, val_ids, test_ids = _split_ids(unique_ids, cfg)

    split_masks = {
        "train": np.isin(windows[group_key], train_ids),
        "val": np.isin(windows[group_key], val_ids),
        "test": np.isin(windows[group_key], test_ids),
    }

    split_data: Dict[str, Dict[str, np.ndarray]] = {}
    for split_name, mask in split_masks.items():
        split_data[split_name] = {
            k: v[mask] for k, v in windows.items()
        }

    return split_data


# -----------------------------------------------------------------------------
# Leakage checks and summaries
# -----------------------------------------------------------------------------

def _set_overlap(a: np.ndarray, b: np.ndarray) -> List[int]:
    return sorted(set(map(int, np.unique(a))).intersection(set(map(int, np.unique(b)))))



def summarize_split(split_windows: Dict[str, np.ndarray]) -> Dict[str, object]:
    if len(split_windows["X"]) == 0:
        return {
            "num_windows": 0,
            "num_trajectories": 0,
            "num_devices": 0,
            "mode_counts": {},
            "spectral_counts": {},
            "coupling_counts": {},
            "transition_fraction": 0.0,
            "mean_load": None,
        }

    def counts(arr: np.ndarray) -> Dict[str, int]:
        values, cnts = np.unique(arr, return_counts=True)
        return {str(int(v)): int(c) for v, c in zip(values, cnts)}

    return {
        "num_windows": int(len(split_windows["X"])),
        "num_trajectories": int(len(np.unique(split_windows["trajectory_id"]))),
        "num_devices": int(len(np.unique(split_windows["device_id"]))),
        "mode_counts": counts(split_windows["mode_id"]),
        "spectral_counts": counts(split_windows["spectral_id"]),
        "coupling_counts": counts(split_windows["coupling_id"]),
        "transition_fraction": float(np.mean(split_windows["is_transition_window"].astype(np.float32))),
        "mean_load": float(np.mean(split_windows["mean_load"])),
    }



def leakage_report(split_data: Dict[str, Dict[str, np.ndarray]], cfg: SplitConfig) -> Dict[str, object]:
    train = split_data["train"]
    val = split_data["val"]
    test = split_data["test"]

    overlap = {
        "trajectory_overlap": {
            "train_val": _set_overlap(train["trajectory_id"], val["trajectory_id"]),
            "train_test": _set_overlap(train["trajectory_id"], test["trajectory_id"]),
            "val_test": _set_overlap(val["trajectory_id"], test["trajectory_id"]),
        },
        "device_overlap": {
            "train_val": _set_overlap(train["device_id"], val["device_id"]),
            "train_test": _set_overlap(train["device_id"], test["device_id"]),
            "val_test": _set_overlap(val["device_id"], test["device_id"]),
        },
    }

    hard_leakage = False
    if cfg.split_by == "trajectory":
        hard_leakage = any(len(v) > 0 for v in overlap["trajectory_overlap"].values())
    elif cfg.split_by == "device":
        hard_leakage = any(len(v) > 0 for v in overlap["device_overlap"].values())

    split_summaries = {name: summarize_split(data) for name, data in split_data.items()}

    return {
        "split_by": cfg.split_by,
        "seed": cfg.seed,
        "has_hard_leakage": bool(hard_leakage),
        "overlap": overlap,
        "splits": split_summaries,
    }


# -----------------------------------------------------------------------------
# Saving
# -----------------------------------------------------------------------------

def save_split_npz(path: str | Path, split_windows: Dict[str, np.ndarray]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **split_windows)



def save_split_bundle(
    output_dir: str | Path,
    split_data: Dict[str, Dict[str, np.ndarray]],
    report: Dict[str, object],
    dataset_metadata: Dict[str, object],
    cfg: SplitConfig,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    save_split_npz(output_dir / "train_windows.npz", split_data["train"])
    save_split_npz(output_dir / "val_windows.npz", split_data["val"])
    save_split_npz(output_dir / "test_windows.npz", split_data["test"])

    manifest = {
        "split_config": {
            "train_ratio": cfg.train_ratio,
            "val_ratio": cfg.val_ratio,
            "test_ratio": cfg.test_ratio,
            "split_by": cfg.split_by,
            "seed": cfg.seed,
        },
        "source_metadata": dataset_metadata,
    }

    with open(output_dir / "split_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    with open(output_dir / "split_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)


# -----------------------------------------------------------------------------
# Printing
# -----------------------------------------------------------------------------

def format_report(report: Dict[str, object]) -> str:
    lines = []
    lines.append(f"split_by: {report['split_by']}")
    lines.append(f"seed: {report['seed']}")
    lines.append(f"has_hard_leakage: {report['has_hard_leakage']}")
    lines.append("")

    lines.append("[overlap]")
    for group_name, group in report["overlap"].items():
        lines.append(f"- {group_name}:")
        for pair_name, ids in group.items():
            lines.append(f"  {pair_name}: {ids}")

    lines.append("")
    lines.append("[splits]")
    for split_name, summary in report["splits"].items():
        lines.append(f"- {split_name}:")
        for k, v in summary.items():
            lines.append(f"  {k}: {v}")

    return "\n".join(lines)
