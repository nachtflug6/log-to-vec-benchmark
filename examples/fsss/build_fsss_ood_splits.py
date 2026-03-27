from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from version2.data.fsss_data import load_fsss_dataset, leakage_report, save_split_bundle, SplitConfig


def _subset_windows(windows: Dict[str, np.ndarray], mask: np.ndarray) -> Dict[str, np.ndarray]:
    return {k: v[mask] for k, v in windows.items()}


def _save(output_dir: Path, split_data: Dict[str, Dict[str, np.ndarray]], report: Dict[str, object], dataset_metadata: Dict[str, object], extra_meta: Dict[str, object]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    dummy_cfg = SplitConfig(train_ratio=0.0, val_ratio=0.0, test_ratio=0.0, split_by="trajectory", seed=42)
    save_split_bundle(output_dir, split_data, report, dataset_metadata, dummy_cfg)
    manifest_path = output_dir / "split_manifest.json"
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    manifest["ood_spec"] = extra_meta
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


def build_factor_combination_ood(dataset: Dict[str, object], heldout_pairs: list[tuple[int, int]]) -> Dict[str, Dict[str, np.ndarray]]:
    windows = dataset["windows"]
    pairs = list(zip(windows["spectral_id"].tolist(), windows["coupling_id"].tolist()))
    is_heldout = np.array([tuple(p) in set(heldout_pairs) for p in pairs], dtype=bool)

    trainval_mask = ~is_heldout
    test_mask = is_heldout

    traj_ids = np.unique(windows["trajectory_id"][trainval_mask])
    rng = np.random.default_rng(42)
    rng.shuffle(traj_ids)
    n_val = max(1, int(0.15 * len(traj_ids)))
    val_traj = set(map(int, traj_ids[:n_val]))

    val_mask = np.isin(windows["trajectory_id"], list(val_traj)) & trainval_mask
    train_mask = trainval_mask & ~val_mask

    return {
        "train": _subset_windows(windows, train_mask),
        "val": _subset_windows(windows, val_mask),
        "test": _subset_windows(windows, test_mask),
    }


def build_load_range_ood(dataset: Dict[str, object], low_max: float, high_min: float) -> Dict[str, Dict[str, np.ndarray]]:
    windows = dataset["windows"]
    load = windows["mean_load"]
    test_mask = (load <= low_max) | (load >= high_min)
    trainval_mask = ~test_mask

    traj_ids = np.unique(windows["trajectory_id"][trainval_mask])
    rng = np.random.default_rng(42)
    rng.shuffle(traj_ids)
    n_val = max(1, int(0.15 * len(traj_ids)))
    val_traj = set(map(int, traj_ids[:n_val]))

    val_mask = np.isin(windows["trajectory_id"], list(val_traj)) & trainval_mask
    train_mask = trainval_mask & ~val_mask

    return {
        "train": _subset_windows(windows, train_mask),
        "val": _subset_windows(windows, val_mask),
        "test": _subset_windows(windows, test_mask),
    }


def build_device_ood(dataset: Dict[str, object], heldout_devices: list[int]) -> Dict[str, Dict[str, np.ndarray]]:
    windows = dataset["windows"]
    device_id = windows["device_id"]
    test_mask = np.isin(device_id, heldout_devices)
    trainval_mask = ~test_mask

    remaining_devices = np.unique(device_id[trainval_mask])
    if len(remaining_devices) < 2:
        raise ValueError("Need at least 2 non-heldout devices to create train/val splits.")
    rng = np.random.default_rng(42)
    rng.shuffle(remaining_devices)
    val_devices = set(map(int, remaining_devices[:max(1, int(0.25 * len(remaining_devices)))]))

    val_mask = np.isin(device_id, list(val_devices)) & trainval_mask
    train_mask = trainval_mask & ~val_mask

    return {
        "train": _subset_windows(windows, train_mask),
        "val": _subset_windows(windows, val_mask),
        "test": _subset_windows(windows, test_mask),
    }


def parse_pair_list(text: str) -> list[tuple[int, int]]:
    # format: "0-2,3-1"
    pairs = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        a, b = item.split("-")
        pairs.append((int(a), int(b)))
    return pairs


def parse_int_list(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build explicit OOD splits for FSSS windows.")
    parser.add_argument("--raw_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--ood_type", type=str, required=True, choices=["factor_combination", "load_range", "device"])
    parser.add_argument("--heldout_pairs", type=str, default="")
    parser.add_argument("--low_max", type=float, default=0.86)
    parser.add_argument("--high_min", type=float, default=1.14)
    parser.add_argument("--heldout_devices", type=str, default="")
    args = parser.parse_args()

    dataset = load_fsss_dataset(args.raw_dir)

    if args.ood_type == "factor_combination":
        heldout_pairs = parse_pair_list(args.heldout_pairs)
        if len(heldout_pairs) == 0:
            raise ValueError("For factor_combination OOD, provide --heldout_pairs like '0-2,3-1'.")
        split_data = build_factor_combination_ood(dataset, heldout_pairs)
        extra_meta = {"ood_type": "factor_combination", "heldout_pairs": heldout_pairs}

    elif args.ood_type == "load_range":
        split_data = build_load_range_ood(dataset, args.low_max, args.high_min)
        extra_meta = {"ood_type": "load_range", "low_max": args.low_max, "high_min": args.high_min}

    else:
        heldout_devices = parse_int_list(args.heldout_devices)
        if len(heldout_devices) == 0:
            raise ValueError("For device OOD, provide --heldout_devices like '4,5'.")
        split_data = build_device_ood(dataset, heldout_devices)
        extra_meta = {"ood_type": "device", "heldout_devices": heldout_devices}

    report = {
        "split_by": f"ood::{args.ood_type}",
        "seed": 42,
        "has_hard_leakage": False,
        "overlap": leakage_report(split_data, SplitConfig(split_by="trajectory"))["overlap"],
        "splits": leakage_report(split_data, SplitConfig(split_by="trajectory"))["splits"],
    }

    _save(Path(args.output_dir), split_data, report, dataset["metadata"], extra_meta)
    print(json.dumps(report, indent=2))
    print(f"Saved OOD splits to: {args.output_dir}")


if __name__ == "__main__":
    main()
