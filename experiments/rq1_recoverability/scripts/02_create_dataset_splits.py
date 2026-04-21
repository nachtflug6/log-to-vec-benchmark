"""RQ1 step 2: build leakage-aware train/val/test splits."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from version2.data.fsss_data import (
    SplitConfig,
    format_report,
    leakage_report,
    load_fsss_dataset,
    save_split_bundle,
    split_fsss_windows,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create train/val/test splits for an RQ1 dataset artifact.")
    parser.add_argument("--dataset_dir", type=str, required=True, help="Directory containing trajectories.csv, windows.npz, metadata.json")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory where the split bundle will be stored")
    parser.add_argument("--split_by", type=str, default="trajectory", choices=["trajectory", "device"])
    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--test_ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = SplitConfig(
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        split_by=args.split_by,
        seed=args.seed,
    )

    dataset = load_fsss_dataset(args.dataset_dir)
    split_data = split_fsss_windows(dataset, cfg)
    report = leakage_report(split_data, cfg)
    save_split_bundle(
        output_dir=Path(args.output_dir),
        split_data=split_data,
        report=report,
        dataset_metadata=dataset["metadata"],
        cfg=cfg,
    )

    print(format_report(report))
    print(f"\nSaved split bundle to: {args.output_dir}")


if __name__ == "__main__":
    main()
