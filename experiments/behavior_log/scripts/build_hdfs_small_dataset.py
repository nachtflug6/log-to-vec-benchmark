#!/usr/bin/env python3
"""Build a smaller HDFS dataset by sampling blocks with a target label ratio."""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]

DEFAULT_INPUT = Path("experiments/behavior_log/artifacts/datasets/HDFS/structured/HDFS_structured.csv")
DEFAULT_OUTPUT_DIR = Path("experiments/behavior_log/artifacts/datasets/HDFS/small")
DEFAULT_PREFIX = "HDFS-small"


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def collect_blocks_by_label(path: Path) -> dict[str, list[str]]:
    blocks_by_label: dict[str, set[str]] = {"Normal": set(), "Anomaly": set()}

    with path.open("r", newline="", encoding="utf-8", errors="replace") as handle:
        reader = csv.DictReader(handle)
        required_columns = {"BlockId", "Label"}
        missing = required_columns.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Input CSV is missing required columns: {', '.join(sorted(missing))}")

        for row in reader:
            block_id = row["BlockId"].strip()
            label = row["Label"].strip()
            if not block_id or label not in blocks_by_label:
                continue
            blocks_by_label[label].add(block_id)

    return {label: sorted(block_ids) for label, block_ids in blocks_by_label.items()}


def sample_blocks(
    blocks_by_label: dict[str, list[str]],
    *,
    normal_blocks: int,
    anomaly_blocks: int,
    seed: int,
) -> dict[str, list[str]]:
    rng = random.Random(seed)

    available_normal = blocks_by_label.get("Normal", [])
    available_anomaly = blocks_by_label.get("Anomaly", [])
    if len(available_normal) < normal_blocks:
        raise ValueError(
            f"Requested {normal_blocks} normal blocks, but only {len(available_normal)} are available."
        )
    if len(available_anomaly) < anomaly_blocks:
        raise ValueError(
            f"Requested {anomaly_blocks} anomaly blocks, but only {len(available_anomaly)} are available."
        )

    return {
        "Normal": sorted(rng.sample(available_normal, normal_blocks)),
        "Anomaly": sorted(rng.sample(available_anomaly, anomaly_blocks)),
    }


def write_selected_blocks(path: Path, sampled_blocks: dict[str, list[str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["BlockId", "Label"])
        for label in ("Normal", "Anomaly"):
            for block_id in sampled_blocks[label]:
                writer.writerow([block_id, label])


def write_small_dataset(
    path: Path,
    input_path: Path,
    selected_blocks: set[str],
) -> tuple[int, Counter[str]]:
    rows_written = 0
    label_counts: Counter[str] = Counter()

    with input_path.open("r", newline="", encoding="utf-8", errors="replace") as input_handle:
        reader = csv.DictReader(input_handle)
        if reader.fieldnames is None:
            raise ValueError("Input CSV is empty.")

        with path.open("w", newline="", encoding="utf-8") as output_handle:
            writer = csv.DictWriter(output_handle, fieldnames=reader.fieldnames)
            writer.writeheader()

            for row in reader:
                block_id = row["BlockId"].strip()
                if block_id not in selected_blocks:
                    continue
                writer.writerow(row)
                rows_written += 1
                label_counts[row["Label"].strip()] += 1

    return rows_written, label_counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Path to the merged HDFS raw CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for the sampled dataset and metadata.",
    )
    parser.add_argument(
        "--prefix",
        default=DEFAULT_PREFIX,
        help="Filename prefix for generated outputs.",
    )
    parser.add_argument(
        "--normal-blocks",
        type=int,
        default=4000,
        help="Number of normal blocks to sample.",
    )
    parser.add_argument(
        "--anomaly-blocks",
        type=int,
        default=1000,
        help="Number of anomaly blocks to sample.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for block sampling.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = resolve_path(args.input)
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    blocks_by_label = collect_blocks_by_label(input_path)
    sampled_blocks = sample_blocks(
        blocks_by_label,
        normal_blocks=args.normal_blocks,
        anomaly_blocks=args.anomaly_blocks,
        seed=args.seed,
    )
    selected_blocks = set(sampled_blocks["Normal"]) | set(sampled_blocks["Anomaly"])

    dataset_path = output_dir / f"{args.prefix}.csv"
    selected_blocks_path = output_dir / f"{args.prefix}_selected_blocks.csv"
    metadata_path = output_dir / f"{args.prefix}_metadata.json"

    rows_written, row_label_counts = write_small_dataset(dataset_path, input_path, selected_blocks)
    write_selected_blocks(selected_blocks_path, sampled_blocks)

    metadata = {
        "input_path": str(input_path),
        "output_path": str(dataset_path),
        "selected_blocks_path": str(selected_blocks_path),
        "seed": args.seed,
        "requested_block_counts": {
            "Normal": args.normal_blocks,
            "Anomaly": args.anomaly_blocks,
        },
        "sampled_block_counts": {
            "Normal": len(sampled_blocks["Normal"]),
            "Anomaly": len(sampled_blocks["Anomaly"]),
            "Total": len(selected_blocks),
        },
        "row_counts": {
            "Normal": row_label_counts.get("Normal", 0),
            "Anomaly": row_label_counts.get("Anomaly", 0),
            "Total": rows_written,
        },
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Input CSV: {input_path}")
    print(f"Output CSV: {dataset_path}")
    print(f"Selected blocks CSV: {selected_blocks_path}")
    print(f"Metadata JSON: {metadata_path}")
    print(
        "Sampled blocks: "
        f"{len(sampled_blocks['Normal'])} normal + {len(sampled_blocks['Anomaly'])} anomaly = {len(selected_blocks)} total"
    )
    print(
        "Output rows: "
        f"{row_label_counts.get('Normal', 0)} normal + {row_label_counts.get('Anomaly', 0)} anomaly = {rows_written} total"
    )


if __name__ == "__main__":
    main()
