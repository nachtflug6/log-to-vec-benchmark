#!/usr/bin/env python3
"""Sample BGL nodes and keep full logs for those nodes.

The input should already be grouped by Node. Sampling is node-level:

- anomaly-containing nodes: nodes with at least one Label == Anomaly
- normal-only nodes: nodes with only Label == Normal

By default, this builds a 2500-node small dataset with a 4:1
normal:anomaly node ratio: 2000 normal-only nodes + 500 anomaly-containing
nodes. All rows from selected nodes are retained.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_INPUT = Path("experiments/behavior_log/artifacts/datasets/BGL/node_grouped/BGL_node_grouped.csv")
DEFAULT_OUTPUT_DIR = Path("experiments/behavior_log/artifacts/datasets/BGL/small_node")
DEFAULT_PREFIX = "BGL-small-node"


@dataclass
class NodeStats:
    node: str
    rows: int = 0
    label_counts: Counter[str] | None = None
    label_type_counts: Counter[str] | None = None

    def __post_init__(self) -> None:
        if self.label_counts is None:
            self.label_counts = Counter()
        if self.label_type_counts is None:
            self.label_type_counts = Counter()

    @property
    def binary_label(self) -> str:
        return "Anomaly" if self.label_counts and self.label_counts.get("Anomaly", 0) > 0 else "Normal"


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def scan_node_stats(path: Path) -> dict[str, NodeStats]:
    stats: dict[str, NodeStats] = {}
    with path.open("r", newline="", encoding="utf-8", errors="replace") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("Input CSV is empty.")
        required = {"Node", "Label"}
        missing = required.difference(reader.fieldnames)
        if missing:
            raise ValueError(f"Input CSV is missing required columns: {', '.join(sorted(missing))}")
        has_label_type = "LabelType" in reader.fieldnames

        for row in reader:
            node = row["Node"].strip()
            if not node:
                continue
            if node not in stats:
                stats[node] = NodeStats(node=node)
            node_stats = stats[node]
            node_stats.rows += 1
            node_stats.label_counts[row["Label"].strip()] += 1
            if has_label_type:
                node_stats.label_type_counts[row["LabelType"].strip()] += 1
    return stats


def sample_nodes(
    node_stats: dict[str, NodeStats],
    *,
    normal_nodes: int,
    anomaly_nodes: int,
    seed: int,
) -> tuple[list[str], list[str]]:
    normal_candidates = sorted(node for node, stats in node_stats.items() if stats.binary_label == "Normal")
    anomaly_candidates = sorted(node for node, stats in node_stats.items() if stats.binary_label == "Anomaly")
    if len(normal_candidates) < normal_nodes:
        raise ValueError(f"Requested {normal_nodes} normal nodes, but only {len(normal_candidates)} are available.")
    if len(anomaly_candidates) < anomaly_nodes:
        raise ValueError(f"Requested {anomaly_nodes} anomaly nodes, but only {len(anomaly_candidates)} are available.")

    rng = random.Random(seed)
    return (
        sorted(rng.sample(normal_candidates, normal_nodes)),
        sorted(rng.sample(anomaly_candidates, anomaly_nodes)),
    )


def write_selected_nodes(path: Path, selected_nodes: list[str], node_stats: dict[str, NodeStats]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Node", "Label", "Rows", "LabelCounts", "LabelTypeCounts"])
        for node in selected_nodes:
            stats = node_stats[node]
            writer.writerow(
                [
                    node,
                    stats.binary_label,
                    stats.rows,
                    json.dumps(dict(sorted((stats.label_counts or Counter()).items()))),
                    json.dumps(dict(sorted((stats.label_type_counts or Counter()).items()))),
                ]
            )


def write_small_dataset(
    *,
    input_path: Path,
    output_path: Path,
    selected_nodes: set[str],
) -> tuple[int, Counter[str]]:
    rows_written = 0
    label_counts: Counter[str] = Counter()
    with input_path.open("r", newline="", encoding="utf-8", errors="replace") as input_handle:
        reader = csv.DictReader(input_handle)
        if reader.fieldnames is None:
            raise ValueError("Input CSV is empty.")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", newline="", encoding="utf-8") as output_handle:
            writer = csv.DictWriter(output_handle, fieldnames=reader.fieldnames)
            writer.writeheader()
            for row in reader:
                node = row["Node"].strip()
                if node not in selected_nodes:
                    continue
                writer.writerow(row)
                rows_written += 1
                label_counts[row["Label"].strip()] += 1
    return rows_written, label_counts


def iter_window_labels(
    labels: list[str],
    *,
    window_size: int,
    stride: int,
    min_window_size: int,
    drop_incomplete: bool,
) -> list[list[str]]:
    windows: list[list[str]] = []
    if len(labels) < min_window_size:
        return windows

    for start in range(0, len(labels), stride):
        end = start + window_size
        if end > len(labels) and drop_incomplete:
            break
        window = labels[start:min(end, len(labels))]
        if len(window) < min_window_size:
            break
        windows.append(window)
        if end >= len(labels):
            break
    return windows


def compute_window_sampling_stats(
    *,
    input_path: Path,
    selected_nodes: set[str],
    window_size: int,
    stride: int,
    min_window_size: int,
    drop_incomplete: bool,
) -> dict[str, float | int]:
    labels_by_node: dict[str, list[str]] = {}

    with input_path.open("r", newline="", encoding="utf-8", errors="replace") as input_handle:
        reader = csv.DictReader(input_handle)
        if reader.fieldnames is None:
            raise ValueError("Input CSV is empty.")
        required = {"Node", "Label"}
        missing = required.difference(reader.fieldnames)
        if missing:
            raise ValueError(f"Input CSV is missing required columns: {', '.join(sorted(missing))}")

        for row in reader:
            node = row["Node"].strip()
            if node not in selected_nodes:
                continue
            labels_by_node.setdefault(node, []).append(row["Label"].strip())

    normal_windows = 0
    anomaly_windows = 0
    anomaly_ratios: list[float] = []
    windows_per_node: list[int] = []

    for node in sorted(selected_nodes):
        node_windows = iter_window_labels(
            labels_by_node.get(node, []),
            window_size=window_size,
            stride=stride,
            min_window_size=min_window_size,
            drop_incomplete=drop_incomplete,
        )
        windows_per_node.append(len(node_windows))
        for labels in node_windows:
            anomaly_count = sum(1 for label in labels if label == "Anomaly")
            if anomaly_count > 0:
                anomaly_windows += 1
                anomaly_ratios.append(anomaly_count / len(labels))
            else:
                normal_windows += 1

    n_windows = normal_windows + anomaly_windows
    return {
        "window_size": window_size,
        "stride": stride,
        "min_window_size": min_window_size,
        "drop_incomplete": bool(drop_incomplete),
        "n_windows": n_windows,
        "normal_windows": normal_windows,
        "anomaly_windows": anomaly_windows,
        "anomaly_window_ratio": anomaly_windows / n_windows if n_windows else 0.0,
        "average_windows_per_node": sum(windows_per_node) / len(windows_per_node) if windows_per_node else 0.0,
        "average_anomaly_ratio_per_anomaly_window": (
            sum(anomaly_ratios) / len(anomaly_ratios) if anomaly_ratios else 0.0
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Grouped BGL node CSV.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Output directory.")
    parser.add_argument("--prefix", default=DEFAULT_PREFIX, help="Filename prefix.")
    parser.add_argument("--total-nodes", type=int, default=2500, help="Total nodes to sample.")
    parser.add_argument("--normal-ratio", type=int, default=4, help="Normal side of normal:anomaly node ratio.")
    parser.add_argument("--anomaly-ratio", type=int, default=1, help="Anomaly side of normal:anomaly node ratio.")
    parser.add_argument("--window-size", type=int, default=50, help="Window size used for post-sampling statistics.")
    parser.add_argument("--stride", type=int, default=25, help="Window stride used for post-sampling statistics.")
    parser.add_argument("--min-window-size", type=int, default=10, help="Minimum window size used for statistics.")
    parser.add_argument(
        "--drop-incomplete",
        action="store_true",
        help="Use the same final-partial-window policy as Stage 01 when computing statistics.",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = resolve_path(args.input)
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.total_nodes <= 0:
        raise ValueError("--total-nodes must be positive.")
    if args.normal_ratio < 0 or args.anomaly_ratio < 0 or args.normal_ratio + args.anomaly_ratio <= 0:
        raise ValueError("--normal-ratio and --anomaly-ratio must be non-negative and cannot both be zero.")
    if args.window_size <= 0 or args.stride <= 0 or args.min_window_size <= 0:
        raise ValueError("--window-size, --stride, and --min-window-size must be positive.")
    if args.min_window_size > args.window_size:
        raise ValueError("--min-window-size cannot exceed --window-size.")

    total_ratio = args.normal_ratio + args.anomaly_ratio
    anomaly_nodes = int(args.total_nodes * args.anomaly_ratio / total_ratio)
    normal_nodes = args.total_nodes - anomaly_nodes

    node_stats = scan_node_stats(input_path)
    selected_normal, selected_anomaly = sample_nodes(
        node_stats,
        normal_nodes=normal_nodes,
        anomaly_nodes=anomaly_nodes,
        seed=args.seed,
    )
    selected_nodes = selected_normal + selected_anomaly
    selected_node_set = set(selected_nodes)

    output_csv = output_dir / f"{args.prefix}.csv"
    selected_nodes_csv = output_dir / f"{args.prefix}_selected_nodes.csv"
    metadata_json = output_dir / f"{args.prefix}_metadata.json"

    rows_written, row_label_counts = write_small_dataset(
        input_path=input_path,
        output_path=output_csv,
        selected_nodes=selected_node_set,
    )
    write_selected_nodes(selected_nodes_csv, selected_nodes, node_stats)
    window_stats = compute_window_sampling_stats(
        input_path=input_path,
        selected_nodes=selected_node_set,
        window_size=args.window_size,
        stride=args.stride,
        min_window_size=args.min_window_size,
        drop_incomplete=bool(args.drop_incomplete),
    )

    selected_node_row_counts = Counter(node_stats[node].binary_label for node in selected_nodes)
    metadata = {
        "input_path": str(input_path),
        "output_path": str(output_csv),
        "selected_nodes_path": str(selected_nodes_csv),
        "seed": args.seed,
        "requested_node_counts": {
            "Normal": normal_nodes,
            "Anomaly": anomaly_nodes,
            "Total": args.total_nodes,
        },
        "available_node_counts": {
            "Normal": sum(1 for stats in node_stats.values() if stats.binary_label == "Normal"),
            "Anomaly": sum(1 for stats in node_stats.values() if stats.binary_label == "Anomaly"),
            "Total": len(node_stats),
        },
        "selected_node_counts": {
            "Normal": selected_node_row_counts.get("Normal", 0),
            "Anomaly": selected_node_row_counts.get("Anomaly", 0),
            "Total": len(selected_nodes),
        },
        "row_counts": {
            "Normal": row_label_counts.get("Normal", 0),
            "Anomaly": row_label_counts.get("Anomaly", 0),
            "Total": rows_written,
        },
        "post_sampling_window_stats": window_stats,
    }
    metadata_json.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Input CSV: {input_path}")
    print(f"Output CSV: {output_csv}")
    print(f"Selected nodes CSV: {selected_nodes_csv}")
    print(f"Metadata JSON: {metadata_json}")
    print(
        "Selected nodes: "
        f"{selected_node_row_counts.get('Normal', 0)} normal + "
        f"{selected_node_row_counts.get('Anomaly', 0)} anomaly = {len(selected_nodes)} total"
    )
    print(
        "Selected rows: "
        f"{row_label_counts.get('Normal', 0)} normal + "
        f"{row_label_counts.get('Anomaly', 0)} anomaly = {rows_written} total"
    )
    print("Post-sampling window stats:")
    print(f"  n_nodes: {len(selected_nodes):,}")
    print(f"  n_rows: {rows_written:,}")
    print(f"  n_windows: {window_stats['n_windows']:,}")
    print(f"  normal_windows: {window_stats['normal_windows']:,}")
    print(f"  anomaly_windows: {window_stats['anomaly_windows']:,}")
    print(f"  anomaly_window_ratio: {window_stats['anomaly_window_ratio']:.6f}")
    print(f"  average_windows_per_node: {window_stats['average_windows_per_node']:.2f}")
    print(
        "  average_anomaly_ratio_per_anomaly_window: "
        f"{window_stats['average_anomaly_ratio_per_anomaly_window']:.6f}"
    )


if __name__ == "__main__":
    main()
