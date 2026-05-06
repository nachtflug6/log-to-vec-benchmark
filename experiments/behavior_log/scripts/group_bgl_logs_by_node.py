#!/usr/bin/env python3
"""Group BGL structured logs by Node and sort each node by Timestamp, LineId.

This script preserves the original row schema. It only changes row order so
that all rows from the same Node are contiguous, and rows inside each Node are
ordered by Timestamp then LineId.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import tempfile
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_INPUT = Path("experiments/behavior_log/artifacts/datasets/BGL/structured/BGL_structured.csv")
DEFAULT_OUTPUT = Path("experiments/behavior_log/artifacts/datasets/BGL/node_grouped/BGL_node_grouped.csv")
DEFAULT_SUMMARY = Path("experiments/behavior_log/artifacts/datasets/BGL/node_grouped/BGL_node_grouped_timegap_summary.json")
DEFAULT_DISTRIBUTION = Path("experiments/behavior_log/artifacts/datasets/BGL/node_grouped/BGL_node_grouped_timegap_distribution.csv")
TIMEGAP_COLUMN = "TimeGap"
TIMEGAP_BINS = (
    (-math.inf, 0.0, "<0s"),
    (0.0, 0.0, "0s"),
    (0.0, 1.0, "(0,1]s"),
    (1.0, 5.0, "(1,5]s"),
    (5.0, 10.0, "(5,10]s"),
    (10.0, 30.0, "(10,30]s"),
    (30.0, 60.0, "(30,60]s"),
    (60.0, 300.0, "(1,5]min"),
    (300.0, 1800.0, "(5,30]min"),
    (1800.0, 3600.0, "(30,60]min"),
    (3600.0, 21600.0, "(1,6]h"),
    (21600.0, 86400.0, "(6,24]h"),
    (86400.0, math.inf, ">24h"),
)


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def stable_shard(node: str, shard_count: int) -> int:
    digest = hashlib.md5(node.encode("utf-8")).hexdigest()
    return int(digest, 16) % shard_count


def sort_key(row: dict[str, str]) -> tuple[float, int]:
    try:
        timestamp = float(row["Timestamp"])
    except ValueError:
        timestamp = 0.0
    try:
        line_id = int(float(row["LineId"]))
    except ValueError:
        line_id = 0
    return timestamp, line_id


def parse_timestamp(row: dict[str, str]) -> float:
    time_value = row.get("Time", "").strip()
    if time_value:
        try:
            return datetime.strptime(time_value, "%Y-%m-%d-%H.%M.%S.%f").timestamp()
        except ValueError:
            pass
    try:
        return float(row["Timestamp"])
    except ValueError:
        return 0.0


def format_timegap(value: float) -> str:
    return f"{value:.6f}".rstrip("0").rstrip(".") if value != 0 else "0"


def timegap_bin(value: float) -> str:
    for left, right, label in TIMEGAP_BINS:
        if left == -math.inf and value < right:
            return label
        if left == -math.inf:
            continue
        if left == right == 0.0 and value == 0.0:
            return label
        if value > left and value <= right:
            return label
    return ">24h"


def percentile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = (len(sorted_values) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    weight = position - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


def write_timegap_distribution(
    path: Path,
    *,
    bin_counts: Counter[str],
    total_rows: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["bin", "count", "share_of_rows"])
        for _, _, label in TIMEGAP_BINS:
            count = bin_counts.get(label, 0)
            writer.writerow([label, count, count / total_rows if total_rows else 0.0])


def write_shards(
    *,
    input_path: Path,
    temp_dir: Path,
    shard_count: int,
    progress_every: int,
    max_rows: int | None,
) -> tuple[list[str], int, int]:
    shard_paths = [temp_dir / f"shard_{idx:04d}.csv" for idx in range(shard_count)]
    handles = [path.open("w", newline="", encoding="utf-8") for path in shard_paths]
    writers: list[csv.DictWriter[str]] = []
    total_rows = 0
    missing_node_rows = 0

    try:
        with input_path.open("r", newline="", encoding="utf-8", errors="replace") as input_handle:
            reader = csv.DictReader(input_handle)
            if reader.fieldnames is None:
                raise ValueError("Input CSV is empty.")
            required = {"LineId", "Node", "Timestamp"}
            missing = required.difference(reader.fieldnames)
            if missing:
                raise ValueError(f"Input CSV is missing required columns: {', '.join(sorted(missing))}")

            for handle in handles:
                writer: csv.DictWriter[str] = csv.DictWriter(handle, fieldnames=reader.fieldnames)
                writer.writeheader()
                writers.append(writer)

            for row in reader:
                if max_rows is not None and total_rows >= max_rows:
                    break
                total_rows += 1
                if progress_every and total_rows % progress_every == 0:
                    print(f"Sharded {total_rows:,} rows")

                node = row["Node"].strip()
                if not node:
                    missing_node_rows += 1
                    continue
                writers[stable_shard(node, shard_count)].writerow(row)
    finally:
        for handle in handles:
            handle.close()

    return list(reader.fieldnames or []), total_rows, missing_node_rows


def merge_sorted_shards(
    *,
    temp_dir: Path,
    output_path: Path,
    fieldnames: list[str],
    shard_count: int,
    progress_every: int,
) -> tuple[int, int, dict[str, object]]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows_written = 0
    nodes_written = 0
    timegap_values: list[float] = []
    timegap_bin_counts: Counter[str] = Counter()
    first_rows_per_node = 0
    output_fieldnames = [*fieldnames]
    if TIMEGAP_COLUMN not in output_fieldnames:
        output_fieldnames.append(TIMEGAP_COLUMN)

    with output_path.open("w", newline="", encoding="utf-8") as output_handle:
        writer = csv.DictWriter(output_handle, fieldnames=output_fieldnames)
        writer.writeheader()

        for shard_idx in range(shard_count):
            shard_path = temp_dir / f"shard_{shard_idx:04d}.csv"
            rows_by_node: defaultdict[str, list[dict[str, str]]] = defaultdict(list)
            with shard_path.open("r", newline="", encoding="utf-8", errors="replace") as shard_handle:
                reader = csv.DictReader(shard_handle)
                for row in reader:
                    rows_by_node[row["Node"].strip()].append(row)

            for node in sorted(rows_by_node):
                nodes_written += 1
                previous_timestamp: float | None = None
                for row in sorted(rows_by_node[node], key=sort_key):
                    timestamp = parse_timestamp(row)
                    if previous_timestamp is None:
                        timegap = 0.0
                        first_rows_per_node += 1
                    else:
                        timegap = timestamp - previous_timestamp
                    previous_timestamp = timestamp
                    row[TIMEGAP_COLUMN] = format_timegap(timegap)
                    timegap_values.append(timegap)
                    timegap_bin_counts[timegap_bin(timegap)] += 1
                    writer.writerow(row)
                    rows_written += 1
                    if progress_every and rows_written % progress_every == 0:
                        print(f"Wrote {rows_written:,} grouped rows")

    sorted_timegaps = sorted(timegap_values)
    nonzero_timegaps = [value for value in sorted_timegaps if value != 0]
    positive_timegaps = [value for value in sorted_timegaps if value > 0]
    negative_timegaps = [value for value in sorted_timegaps if value < 0]
    timegap_summary = {
        "column": TIMEGAP_COLUMN,
        "unit": "seconds",
        "count": len(sorted_timegaps),
        "zero_count": timegap_bin_counts.get("0s", 0),
        "negative_count": len(negative_timegaps),
        "positive_count": len(positive_timegaps),
        "nonzero_count": len(nonzero_timegaps),
        "first_rows_per_node": first_rows_per_node,
        "mean": sum(sorted_timegaps) / len(sorted_timegaps) if sorted_timegaps else 0.0,
        "mean_nonzero": sum(nonzero_timegaps) / len(nonzero_timegaps) if nonzero_timegaps else 0.0,
        "mean_positive": sum(positive_timegaps) / len(positive_timegaps) if positive_timegaps else 0.0,
        "mean_negative": sum(negative_timegaps) / len(negative_timegaps) if negative_timegaps else 0.0,
        "min": sorted_timegaps[0] if sorted_timegaps else 0.0,
        "max": sorted_timegaps[-1] if sorted_timegaps else 0.0,
        "p50": percentile(sorted_timegaps, 0.50),
        "p90": percentile(sorted_timegaps, 0.90),
        "p95": percentile(sorted_timegaps, 0.95),
        "p99": percentile(sorted_timegaps, 0.99),
        "bin_counts": dict(timegap_bin_counts),
    }
    return rows_written, nodes_written, timegap_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Path to BGL_structured.csv.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output grouped CSV.")
    parser.add_argument("--timegap-summary", type=Path, default=DEFAULT_SUMMARY, help="Output TimeGap summary JSON.")
    parser.add_argument(
        "--timegap-distribution",
        type=Path,
        default=DEFAULT_DISTRIBUTION,
        help="Output TimeGap binned distribution CSV.",
    )
    parser.add_argument("--shards", type=int, default=128, help="Number of temporary node hash shards.")
    parser.add_argument("--temp-dir", type=Path, help="Optional temporary directory.")
    parser.add_argument("--max-rows", type=int, help="Optional row limit for smoke tests.")
    parser.add_argument("--progress-every", type=int, default=1_000_000, help="Print progress every N rows.")
    parser.add_argument("--keep-temp", action="store_true", help="Keep temporary shard files.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.shards <= 0:
        raise ValueError("--shards must be positive.")

    input_path = resolve_path(args.input)
    output_path = resolve_path(args.output)
    timegap_summary_path = resolve_path(args.timegap_summary)
    timegap_distribution_path = resolve_path(args.timegap_distribution)
    temp_root = resolve_path(args.temp_dir) if args.temp_dir else Path(
        tempfile.mkdtemp(prefix="bgl_node_group_", dir="/tmp")
    )
    temp_root.mkdir(parents=True, exist_ok=True)

    try:
        fieldnames, total_rows, missing_node_rows = write_shards(
            input_path=input_path,
            temp_dir=temp_root,
            shard_count=args.shards,
            progress_every=args.progress_every,
            max_rows=args.max_rows,
        )
        rows_written, nodes_written, timegap_summary = merge_sorted_shards(
            temp_dir=temp_root,
            output_path=output_path,
            fieldnames=fieldnames,
            shard_count=args.shards,
            progress_every=args.progress_every,
        )
    finally:
        if not args.keep_temp:
            shutil.rmtree(temp_root, ignore_errors=True)

    timegap_summary_payload = {
        "input_path": str(input_path),
        "output_path": str(output_path),
        "rows_read": total_rows,
        "rows_written": rows_written,
        "nodes_written": nodes_written,
        "missing_node_rows_skipped": missing_node_rows,
        "timegap": timegap_summary,
    }
    timegap_summary_path.parent.mkdir(parents=True, exist_ok=True)
    timegap_summary_path.write_text(
        json.dumps(timegap_summary_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_timegap_distribution(
        timegap_distribution_path,
        bin_counts=Counter(timegap_summary["bin_counts"]),
        total_rows=rows_written,
    )

    print(f"Input CSV: {input_path}")
    print(f"Output CSV: {output_path}")
    print(f"TimeGap summary JSON: {timegap_summary_path}")
    print(f"TimeGap distribution CSV: {timegap_distribution_path}")
    print(f"Rows read: {total_rows:,}")
    print(f"Rows written: {rows_written:,}")
    print(f"Nodes written: {nodes_written:,}")
    print(f"Missing-node rows skipped: {missing_node_rows:,}")
    print(
        "TimeGap seconds: "
        f"mean={timegap_summary['mean']:.4f}, "
        f"p50={timegap_summary['p50']:.4f}, "
        f"p95={timegap_summary['p95']:.4f}, "
        f"max={timegap_summary['max']:.4f}"
    )


if __name__ == "__main__":
    main()
