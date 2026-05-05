#!/usr/bin/env python3
"""Stream basic distribution statistics from the merged HDFS raw CSV.

This script is designed for the large `HDFS_raw.csv` file. It scans the CSV
once, keeps only a handful of counters in memory, and writes ratio tables for:

- EventId
- BlockId
- Label
- Level
- Pid
- Component

It also writes block-level event trace artifacts:

- Event_traces.csv
- Event_occurrence_matrix.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]

DEFAULT_INPUT = Path(
    "experiments/behavior_log/artifacts/datasets/HDFS/raw/HDFS_raw.csv"
)
DEFAULT_OUTPUT_DIR = Path(
    "experiments/behavior_log/artifacts/datasets/HDFS/analysis/basic_distributions"
)

FIELDS = ("EventId", "BlockId", "Label", "Level", "Pid", "Component")
TOP_K_VALUES = (1, 5, 10, 20, 50, 100)


@dataclass
class BlockTrace:
    label: str = ""
    events: list[str] = field(default_factory=list)
    event_counts: Counter[str] = field(default_factory=Counter)
    intervals: list[int] = field(default_factory=list)
    first_timestamp: int | None = None
    last_timestamp: int | None = None

    def add_event(self, *, label: str, event_id: str, timestamp: int | None) -> None:
        if label and not self.label:
            self.label = label
        self.events.append(event_id)
        self.event_counts[event_id] += 1

        if timestamp is None:
            return
        if self.first_timestamp is None:
            self.first_timestamp = timestamp
        if self.last_timestamp is not None:
            self.intervals.append(max(0, timestamp - self.last_timestamp))
        self.last_timestamp = timestamp

    @property
    def latency(self) -> int:
        if self.first_timestamp is None or self.last_timestamp is None:
            return 0
        return max(0, self.last_timestamp - self.first_timestamp)


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def counter_summary(counter: Counter[str], *, total_rows: int, nonempty_rows: int) -> dict[str, object]:
    counts = [count for _, count in counter.most_common()]
    top_share: dict[str, float] = {}
    running_sum = 0
    for index, count in enumerate(counts, start=1):
        running_sum += count
        if index in TOP_K_VALUES:
            top_share[f"top_{index}"] = running_sum / nonempty_rows if nonempty_rows else 0.0

    for top_k in TOP_K_VALUES:
        key = f"top_{top_k}"
        if key not in top_share:
            top_share[key] = running_sum / nonempty_rows if nonempty_rows else 0.0

    return {
        "unique_values": len(counter),
        "nonempty_rows": nonempty_rows,
        "missing_rows": total_rows - nonempty_rows,
        "coverage_of_rows": nonempty_rows / total_rows if total_rows else 0.0,
        "top_share_of_nonempty": top_share,
    }


def write_distribution_csv(
    path: Path,
    *,
    field_name: str,
    counter: Counter[str],
    total_rows: int,
    nonempty_rows: int,
    top_n: int | None,
) -> None:
    items = counter.most_common(top_n)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "rank",
                field_name,
                "count",
                "share_of_rows",
                "share_of_nonempty",
            ]
        )
        for rank, (value, count) in enumerate(items, start=1):
            writer.writerow(
                [
                    rank,
                    value,
                    count,
                    count / total_rows if total_rows else 0.0,
                    count / nonempty_rows if nonempty_rows else 0.0,
                ]
            )


def event_id_sort_key(event_id: str) -> tuple[int, str]:
    if event_id.startswith("E") and event_id[1:].isdigit():
        return int(event_id[1:]), event_id
    return 1_000_000_000, event_id


def parse_hdfs_timestamp(date_value: str, time_value: str) -> int | None:
    try:
        return int(datetime.strptime(date_value + time_value, "%y%m%d%H%M%S").timestamp())
    except ValueError:
        return None


def write_event_traces_csv(path: Path, block_traces: dict[str, BlockTrace]) -> dict[str, object]:
    label_counts: Counter[str] = Counter()
    trace_lengths: list[int] = []
    latencies: list[int] = []

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["BlockId", "Label", "Features", "TimeInterval", "Latency"])
        for block_id in sorted(block_traces):
            trace = block_traces[block_id]
            label_counts[trace.label] += 1
            trace_lengths.append(len(trace.events))
            latencies.append(trace.latency)
            event_sequence = " ".join(trace.events)
            writer.writerow(
                [
                    block_id,
                    trace.label,
                    event_sequence,
                    json.dumps(trace.intervals),
                    trace.latency,
                ]
            )

    return {
        "path": str(path),
        "block_count": len(block_traces),
        "label_counts": dict(label_counts),
        "average_trace_length": sum(trace_lengths) / len(trace_lengths) if trace_lengths else 0.0,
        "max_trace_length": max(trace_lengths) if trace_lengths else 0,
        "average_latency": sum(latencies) / len(latencies) if latencies else 0.0,
        "max_latency": max(latencies) if latencies else 0,
    }


def write_occurrence_matrix_csv(
    path: Path,
    block_traces: dict[str, BlockTrace],
    event_ids: list[str],
) -> dict[str, object]:
    nonzero_counts: list[int] = []

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["BlockId", "Label", *event_ids])
        for block_id in sorted(block_traces):
            trace = block_traces[block_id]
            nonzero_counts.append(sum(1 for event_id in event_ids if trace.event_counts[event_id] > 0))
            writer.writerow(
                [
                    block_id,
                    trace.label,
                    *[trace.event_counts[event_id] for event_id in event_ids],
                ]
            )

    return {
        "path": str(path),
        "block_count": len(block_traces),
        "event_feature_count": len(event_ids),
        "average_nonzero_events_per_block": (
            sum(nonzero_counts) / len(nonzero_counts) if nonzero_counts else 0.0
        ),
        "max_nonzero_events_per_block": max(nonzero_counts) if nonzero_counts else 0,
    }


def write_markdown_report(path: Path, analysis: dict[str, object]) -> None:
    lines = [
        "# HDFS Raw Basic Distribution Analysis",
        "",
        f"- Input file: `{analysis['input_path']}`",
        f"- Total rows: {analysis['total_rows']:,}",
        f"- Max rows: {analysis['max_rows'] if analysis['max_rows'] is not None else 'all'}",
        "",
    ]

    for field in FIELDS:
        field_stats = analysis["fields"][field]
        lines.extend(
            [
                f"## {field}",
                f"- Unique values: {field_stats['unique_values']:,}",
                f"- Non-empty rows: {field_stats['nonempty_rows']:,}",
                f"- Missing rows: {field_stats['missing_rows']:,}",
                f"- Coverage of rows: {field_stats['coverage_of_rows']:.2%}",
                "",
                "| Top-K | Cumulative share of non-empty rows |",
                "|---|---:|",
            ]
        )
        for key, value in field_stats["top_share_of_nonempty"].items():
            lines.append(f"| {key} | {value:.2%} |")
        lines.append("")

    trace_stats = analysis.get("event_traces")
    if trace_stats:
        lines.extend(
            [
                "## Event Traces",
                f"- Output file: `{trace_stats['path']}`",
                f"- Blocks: {trace_stats['block_count']:,}",
                "- Features column: event sequence for each block-level trace",
                f"- Average trace length: {trace_stats['average_trace_length']:.2f}",
                f"- Max trace length: {trace_stats['max_trace_length']:,}",
                f"- Average latency: {trace_stats['average_latency']:.2f} seconds",
                f"- Max latency: {trace_stats['max_latency']:,} seconds",
                "",
                "| Label | Blocks |",
                "|---|---:|",
            ]
        )
        for label, count in sorted(trace_stats["label_counts"].items()):
            lines.append(f"| {label or '(missing)'} | {count:,} |")
        lines.append("")

    matrix_stats = analysis.get("event_occurrence_matrix")
    if matrix_stats:
        lines.extend(
            [
                "## Event Occurrence Matrix",
                f"- Output file: `{matrix_stats['path']}`",
                f"- Blocks: {matrix_stats['block_count']:,}",
                "- Columns: BlockId, Label, and one count column per EventId",
                f"- Event feature columns: {matrix_stats['event_feature_count']:,}",
                f"- Average non-zero event types per block: {matrix_stats['average_nonzero_events_per_block']:.2f}",
                f"- Max non-zero event types per block: {matrix_stats['max_nonzero_events_per_block']:,}",
                "",
            ]
        )

    path.write_text("\n".join(lines), encoding="utf-8")


def analyze_hdfs_raw(
    path: Path,
    *,
    max_rows: int | None = None,
    progress_every: int = 1_000_000,
) -> dict[str, object]:
    counters = {field: Counter() for field in FIELDS}
    nonempty_rows = {field: 0 for field in FIELDS}
    block_traces: defaultdict[str, BlockTrace] = defaultdict(BlockTrace)
    total_rows = 0

    with path.open("r", newline="", encoding="utf-8", errors="replace") as handle:
        reader = csv.reader(handle)
        header = next(reader, None)
        if header is None:
            raise ValueError("Input CSV is empty.")

        index_map = {name: idx for idx, name in enumerate(header)}
        required_fields = (*FIELDS, "Date", "Time")
        missing = [field for field in required_fields if field not in index_map]
        if missing:
            raise ValueError(f"Input CSV is missing required columns: {', '.join(missing)}")

        event_idx = index_map["EventId"]
        block_idx = index_map["BlockId"]
        label_idx = index_map["Label"]
        date_idx = index_map["Date"]
        time_idx = index_map["Time"]

        for row in reader:
            if max_rows is not None and total_rows >= max_rows:
                break

            total_rows += 1
            if progress_every and total_rows % progress_every == 0:
                print(f"Processed {total_rows:,} rows")

            event_id = row[event_idx].strip()
            block_id = row[block_idx].strip()
            label = row[label_idx].strip()

            for field in FIELDS:
                value = row[index_map[field]].strip()
                if value:
                    counters[field][value] += 1
                    nonempty_rows[field] += 1

            if block_id and event_id:
                timestamp = parse_hdfs_timestamp(row[date_idx].strip(), row[time_idx].strip())
                block_traces[block_id].add_event(label=label, event_id=event_id, timestamp=timestamp)

    field_summaries = {
        field: counter_summary(counters[field], total_rows=total_rows, nonempty_rows=nonempty_rows[field])
        for field in FIELDS
    }

    return {
        "input_path": str(path),
        "total_rows": total_rows,
        "max_rows": max_rows,
        "fields": field_summaries,
        "top_values": {
            field: counters[field].most_common(20)
            for field in FIELDS
        },
        "counters": counters,
        "nonempty_rows": nonempty_rows,
        "block_traces": dict(block_traces),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Path to HDFS_raw.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for summary files.",
    )
    parser.add_argument(
        "--prefix",
        default="hdfs_raw",
        help="Filename prefix for generated outputs.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        help="Optional row limit for smoke tests.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=1_000_000,
        help="Print progress every N rows. Use 0 to disable.",
    )
    parser.add_argument(
        "--full-blockid-csv",
        action="store_true",
        help="Write the full BlockId distribution CSV instead of only the top 1000 values.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = resolve_path(args.input)
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    analysis = analyze_hdfs_raw(
        input_path,
        max_rows=args.max_rows,
        progress_every=args.progress_every,
    )

    serializable = {
        "input_path": analysis["input_path"],
        "total_rows": analysis["total_rows"],
        "max_rows": analysis["max_rows"],
        "fields": analysis["fields"],
        "top_values": analysis["top_values"],
    }

    json_path = output_dir / f"{args.prefix}_distribution_summary.json"
    markdown_path = output_dir / f"{args.prefix}_distribution_summary.md"

    counters: dict[str, Counter[str]] = analysis["counters"]
    nonempty_rows: dict[str, int] = analysis["nonempty_rows"]
    block_traces: dict[str, BlockTrace] = analysis["block_traces"]
    event_ids = sorted(counters["EventId"], key=event_id_sort_key)

    event_traces_path = output_dir / "Event_traces.csv"
    occurrence_matrix_path = output_dir / "Event_occurrence_matrix.csv"
    event_traces_stats = write_event_traces_csv(event_traces_path, block_traces)
    occurrence_matrix_stats = write_occurrence_matrix_csv(
        occurrence_matrix_path,
        block_traces,
        event_ids,
    )
    serializable["event_traces"] = event_traces_stats
    serializable["event_occurrence_matrix"] = occurrence_matrix_stats

    json_path.write_text(json.dumps(serializable, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown_report(markdown_path, serializable)
    print(f"Wrote event traces: {event_traces_path}")
    print(f"Wrote event occurrence matrix: {occurrence_matrix_path}")

    for field in FIELDS:
        top_n = None
        if field == "BlockId" and not args.full_blockid_csv:
            top_n = 1000
        csv_path = output_dir / f"{args.prefix}_{field.lower()}_distribution.csv"
        write_distribution_csv(
            csv_path,
            field_name=field,
            counter=counters[field],
            total_rows=analysis["total_rows"],
            nonempty_rows=nonempty_rows[field],
            top_n=top_n,
        )
        print(f"Wrote {field} distribution: {csv_path}")

    print(f"Wrote JSON summary: {json_path}")
    print(f"Wrote Markdown summary: {markdown_path}")


if __name__ == "__main__":
    main()
