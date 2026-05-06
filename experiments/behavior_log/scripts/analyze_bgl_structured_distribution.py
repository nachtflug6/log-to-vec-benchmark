#!/usr/bin/env python3
"""Stream basic distribution statistics from the structured BGL CSV.

This script is designed for the large `BGL_structured.csv` file. It scans the
CSV once, keeps compact counters in memory, and writes ratio tables for:

- Label
- LabelType
- EventId
- Node
- NodeRepeat
- Type
- Component
- Level
- Date

BGL does not have a natural BlockId/session key like HDFS. If requested,
`--write-node-traces` additionally groups rows by `Node` and writes node-level
event traces plus an occurrence matrix.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]

DEFAULT_INPUT = Path(
    "experiments/behavior_log/artifacts/datasets/BGL/structured/BGL_structured.csv"
)
DEFAULT_OUTPUT_DIR = Path(
    "experiments/behavior_log/artifacts/datasets/BGL/analysis/basic_distributions"
)

FIELDS = ("Label", "LabelType", "EventId", "Node", "NodeRepeat", "Type", "Component", "Level", "Date")
TOP_K_VALUES = (1, 5, 10, 20, 50, 100)


@dataclass
class NodeTrace:
    events: list[str] = field(default_factory=list)
    event_counts: Counter[str] = field(default_factory=Counter)
    raw_label_counts: Counter[str] = field(default_factory=Counter)
    intervals: list[int] = field(default_factory=list)
    first_timestamp: int | None = None
    last_timestamp: int | None = None

    def add_event(self, *, label: str, event_id: str, timestamp: int | None) -> None:
        self.events.append(event_id)
        self.event_counts[event_id] += 1
        self.raw_label_counts[label] += 1

        if timestamp is None:
            return
        if self.first_timestamp is None:
            self.first_timestamp = timestamp
        if self.last_timestamp is not None:
            self.intervals.append(max(0, timestamp - self.last_timestamp))
        self.last_timestamp = timestamp

    @property
    def binary_label(self) -> str:
        return "Anomaly" if any(label != "-" for label in self.raw_label_counts) else "Normal"

    @property
    def latency(self) -> int:
        if self.first_timestamp is None or self.last_timestamp is None:
            return 0
        return max(0, self.last_timestamp - self.first_timestamp)


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def binary_label(raw_label: str) -> str:
    return "Normal" if raw_label == "-" else "Anomaly"


def event_id_sort_key(event_id: str) -> tuple[int, str]:
    if event_id.startswith("E") and event_id[1:].isdigit():
        return int(event_id[1:]), event_id
    return 1_000_000_000, event_id


def parse_timestamp(value: str) -> int | None:
    try:
        return int(float(value))
    except ValueError:
        return None


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
        writer.writerow(["rank", field_name, "count", "share_of_rows", "share_of_nonempty"])
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


def write_node_event_traces_csv(path: Path, node_traces: dict[str, NodeTrace]) -> dict[str, object]:
    label_counts: Counter[str] = Counter()
    trace_lengths: list[int] = []
    latencies: list[int] = []

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Node", "Label", "Features", "TimeInterval", "Latency", "RawLabelCounts"])
        for node in sorted(node_traces):
            trace = node_traces[node]
            label_counts[trace.binary_label] += 1
            trace_lengths.append(len(trace.events))
            latencies.append(trace.latency)
            writer.writerow(
                [
                    node,
                    trace.binary_label,
                    " ".join(trace.events),
                    json.dumps(trace.intervals),
                    trace.latency,
                    json.dumps(dict(sorted(trace.raw_label_counts.items()))),
                ]
            )

    return {
        "path": str(path),
        "node_count": len(node_traces),
        "label_counts": dict(label_counts),
        "average_trace_length": sum(trace_lengths) / len(trace_lengths) if trace_lengths else 0.0,
        "max_trace_length": max(trace_lengths) if trace_lengths else 0,
        "average_latency": sum(latencies) / len(latencies) if latencies else 0.0,
        "max_latency": max(latencies) if latencies else 0,
    }


def write_node_occurrence_matrix_csv(
    path: Path,
    node_traces: dict[str, NodeTrace],
    event_ids: list[str],
) -> dict[str, object]:
    nonzero_counts: list[int] = []

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Node", "Label", *event_ids])
        for node in sorted(node_traces):
            trace = node_traces[node]
            nonzero_counts.append(sum(1 for event_id in event_ids if trace.event_counts[event_id] > 0))
            writer.writerow([node, trace.binary_label, *[trace.event_counts[event_id] for event_id in event_ids]])

    return {
        "path": str(path),
        "node_count": len(node_traces),
        "event_feature_count": len(event_ids),
        "average_nonzero_events_per_node": (
            sum(nonzero_counts) / len(nonzero_counts) if nonzero_counts else 0.0
        ),
        "max_nonzero_events_per_node": max(nonzero_counts) if nonzero_counts else 0,
    }


def write_markdown_report(path: Path, analysis: dict[str, object]) -> None:
    lines = [
        "# BGL Structured Basic Distribution Analysis",
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

    node_trace_stats = analysis.get("node_event_traces")
    if node_trace_stats:
        lines.extend(
            [
                "## Node Event Traces",
                f"- Output file: `{node_trace_stats['path']}`",
                f"- Nodes: {node_trace_stats['node_count']:,}",
                "- Features column: event sequence for each node-level trace",
                f"- Average trace length: {node_trace_stats['average_trace_length']:.2f}",
                f"- Max trace length: {node_trace_stats['max_trace_length']:,}",
                f"- Average latency: {node_trace_stats['average_latency']:.2f} seconds",
                f"- Max latency: {node_trace_stats['max_latency']:,} seconds",
                "",
                "| Label | Nodes |",
                "|---|---:|",
            ]
        )
        for label, count in sorted(node_trace_stats["label_counts"].items()):
            lines.append(f"| {label or '(missing)'} | {count:,} |")
        lines.append("")

    node_matrix_stats = analysis.get("node_occurrence_matrix")
    if node_matrix_stats:
        lines.extend(
            [
                "## Node Occurrence Matrix",
                f"- Output file: `{node_matrix_stats['path']}`",
                f"- Nodes: {node_matrix_stats['node_count']:,}",
                "- Columns: Node, Label, and one count column per EventId",
                f"- Event feature columns: {node_matrix_stats['event_feature_count']:,}",
                f"- Average non-zero event types per node: {node_matrix_stats['average_nonzero_events_per_node']:.2f}",
                f"- Max non-zero event types per node: {node_matrix_stats['max_nonzero_events_per_node']:,}",
                "",
            ]
        )

    path.write_text("\n".join(lines), encoding="utf-8")


def analyze_bgl_structured(
    path: Path,
    *,
    max_rows: int | None = None,
    progress_every: int = 1_000_000,
    collect_node_traces: bool = False,
) -> dict[str, object]:
    counters = {field: Counter() for field in FIELDS}
    nonempty_rows = {field: 0 for field in FIELDS}
    node_traces: defaultdict[str, NodeTrace] = defaultdict(NodeTrace)
    total_rows = 0

    with path.open("r", newline="", encoding="utf-8", errors="replace") as handle:
        reader = csv.reader(handle)
        header = next(reader, None)
        if header is None:
            raise ValueError("Input CSV is empty.")

        index_map = {name: idx for idx, name in enumerate(header)}
        required_fields = (*FIELDS, "Timestamp")
        missing = [field for field in required_fields if field not in index_map]
        if missing:
            raise ValueError(f"Input CSV is missing required columns: {', '.join(missing)}")

        label_idx = index_map["Label"]
        label_type_idx = index_map["LabelType"]
        event_idx = index_map["EventId"]
        node_idx = index_map["Node"]
        timestamp_idx = index_map["Timestamp"]

        for row in reader:
            if max_rows is not None and total_rows >= max_rows:
                break

            total_rows += 1
            if progress_every and total_rows % progress_every == 0:
                print(f"Processed {total_rows:,} rows")

            label_type = row[label_type_idx].strip()

            for field in FIELDS:
                value = row[index_map[field]].strip()
                if value:
                    counters[field][value] += 1
                    nonempty_rows[field] += 1

            if collect_node_traces:
                node = row[node_idx].strip()
                event_id = row[event_idx].strip()
                if node and event_id:
                    node_traces[node].add_event(
                        label=label_type,
                        event_id=event_id,
                        timestamp=parse_timestamp(row[timestamp_idx].strip()),
                    )

    field_summaries = {
        field: counter_summary(counters[field], total_rows=total_rows, nonempty_rows=nonempty_rows[field])
        for field in FIELDS
    }

    return {
        "input_path": str(path),
        "total_rows": total_rows,
        "max_rows": max_rows,
        "fields": field_summaries,
        "top_values": {field: counters[field].most_common(20) for field in FIELDS},
        "counters": counters,
        "nonempty_rows": nonempty_rows,
        "node_traces": dict(node_traces),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Path to BGL_structured.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for summary files.",
    )
    parser.add_argument(
        "--prefix",
        default="bgl_structured",
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
        "--full-node-csv",
        action="store_true",
        help="Write the full Node distribution CSV instead of only the top 1000 values.",
    )
    parser.add_argument(
        "--write-node-traces",
        action="store_true",
        help="Also write node-level Event_traces.csv and Event_occurrence_matrix.csv.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = resolve_path(args.input)
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    analysis = analyze_bgl_structured(
        input_path,
        max_rows=args.max_rows,
        progress_every=args.progress_every,
        collect_node_traces=args.write_node_traces,
    )

    serializable = {
        "input_path": analysis["input_path"],
        "total_rows": analysis["total_rows"],
        "max_rows": analysis["max_rows"],
        "fields": analysis["fields"],
        "top_values": analysis["top_values"],
    }

    counters: dict[str, Counter[str]] = analysis["counters"]
    nonempty_rows: dict[str, int] = analysis["nonempty_rows"]

    if args.write_node_traces:
        node_traces: dict[str, NodeTrace] = analysis["node_traces"]
        event_ids = sorted(counters["EventId"], key=event_id_sort_key)
        node_event_traces_path = output_dir / "Node_event_traces.csv"
        node_occurrence_matrix_path = output_dir / "Node_event_occurrence_matrix.csv"
        serializable["node_event_traces"] = write_node_event_traces_csv(node_event_traces_path, node_traces)
        serializable["node_occurrence_matrix"] = write_node_occurrence_matrix_csv(
            node_occurrence_matrix_path,
            node_traces,
            event_ids,
        )
        print(f"Wrote node event traces: {node_event_traces_path}")
        print(f"Wrote node occurrence matrix: {node_occurrence_matrix_path}")

    json_path = output_dir / f"{args.prefix}_distribution_summary.json"
    markdown_path = output_dir / f"{args.prefix}_distribution_summary.md"
    json_path.write_text(json.dumps(serializable, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown_report(markdown_path, serializable)

    for field in FIELDS:
        top_n = None
        if field == "Node" and not args.full_node_csv:
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
