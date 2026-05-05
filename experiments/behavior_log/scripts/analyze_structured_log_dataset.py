#!/usr/bin/env python3
"""Analyze a structured log CSV dataset.

The script streams the CSV file row by row, so it can handle large files such
as the full BGL structured log without loading the whole dataset into memory.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[3]

DEFAULT_INPUT = Path(
    "experiments/behavior_log/artifacts/datasets/BGL/original/BGL_full.log_structured.csv"
)


def percentile(sorted_values: list[int], q: float) -> int:
    """Return the nearest-rank percentile from an already sorted list."""
    if not sorted_values:
        return 0
    index = round((len(sorted_values) - 1) * q)
    return sorted_values[index]


def cumulative_share(counter: Counter[str], top_k_values: Iterable[int], total: int) -> dict[int, float]:
    counts = [count for _, count in counter.most_common()]
    shares: dict[int, float] = {}
    for top_k in top_k_values:
        shares[top_k] = sum(counts[:top_k]) / total if total else 0.0
    return shares


def analyze_csv(path: Path) -> dict:
    event_counts: Counter[str] = Counter()
    template_counts: Counter[str] = Counter()
    word_lengths: list[int] = []

    total_rows = 0
    empty_content = 0
    empty_event_id = 0
    empty_template = 0
    min_line_id: int | None = None
    max_line_id: int | None = None
    line_id_jumps = 0
    previous_line_id = 0

    with path.open(newline="", encoding="utf-8", errors="replace") as handle:
        reader = csv.DictReader(handle)
        required_columns = {"LineId", "Content", "EventId", "EventTemplate"}
        missing = required_columns.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")

        for row in reader:
            total_rows += 1

            content = row["Content"]
            event_id = row["EventId"]
            template = row["EventTemplate"]

            empty_content += int(not content)
            empty_event_id += int(not event_id)
            empty_template += int(not template)

            event_counts[event_id] += 1
            template_counts[template] += 1
            word_lengths.append(len(content.split()))

            try:
                line_id = int(row["LineId"])
            except ValueError:
                line_id_jumps += 1
                continue

            min_line_id = line_id if min_line_id is None else min(min_line_id, line_id)
            max_line_id = line_id if max_line_id is None else max(max_line_id, line_id)
            if line_id != previous_line_id + 1:
                line_id_jumps += 1
            previous_line_id = line_id

    sorted_lengths = sorted(word_lengths)
    top_shares = cumulative_share(event_counts, [1, 5, 10, 20, 50], total_rows)

    return {
        "input_path": str(path),
        "total_rows": total_rows,
        "unique_event_ids": len(event_counts),
        "unique_templates": len(template_counts),
        "empty_content": empty_content,
        "empty_event_id": empty_event_id,
        "empty_template": empty_template,
        "line_id": {
            "min": min_line_id,
            "max": max_line_id,
            "non_consecutive_or_invalid": line_id_jumps,
        },
        "word_length": {
            "average": sum(word_lengths) / total_rows if total_rows else 0.0,
            "min": min(sorted_lengths) if sorted_lengths else 0,
            "p50": percentile(sorted_lengths, 0.50),
            "p90": percentile(sorted_lengths, 0.90),
            "p95": percentile(sorted_lengths, 0.95),
            "p99": percentile(sorted_lengths, 0.99),
            "max": max(sorted_lengths) if sorted_lengths else 0,
        },
        "event_concentration": {f"top_{k}": share for k, share in top_shares.items()},
        "rare_events": {
            "singletons": sum(1 for count in event_counts.values() if count == 1),
            "less_than_10": sum(1 for count in event_counts.values() if count < 10),
            "less_than_100": sum(1 for count in event_counts.values() if count < 100),
        },
        "top_events": event_counts.most_common(20),
        "top_templates": template_counts.most_common(20),
    }


def write_event_counts(path: Path, analysis: dict) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["rank", "event_id", "count", "share"])
        total = analysis["total_rows"]
        for rank, (event_id, count) in enumerate(analysis["top_events"], start=1):
            writer.writerow([rank, event_id, count, count / total if total else 0.0])


def write_markdown(path: Path, analysis: dict) -> None:
    lines = [
        "# Structured Log Dataset Analysis",
        "",
        f"- Input file: `{analysis['input_path']}`",
        f"- Total log records: {analysis['total_rows']:,}",
        f"- Unique event IDs: {analysis['unique_event_ids']:,}",
        f"- Unique event templates: {analysis['unique_templates']:,}",
        f"- Empty content rows: {analysis['empty_content']:,}",
        f"- Empty event IDs: {analysis['empty_event_id']:,}",
        f"- Empty templates: {analysis['empty_template']:,}",
        "",
        "## Event Concentration",
    ]

    for name, share in analysis["event_concentration"].items():
        lines.append(f"- {name.replace('_', ' ').title()}: {share:.2%}")

    lines.extend(
        [
            "",
            "## Rare Events",
            f"- Singleton event IDs: {analysis['rare_events']['singletons']:,}",
            f"- Event IDs with fewer than 10 records: {analysis['rare_events']['less_than_10']:,}",
            f"- Event IDs with fewer than 100 records: {analysis['rare_events']['less_than_100']:,}",
            "",
            "## Word Length",
        ]
    )

    for name, value in analysis["word_length"].items():
        if isinstance(value, float):
            lines.append(f"- {name}: {value:.2f}")
        else:
            lines.append(f"- {name}: {value}")

    lines.extend(["", "## Top Event IDs", "", "| Rank | Event ID | Count | Share |", "|---:|---|---:|---:|"])
    total = analysis["total_rows"]
    for rank, (event_id, count) in enumerate(analysis["top_events"][:10], start=1):
        share = count / total if total else 0.0
        lines.append(f"| {rank} | {event_id} | {count:,} | {share:.2%} |")

    lines.extend(["", "## Top Event Templates", "", "| Rank | Count | Share | Template |", "|---:|---:|---:|---|"])
    for rank, (template, count) in enumerate(analysis["top_templates"][:10], start=1):
        share = count / total if total else 0.0
        safe_template = template.replace("|", "\\|")
        lines.append(f"| {rank} | {count:,} | {share:.2%} | `{safe_template}` |")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Path to a structured log CSV file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/dataset_analysis"),
        help="Directory for generated analysis files.",
    )
    parser.add_argument(
        "--prefix",
        default="bgl_full",
        help="Filename prefix for generated outputs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = args.input
    if not input_path.is_absolute():
        input_path = PROJECT_ROOT / input_path

    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir

    output_dir.mkdir(parents=True, exist_ok=True)
    analysis = analyze_csv(input_path)

    json_path = output_dir / f"{args.prefix}_analysis.json"
    markdown_path = output_dir / f"{args.prefix}_analysis.md"
    counts_path = output_dir / f"{args.prefix}_top_events.csv"

    json_path.write_text(json.dumps(analysis, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(markdown_path, analysis)
    write_event_counts(counts_path, analysis)

    print(f"Wrote JSON summary: {json_path}")
    print(f"Wrote Markdown report: {markdown_path}")
    print(f"Wrote top event counts: {counts_path}")


if __name__ == "__main__":
    main()
