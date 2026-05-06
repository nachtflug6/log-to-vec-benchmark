#!/usr/bin/env python3
"""Merge raw real-log files with structured log templates.

The raw log contributes timestamp/source/severity metadata, while the
structured CSV contributes LineId, EventId, and EventTemplate. Files are read
in lockstep by line order, which matches the LogPai-style structured outputs
used in this repository.
"""

from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
BLOCK_ID_PATTERN = re.compile(r"blk_-?\d+")


@dataclass(frozen=True)
class DatasetPreset:
    raw_log: Path
    structured_csv: Path
    output_csv: Path
    raw_columns: tuple[str, ...]
    derived_columns: tuple[str, ...] = ()
    label_csv: Path | None = None
    binary_label_column: str | None = None
    binary_label_source_column: str = "Label"


PRESETS = {
    "bgl": DatasetPreset(
        raw_log=Path("experiments/behavior_log/artifacts/datasets/BGL/original/BGL/BGL_full.log"),
        structured_csv=Path(
            "experiments/behavior_log/artifacts/datasets/BGL/original/BGL/BGL_full.log_structured.csv"
        ),
        output_csv=Path(
            "experiments/behavior_log/artifacts/datasets/BGL/structured/BGL_structured.csv"
        ),
        raw_columns=(
            "LabelType",
            "Timestamp",
            "Date",
            "Node",
            "Time",
            "NodeRepeat",
            "Type",
            "Component",
            "Level",
            "Content",
        ),
        binary_label_column="Label",
        binary_label_source_column="LabelType",
    ),
    "hdfs": DatasetPreset(
        raw_log=Path("experiments/behavior_log/artifacts/datasets/HDFS/original/HDFS_full.log"),
        structured_csv=Path(
            "experiments/behavior_log/artifacts/datasets/HDFS/original/HDFS_full.log_structured.csv"
        ),
        output_csv=Path(
            "experiments/behavior_log/artifacts/datasets/HDFS/structured/HDFS_structured.csv"
        ),
        raw_columns=("Date", "Time", "Pid", "Level", "Component", "Content"),
        derived_columns=("BlockId",),
        label_csv=Path("experiments/behavior_log/artifacts/datasets/HDFS/original/anomaly_label.csv"),
    ),
}

STRUCTURED_COLUMNS = ("LineId", "Content", "EventId", "EventTemplate")


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def parse_raw_line(line: str, raw_columns: tuple[str, ...], line_number: int) -> dict[str, str]:
    if "Content" not in raw_columns:
        raise ValueError("raw_columns must include a Content column.")

    content_index = raw_columns.index("Content")
    if content_index != len(raw_columns) - 1:
        raise ValueError("Content must be the last raw column because it may contain spaces.")

    fixed_column_count = len(raw_columns) - 1
    parts = line.rstrip("\n").split(None, fixed_column_count)
    if len(parts) != len(raw_columns):
        raise ValueError(
            f"Raw log line {line_number} has {len(parts)} fields, "
            f"but {len(raw_columns)} were expected: {line.rstrip()}"
        )

    return dict(zip(raw_columns, parts, strict=True))


def normalize_content(content: str) -> str:
    """Match the structured CSV behavior for raw log content.

    Some real logs contain NUL bytes. The structured BGL CSV is truncated at
    the first NUL byte, so we apply the same rule before checking alignment and
    writing the merged dataset.
    """
    return content.split("\x00", 1)[0]


def derive_fields(content: str, derived_columns: tuple[str, ...]) -> dict[str, str]:
    fields: dict[str, str] = {}
    if "BlockId" in derived_columns:
        match = BLOCK_ID_PATTERN.search(content)
        fields["BlockId"] = match.group(0) if match else ""
    return fields


def validate_structured_header(fieldnames: list[str] | None) -> None:
    missing = set(STRUCTURED_COLUMNS).difference(fieldnames or [])
    if missing:
        raise ValueError(f"Structured CSV is missing columns: {', '.join(sorted(missing))}")


def load_block_labels(label_csv: Path | None) -> dict[str, str]:
    if label_csv is None:
        return {}

    labels: dict[str, str] = {}
    with label_csv.open(newline="", encoding="utf-8", errors="replace") as handle:
        reader = csv.DictReader(handle)
        required_columns = {"BlockId", "Label"}
        missing = required_columns.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Label CSV is missing columns: {', '.join(sorted(missing))}")

        for row in reader:
            labels[row["BlockId"]] = row["Label"]
    return labels


def derive_binary_label(label: str) -> str:
    return "Normal" if label == "-" else "Anomaly"


def merge_logs(
    raw_log: Path,
    structured_csv: Path,
    output_csv: Path,
    raw_columns: tuple[str, ...],
    derived_columns: tuple[str, ...] = (),
    block_labels: dict[str, str] | None = None,
    binary_label_column: str | None = None,
    binary_label_source_column: str = "Label",
    *,
    max_rows: int | None = None,
    check_content: bool = True,
) -> dict[str, int]:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    label_columns = ("Label",) if block_labels and "Label" not in raw_columns else ()
    if binary_label_column:
        output_columns = [
            "LineId",
            binary_label_column,
            *raw_columns,
            *derived_columns,
            *label_columns,
            "EventId",
            "EventTemplate",
        ]
    else:
        output_columns = [
            "LineId",
            *raw_columns,
            *derived_columns,
            *label_columns,
            "EventId",
            "EventTemplate",
        ]

    rows_written = 0
    content_mismatches = 0
    missing_block_ids = 0
    missing_labels = 0
    raw_extra_lines = 0
    structured_extra_rows = 0
    block_labels = block_labels or {}

    with raw_log.open(encoding="utf-8", errors="replace") as raw_handle:
        with structured_csv.open(newline="", encoding="utf-8", errors="replace") as structured_handle:
            reader = csv.DictReader(structured_handle)
            validate_structured_header(reader.fieldnames)

            with output_csv.open("w", newline="", encoding="utf-8") as output_handle:
                writer = csv.DictWriter(output_handle, fieldnames=output_columns)
                writer.writeheader()

                for line_number, structured_row in enumerate(reader, start=1):
                    if max_rows is not None and rows_written >= max_rows:
                        break

                    raw_line = raw_handle.readline()
                    if raw_line == "":
                        structured_extra_rows += 1
                        break

                    raw_row = parse_raw_line(raw_line, raw_columns, line_number)
                    raw_row["Content"] = normalize_content(raw_row["Content"])
                    derived_row = derive_fields(raw_row["Content"], derived_columns)
                    missing_block_ids += int("BlockId" in derived_columns and not derived_row["BlockId"])
                    label_row: dict[str, str] = {}
                    if label_columns:
                        label = block_labels.get(derived_row.get("BlockId", ""))
                        missing_labels += int(label is None)
                        label_row["Label"] = label or ""
                    binary_label_row = {}
                    if binary_label_column:
                        if binary_label_source_column not in raw_row:
                            raise ValueError(
                                f"Binary label derivation requires a raw {binary_label_source_column} column."
                            )
                        binary_label_row[binary_label_column] = derive_binary_label(
                            raw_row[binary_label_source_column]
                        )
                    if check_content and raw_row["Content"] != structured_row["Content"]:
                        content_mismatches += 1

                    writer.writerow(
                        {
                            "LineId": structured_row["LineId"],
                            **raw_row,
                            **binary_label_row,
                            **derived_row,
                            **label_row,
                            "EventId": structured_row["EventId"],
                            "EventTemplate": structured_row["EventTemplate"],
                        }
                    )
                    rows_written += 1

                if max_rows is None:
                    raw_extra_lines = sum(1 for _ in raw_handle)

                    # If the raw log ended first, count the remaining structured rows.
                    if structured_extra_rows:
                        structured_extra_rows += sum(1 for _ in reader)

    return {
        "rows_written": rows_written,
        "content_mismatches": content_mismatches,
        "missing_block_ids": missing_block_ids,
        "missing_labels": missing_labels,
        "raw_extra_lines": raw_extra_lines,
        "structured_extra_rows": structured_extra_rows,
    }


def parse_raw_columns(value: str | None, preset: DatasetPreset) -> tuple[str, ...]:
    if value is None:
        return preset.raw_columns

    columns = tuple(column.strip() for column in value.split(",") if column.strip())
    if not columns:
        raise ValueError("--raw-columns cannot be empty.")
    return columns


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        choices=sorted(PRESETS),
        default="bgl",
        metavar="{bgl,hdfs}",
        help="Dataset preset to use. Choose from: bgl, hdfs. Default: bgl.",
    )

    parser.add_argument("--raw-log", type=Path, help="Override the preset raw log path.")
    parser.add_argument(
        "--structured-csv",
        type=Path,
        help="Override the preset structured CSV path.",
    )
    parser.add_argument("--output", type=Path, help="Override the preset output CSV path.")
    parser.add_argument(
        "--raw-columns",
        help=(
            "Comma-separated raw log columns. Content must be the last column, "
            "because it may contain spaces."
        ),
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        help="Optional row limit for smoke tests or small samples.",
    )
    parser.add_argument(
        "--skip-content-check",
        action="store_true",
        help="Do not compare raw Content against structured CSV Content.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    preset = PRESETS[args.dataset]

    raw_log = resolve_path(args.raw_log or preset.raw_log)
    structured_csv = resolve_path(args.structured_csv or preset.structured_csv)
    output_csv = resolve_path(args.output or preset.output_csv)
    raw_columns = parse_raw_columns(args.raw_columns, preset)
    derived_columns = preset.derived_columns if args.raw_columns is None else ()
    label_csv = resolve_path(preset.label_csv) if preset.label_csv and args.raw_columns is None else None
    block_labels = load_block_labels(label_csv)

    stats = merge_logs(
        raw_log=raw_log,
        structured_csv=structured_csv,
        output_csv=output_csv,
        raw_columns=raw_columns,
        derived_columns=derived_columns,
        block_labels=block_labels,
        binary_label_column=preset.binary_label_column if args.raw_columns is None else None,
        binary_label_source_column=preset.binary_label_source_column,
        max_rows=args.max_rows,
        check_content=not args.skip_content_check,
    )

    print(f"Dataset: {args.dataset}")
    print(f"Raw log: {raw_log}")
    print(f"Structured CSV: {structured_csv}")
    if label_csv:
        print(f"Label CSV: {label_csv}")
    print(f"Output CSV: {output_csv}")
    print(f"Rows written: {stats['rows_written']:,}")
    print(f"Content mismatches: {stats['content_mismatches']:,}")
    if "BlockId" in derived_columns:
        print(f"Missing block IDs: {stats['missing_block_ids']:,}")
    if block_labels:
        print(f"Missing labels: {stats['missing_labels']:,}")
    print(f"Extra raw lines: {stats['raw_extra_lines']:,}")
    print(f"Extra structured rows: {stats['structured_extra_rows']:,}")


if __name__ == "__main__":
    main()
