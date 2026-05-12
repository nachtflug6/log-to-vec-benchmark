#!/usr/bin/env python3
"""Join main operation rows with subprocess sequence summaries at case level."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def load_subprocess_sequences(path: Path) -> dict[str, dict[str, object]]:
    sequences: dict[str, dict[str, object]] = {}
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            subprocess_id = row.get("subprocess_id", "")
            if subprocess_id:
                sequences[str(subprocess_id)] = row
    return sequences


def load_main_operations(path: Path) -> dict[str, list[dict[str, str]]]:
    rows_by_case: dict[str, list[dict[str, str]]] = defaultdict(list)
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            rows_by_case[row["case_id"]].append(row)
    for rows in rows_by_case.values():
        rows.sort(key=lambda row: int(row["op_index"]))
    return rows_by_case


def build_case_records(
    rows_by_case: dict[str, list[dict[str, str]]],
    subprocess_sequences: dict[str, dict[str, object]],
) -> list[dict[str, object]]:
    case_records: list[dict[str, object]] = []

    for case_id in sorted(rows_by_case):
        main_rows = rows_by_case[case_id]
        operations: list[dict[str, object]] = []
        subprocess_count = 0
        sub_event_count = 0

        for row in main_rows:
            subprocess_id = row.get("subprocess_id", "")
            subprocess = subprocess_sequences.get(subprocess_id, {}) if subprocess_id else {}
            sub_activities = subprocess.get("sub_activities", [])
            if subprocess:
                subprocess_count += 1
                sub_event_count += int(subprocess.get("sub_event_count", 0))

            operations.append(
                {
                    "op_index": int(row["op_index"]),
                    "main_activity": row.get("activity", ""),
                    "resource": row.get("resource", ""),
                    "outcome": row.get("outcome", ""),
                    "subprocess_id": subprocess_id,
                    "sub_activities": sub_activities,
                }
            )

        case_records.append(
            {
                "case_id": case_id,
                "process_model_id": main_rows[0].get("process_model_id", "") if main_rows else "",
                "operation_count": len(main_rows),
                "subprocess_count": subprocess_count,
                "sub_event_count": sub_event_count,
                "has_failure": int(any(row.get("outcome") == "failure" for row in main_rows)),
                "operations": operations,
            }
        )

    return case_records


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False))
            fh.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--main-operations", type=Path, default=Path("/home/lyra/projects/dataset/main_operations_clean.csv"))
    parser.add_argument(
        "--subprocess-sequences",
        type=Path,
        default=Path("/home/lyra/projects/dataset/subprocess/subprocess_sequences.jsonl"),
    )
    parser.add_argument("--output", type=Path, default=Path("/home/lyra/projects/dataset/case_main_subprocess.jsonl"))
    args = parser.parse_args()

    subprocess_sequences = load_subprocess_sequences(args.subprocess_sequences)
    rows_by_case = load_main_operations(args.main_operations)
    case_records = build_case_records(rows_by_case, subprocess_sequences)
    write_jsonl(args.output, case_records)

    print(f"cases={len(case_records)}")
    print(f"operations={sum(row['operation_count'] for row in case_records)}")
    print(f"subprocesses={sum(row['subprocess_count'] for row in case_records)}")
    print(f"sub_events={sum(row['sub_event_count'] for row in case_records)}")
    print(f"output={args.output}")


if __name__ == "__main__":
    main()
