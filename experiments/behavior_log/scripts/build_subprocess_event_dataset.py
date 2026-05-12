#!/usr/bin/env python3
"""Build clean event rows and sequence samples for all subprocess XES files."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from xml.etree import ElementTree as ET


SIMPLE_TAGS = {"string", "date", "int", "float", "boolean"}

SUBPROCESS_EVENT_FIELDS = [
    "subprocess_id",
    "sub_event_index",
    "sub_activity",
    "sub_resource",
    "sub_timestamp",
    "sub_end_time",
]


def local_name(tag: str) -> str:
    if "}" in tag:
        return tag.rsplit("}", 1)[1]
    return tag


def attr_key(elem: ET.Element) -> str | None:
    return elem.attrib.get("key")


def attr_value(elem: ET.Element) -> str:
    return elem.attrib.get("value", "")


def parse_timestamp(value: object) -> datetime | None:
    if not value:
        return None
    text = str(value).strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def duration_seconds(start: object, end: object) -> str:
    start_dt = parse_timestamp(start)
    end_dt = parse_timestamp(end)
    if start_dt is None or end_dt is None:
        return ""
    return f"{(end_dt - start_dt).total_seconds():.6f}".rstrip("0").rstrip(".")


def parse_subprocess_file(path: Path) -> tuple[list[dict[str, object]], str | None]:
    rows: list[dict[str, object]] = []
    stack: list[tuple[str, str | None]] = []
    subprocess_id = path.stem
    current_event: dict[str, object] | None = None
    sub_event_index = -1

    try:
        for action, elem in ET.iterparse(path, events=("start", "end")):
            tag = local_name(elem.tag)
            key = attr_key(elem)

            if action == "start":
                stack.append((tag, key))
                if tag == "event":
                    sub_event_index += 1
                    current_event = {
                        "subprocess_id": subprocess_id,
                        "sub_event_index": sub_event_index,
                    }
                continue

            parent_tag = stack[-2][0] if len(stack) >= 2 else None

            if tag in SIMPLE_TAGS and key:
                value = attr_value(elem)
                if current_event is not None and parent_tag == "event":
                    current_event[key] = value
            elif tag == "event" and current_event is not None:
                activity = current_event.get("concept:name", "")
                resource = current_event.get("org:resource", "")
                timestamp = current_event.get("time:timestamp", "")
                end_time = current_event.get("operation_end_time", "")

                rows.append(
                    {
                        "subprocess_id": subprocess_id,
                        "sub_event_index": current_event["sub_event_index"],
                        "sub_activity": activity,
                        "sub_resource": resource,
                        "sub_timestamp": timestamp,
                        "sub_end_time": end_time,
                    }
                )
                current_event = None
                elem.clear()
            elif tag == "trace":
                elem.clear()

            stack.pop()
    except ET.ParseError as exc:
        return rows, f"{path}: {exc}"

    return rows, None


def update_sequence_summary(summaries: dict[str, dict[str, object]], row: dict[str, object]) -> None:
    subprocess_id = str(row["subprocess_id"])
    summary = summaries.setdefault(
        subprocess_id,
        {
            "sub_activities": [],
            "sub_resources": set(),
            "sub_event_count": 0,
            "start_time": None,
            "end_time": None,
        },
    )
    summary["sub_activities"].append(str(row["sub_activity"]))
    if row["sub_resource"]:
        summary["sub_resources"].add(str(row["sub_resource"]))
    summary["sub_event_count"] = int(summary["sub_event_count"]) + 1

    start = parse_timestamp(row["sub_timestamp"])
    end = parse_timestamp(row["sub_end_time"])
    if start is not None and (summary["start_time"] is None or start < summary["start_time"]):
        summary["start_time"] = start
    if end is not None and (summary["end_time"] is None or end > summary["end_time"]):
        summary["end_time"] = end


def load_main_operation_by_subprocess(path: Path | None) -> dict[str, dict[str, object]]:
    if path is None or not path.exists():
        return {}

    main_operation_by_subprocess: dict[str, dict[str, object]] = {}
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            subprocess_id = row.get("subprocess_id", "")
            if subprocess_id:
                main_operation_by_subprocess[subprocess_id] = {
                    "case_id": row.get("case_id", ""),
                    "process_model_id": row.get("process_model_id", ""),
                    "parent_op_index": row.get("op_index", ""),
                    "main_activity": row.get("activity", ""),
                    "has_failure": int(row.get("outcome", "") == "failure"),
                }
    return main_operation_by_subprocess


def write_sequences_jsonl(
    path: Path,
    summaries: dict[str, dict[str, object]],
    main_operation_by_subprocess: dict[str, dict[str, object]] | None = None,
) -> int:
    main_operation_by_subprocess = main_operation_by_subprocess or {}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for subprocess_id in sorted(summaries):
            summary = summaries[subprocess_id]
            start_dt = summary["start_time"]
            end_dt = summary["end_time"]
            start_time = start_dt.isoformat() if isinstance(start_dt, datetime) else ""
            end_time = end_dt.isoformat() if isinstance(end_dt, datetime) else ""
            duration = duration_seconds(start_time, end_time)
            sub_resources = sorted(summary["sub_resources"])
            main_operation = main_operation_by_subprocess.get(subprocess_id, {})

            record = {
                "subprocess_id": subprocess_id,
                "case_id": main_operation.get("case_id", ""),
                "process_model_id": main_operation.get("process_model_id", ""),
                "parent_op_index": main_operation.get("parent_op_index", ""),
                "main_activity": main_operation.get("main_activity", ""),
                "has_failure": main_operation.get("has_failure", 0),
                "resource": sub_resources[0] if len(sub_resources) == 1 else "|".join(sub_resources),
                "sub_event_count": summary["sub_event_count"],
                "sub_activities": summary["sub_activities"],
                "duration_seconds": float(duration) if duration else "",
                "start_time": start_time,
                "end_time": end_time,
            }
            fh.write(json.dumps(record, ensure_ascii=False))
            fh.write("\n")
    return len(summaries)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, default=Path("/home/lyra/projects/dataset/Cleaned Event Log"))
    parser.add_argument("--output-dir", type=Path, default=Path("/home/lyra/projects/dataset"))
    parser.add_argument(
        "--main-operations-path",
        type=Path,
        default=Path("/home/lyra/projects/dataset/main_operations_clean.csv"),
        help="Optional main operation CSV used to add main_activity to subprocess sequence JSONL.",
    )
    parser.add_argument("--events-output", type=Path)
    parser.add_argument("--sequences-output", type=Path)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail on malformed subprocess XES files instead of keeping parsed events and continuing.",
    )
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--max-files", type=int, help="Process only the first N selected files for quick checks.")
    args = parser.parse_args()

    if args.num_shards < 1:
        raise ValueError("--num-shards must be >= 1")
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("--shard-index must satisfy 0 <= shard-index < num-shards")

    subprocess_files = [path for path in args.dataset_dir.glob("*.xes") if path.name != "MainProcess.xes"]
    subprocess_files.sort()
    all_file_count = len(subprocess_files)
    if args.num_shards > 1:
        subprocess_files = [
            path for index, path in enumerate(subprocess_files) if index % args.num_shards == args.shard_index
        ]
    if args.max_files is not None:
        subprocess_files = subprocess_files[: args.max_files]

    event_count = 0
    sequence_summaries: dict[str, dict[str, object]] = {}
    parse_warnings: list[str] = []
    if args.num_shards > 1:
        default_events_name = f"subprocess_events_clean_shard_{args.shard_index:04d}.csv"
        default_sequences_name = f"subprocess_sequences_shard_{args.shard_index:04d}.jsonl"
    else:
        default_events_name = "subprocess_events_clean.csv"
        default_sequences_name = "subprocess_sequences.jsonl"
    events_path = args.events_output or args.output_dir / default_events_name
    sequences_path = args.sequences_output or args.output_dir / default_sequences_name

    args.output_dir.mkdir(parents=True, exist_ok=True)
    events_path.parent.mkdir(parents=True, exist_ok=True)
    sequences_path.parent.mkdir(parents=True, exist_ok=True)
    print(
        f"all_subprocess_files={all_file_count} selected_files={len(subprocess_files)} "
        f"shard_index={args.shard_index} num_shards={args.num_shards}",
        flush=True,
    )
    with events_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=SUBPROCESS_EVENT_FIELDS)
        writer.writeheader()
        for index, path in enumerate(subprocess_files, start=1):
            size_mb = path.stat().st_size / (1024 * 1024)
            print(
                f"parsing_file={index}/{len(subprocess_files)} name={path.name} size_mb={size_mb:.1f}",
                flush=True,
            )
            file_rows, warning = parse_subprocess_file(path)
            file_rows.sort(key=lambda row: int(row["sub_event_index"]))
            writer.writerows(file_rows)
            fh.flush()
            for row in file_rows:
                update_sequence_summary(sequence_summaries, row)
            event_count += len(file_rows)

            if warning:
                if args.strict:
                    raise ET.ParseError(warning)
                parse_warnings.append(warning)
                print(f"warning_parse_error={warning}", flush=True)
            if args.progress_every and index % args.progress_every == 0:
                print(f"parsed_subprocess_files={index} subprocess_events={event_count}", flush=True)

    main_operation_by_subprocess = load_main_operation_by_subprocess(args.main_operations_path)
    sequence_count = write_sequences_jsonl(sequences_path, sequence_summaries, main_operation_by_subprocess)

    print(f"subprocess_files={len(subprocess_files)}")
    print(f"subprocess_events={event_count}")
    print(f"subprocess_sequences={sequence_count}")
    print(f"parse_warnings={len(parse_warnings)}")
    print(f"events_path={events_path}")
    print(f"sequences_path={sequences_path}")


if __name__ == "__main__":
    main()
