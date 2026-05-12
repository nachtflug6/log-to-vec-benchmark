#!/usr/bin/env python3
"""Create lightweight CSV extracts for the cleaned IoT-enriched XES log."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from xml.etree import ElementTree as ET


SIMPLE_TAGS = {"string", "date", "int", "float", "boolean"}

MAIN_EVENT_SEQUENCE_FIELDS = [
    "case_id",
    "event_index",
    "activity",
    "resource",
    "transition",
    "state",
    "timestamp",
    "end_time",
    "subprocess_id",
    "process_model_id",
    "planned_time",
    "service_time",
]

MAIN_EVENT_SEQUENCE_FIELD_MAP = {
    "event_index": "event_index",
    "activity": "concept:name",
    "resource": "org:resource",
    "transition": "lifecycle:transition",
    "state": "lifecycle:state",
    "timestamp": "time:timestamp",
    "end_time": "operation_end_time",
    "subprocess_id": "SubProcessID",
    "process_model_id": "process_model_id",
    "planned_time": "planned_operation_time",
    "service_time": "complete_service_time",
}

MAIN_OPERATION_FIELDS = [
    "case_id",
    "op_index",
    "activity",
    "resource",
    "scheduled_time",
    "start_time",
    "complete_time",
    "outcome",
    "subprocess_id",
    "process_model_id",
    "operation_duration_seconds",
    "waiting_time_seconds",
]


def local_name(tag: str) -> str:
    if "}" in tag:
        return tag.rsplit("}", 1)[1]
    return tag


def attr_key(elem: ET.Element) -> str | None:
    return elem.attrib.get("key")


def attr_value(elem: ET.Element) -> str:
    return elem.attrib.get("value", "")


def write_csv(path: Path, rows: list[dict[str, object]], preferred: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = set()
    for row in rows:
        keys.update(row.keys())
    fieldnames = preferred + sorted(keys.difference(preferred))
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_main_event_sequence_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    sequence_rows: list[dict[str, object]] = []
    for row in rows:
        sequence_row: dict[str, object] = {
            "case_id": row.get("case:concept:name") or row.get("case") or row.get("trace_concept:name", "")
        }
        for new_field, source_field in MAIN_EVENT_SEQUENCE_FIELD_MAP.items():
            sequence_row[new_field] = row.get(source_field, "")
        sequence_rows.append(sequence_row)
    return sequence_rows


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


def elapsed_seconds(start: object, end: object) -> str:
    start_dt = parse_timestamp(start)
    end_dt = parse_timestamp(end)
    if start_dt is None or end_dt is None:
        return ""
    return f"{(end_dt - start_dt).total_seconds():.6f}".rstrip("0").rstrip(".")


def case_id_for_row(row: dict[str, object]) -> str:
    return str(row.get("case:concept:name") or row.get("case") or row.get("trace_concept:name", ""))


def build_main_operation_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    operations: dict[tuple[str, str], dict[str, object]] = {}
    first_event_index: dict[tuple[str, str], int] = {}

    for row in rows:
        case_id = case_id_for_row(row)
        event_id = str(row.get("event_id") or row.get("event_index") or "")
        key = (case_id, event_id)
        operation = operations.setdefault(
            key,
            {
                "case_id": case_id,
                "activity": row.get("concept:name", ""),
                "resource": row.get("org:resource", ""),
                "scheduled_time": "",
                "start_time": "",
                "complete_time": "",
                "outcome": "",
                "subprocess_id": "",
                "process_model_id": row.get("process_model_id", ""),
            },
        )
        operation["activity"] = operation.get("activity") or row.get("concept:name", "")
        operation["resource"] = operation.get("resource") or row.get("org:resource", "")
        operation["process_model_id"] = operation.get("process_model_id") or row.get("process_model_id", "")

        try:
            event_index = int(str(row.get("event_index", "0")))
        except ValueError:
            event_index = 0
        first_event_index[key] = min(first_event_index.get(key, event_index), event_index)

        transition = row.get("lifecycle:transition", "")
        timestamp = row.get("time:timestamp", "")
        if transition == "scheduled":
            operation["scheduled_time"] = timestamp
        elif transition == "start":
            operation["start_time"] = timestamp
            operation["subprocess_id"] = row.get("SubProcessID", "")
        elif transition == "complete":
            operation["complete_time"] = timestamp
            operation["outcome"] = row.get("lifecycle:state", "")

    rows_by_case: dict[str, list[tuple[tuple[str, str], dict[str, object]]]] = defaultdict(list)
    for key, operation in operations.items():
        rows_by_case[key[0]].append((key, operation))

    operation_rows: list[dict[str, object]] = []
    for case_id in sorted(rows_by_case):
        case_operations = sorted(rows_by_case[case_id], key=lambda item: first_event_index[item[0]])
        for op_index, (_, operation) in enumerate(case_operations):
            operation_row = {
                **operation,
                "op_index": op_index,
                "operation_duration_seconds": elapsed_seconds(
                    operation.get("start_time", ""),
                    operation.get("complete_time", ""),
                ),
                "waiting_time_seconds": elapsed_seconds(
                    operation.get("scheduled_time", ""),
                    operation.get("start_time", ""),
                ),
            }
            operation_rows.append(operation_row)

    return operation_rows


def write_case_main_only_jsonl(path: Path, operation_rows: list[dict[str, object]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows_by_case: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in operation_rows:
        rows_by_case[str(row["case_id"])].append(row)

    with path.open("w", encoding="utf-8") as fh:
        for case_id in sorted(rows_by_case):
            case_rows = sorted(rows_by_case[case_id], key=lambda row: int(row["op_index"]))
            record = {
                "case_id": case_id,
                "process_model_id": case_rows[0].get("process_model_id", "") if case_rows else "",
                "main_operations": [
                    f"{row.get('activity', '') or ''}|{row.get('resource', '') or ''}" for row in case_rows
                ],
                "operation_count": len(case_rows),
                "has_failure": any(row.get("outcome") == "failure" for row in case_rows),
            }
            fh.write(json.dumps(record, ensure_ascii=False))
            fh.write("\n")
    return len(rows_by_case)


def parse_main_process(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    trace_attrs: dict[str, str] = {}
    current_event: dict[str, str] | None = None
    stack: list[tuple[str, str | None]] = []
    trace_index = -1
    event_index = -1

    for action, elem in ET.iterparse(path, events=("start", "end")):
        tag = local_name(elem.tag)
        key = attr_key(elem)

        if action == "start":
            stack.append((tag, key))
            if tag == "trace":
                trace_index += 1
                event_index = -1
                trace_attrs = {"trace_index": str(trace_index)}
            elif tag == "event":
                event_index += 1
                current_event = {"event_index": str(event_index)}
            continue

        parent_tag = stack[-2][0] if len(stack) >= 2 else None

        if tag in SIMPLE_TAGS and key:
            value = attr_value(elem)
            if current_event is not None and parent_tag == "event":
                current_event[key] = value
            elif parent_tag == "trace":
                trace_attrs[f"trace_{key}"] = value
        elif tag == "list" and key == "parameters" and current_event is not None:
            params = {}
            for child in elem.iter():
                child_tag = local_name(child.tag)
                child_key = attr_key(child)
                if child_tag in SIMPLE_TAGS and child_key:
                    params[child_key] = attr_value(child)
            if params:
                current_event["parameters_json"] = json.dumps(params, ensure_ascii=False, sort_keys=True)
        elif tag == "event" and current_event is not None:
            rows.append({**trace_attrs, **current_event})
            current_event = None
            elem.clear()
        elif tag == "trace":
            elem.clear()

        stack.pop()

    return rows


def value_shape(value: str) -> str:
    text = value.strip()
    if text.startswith("[") and text.endswith("]"):
        inner = text[1:-1].strip()
        if not inner:
            return "list[0]"
        return f"list[{inner.count(',') + 1}]"
    if text in {"true", "false"}:
        return "boolean"
    try:
        float(text)
    except ValueError:
        return "string"
    return "scalar"


def short_uri(value: str) -> str:
    if "#" in value:
        return value.rsplit("#", 1)[1]
    return value.rsplit("/", 1)[-1]


def parse_subprocess_file(
    path: Path,
    sample_number: int,
    sensor_summary: dict[tuple[str, str, str, str], dict[str, object]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    stack: list[tuple[str, str | None]] = []
    subprocess_id = path.stem
    current_event: dict[str, object] | None = None
    current_point: dict[str, str] | None = None
    event_index = -1

    for action, elem in ET.iterparse(path, events=("start", "end")):
        tag = local_name(elem.tag)
        key = attr_key(elem)

        if action == "start":
            stack.append((tag, key))
            if tag == "event":
                event_index += 1
                current_event = {
                    "sample_number": sample_number,
                    "subprocess_file": path.name,
                    "subprocess_id": subprocess_id,
                    "event_index": event_index,
                    "sensor_point_count": 0,
                    "_systems": set(),
                    "_observations": set(),
                    "_specs": set(),
                    "_examples": [],
                }
            elif current_event is not None and tag == "list" and key == "stream:point":
                current_point = {
                    "system": elem.attrib.get("{https://cpee.org/datastream/datastream.xesext}system", "")
                    or elem.attrib.get("stream:system", ""),
                    "system_type": elem.attrib.get("{https://cpee.org/datastream/datastream.xesext}system_type", "")
                    or elem.attrib.get("stream:system_type", ""),
                    "observation": elem.attrib.get("{https://cpee.org/datastream/datastream.xesext}observation", "")
                    or elem.attrib.get("stream:observation", ""),
                    "observation_specification": elem.attrib.get(
                        "{https://cpee.org/datastream/datastream.xesext}observation_specification", ""
                    )
                    or elem.attrib.get("stream:observation_specification", ""),
                    "procedure_type": elem.attrib.get("{https://cpee.org/datastream/datastream.xesext}procedure_type", "")
                    or elem.attrib.get("stream:procedure_type", ""),
                    "interaction_type": elem.attrib.get("{https://cpee.org/datastream/datastream.xesext}interaction_type", "")
                    or elem.attrib.get("stream:interaction_type", ""),
                }
            continue

        parent_tag = stack[-2][0] if len(stack) >= 2 else None

        if tag in SIMPLE_TAGS and key:
            value = attr_value(elem)
            if current_point is not None:
                current_point[key] = value
            elif current_event is not None and parent_tag == "event":
                current_event[key] = value
        elif tag == "list" and key == "stream:point" and current_event is not None and current_point is not None:
            value = current_point.get("stream:value", "")
            system = current_point.get("system", "")
            observation = current_point.get("observation", "")
            spec = current_point.get("observation_specification", "")
            shape = value_shape(value)
            summary_key = (system, observation, spec, shape)
            summary = sensor_summary.setdefault(
                summary_key,
                {
                    "system": system,
                    "system_short": short_uri(system),
                    "system_type": current_point.get("system_type", ""),
                    "observation": observation,
                    "observation_short": short_uri(observation),
                    "observation_specification": spec,
                    "procedure_type": current_point.get("procedure_type", ""),
                    "interaction_type": current_point.get("interaction_type", ""),
                    "value_shape": shape,
                    "point_count": 0,
                    "files": set(),
                    "subprocesses": set(),
                    "events": set(),
                    "example_values": [],
                    "first_timestamp": "",
                    "last_timestamp": "",
                },
            )
            summary["point_count"] = int(summary["point_count"]) + 1
            summary["files"].add(path.name)
            summary["subprocesses"].add(subprocess_id)
            summary["events"].add((path.name, event_index))
            timestamp = current_point.get("stream:timestamp", "")
            if timestamp:
                if not summary["first_timestamp"] or timestamp < summary["first_timestamp"]:
                    summary["first_timestamp"] = timestamp
                if not summary["last_timestamp"] or timestamp > summary["last_timestamp"]:
                    summary["last_timestamp"] = timestamp
            examples = summary["example_values"]
            if value and value not in examples and len(examples) < 5:
                examples.append(value)

            current_event["sensor_point_count"] = int(current_event["sensor_point_count"]) + 1
            current_event["_systems"].add(short_uri(system))
            current_event["_observations"].add(short_uri(observation))
            current_event["_specs"].add(spec)
            if value and len(current_event["_examples"]) < 5:
                current_event["_examples"].append(value)
            current_point = None
            elem.clear()
        elif tag == "event" and current_event is not None:
            systems = sorted(current_event.pop("_systems"))
            observations = sorted(current_event.pop("_observations"))
            specs = sorted(current_event.pop("_specs"))
            examples = current_event.pop("_examples")
            current_event["sensor_system_count"] = len(systems)
            current_event["sensor_observation_count"] = len(observations)
            current_event["sensor_spec_count"] = len(specs)
            current_event["sensor_systems_sample"] = "|".join(systems[:10])
            current_event["sensor_observations_sample"] = "|".join(observations[:10])
            current_event["sensor_specs_sample"] = "|".join(specs[:5])
            current_event["sensor_value_examples"] = "|".join(examples)
            rows.append(current_event)
            current_event = None
            elem.clear()
        elif tag == "trace":
            elem.clear()

        stack.pop()

    return rows


def serialize_sensor_summary(summary: dict[tuple[str, str, str, str], dict[str, object]]) -> list[dict[str, object]]:
    rows = []
    for item in summary.values():
        row = dict(item)
        row["file_count"] = len(row.pop("files"))
        row["subprocess_count"] = len(row.pop("subprocesses"))
        row["event_count"] = len(row.pop("events"))
        row["example_values"] = "|".join(row["example_values"])
        rows.append(row)
    rows.sort(key=lambda row: int(row["point_count"]), reverse=True)
    return rows


def has_sensor_points(path: Path) -> bool:
    needle = b"stream:point"
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(1024 * 1024)
            if not chunk:
                return False
            if needle in chunk:
                return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, default=Path("/home/lyra/projects/dataset/Cleaned Event Log"))
    parser.add_argument("--output-dir", type=Path, default=Path("/home/lyra/projects/dataset"))
    parser.add_argument("--sample-size", type=int, default=100)
    parser.add_argument(
        "--sample-mode",
        choices=("smallest", "name"),
        default="smallest",
        help="Choose subprocess files by smallest file size or filename order.",
    )
    parser.add_argument(
        "--require-sensor-points",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Only sample subprocess files that contain stream:point sensor entries.",
    )
    args = parser.parse_args()

    main_path = args.dataset_dir / "MainProcess.xes"
    main_rows = parse_main_process(main_path)
    write_csv(
        args.output_dir / "main_events.csv",
        main_rows,
        [
            "trace_index",
            "trace_concept:name",
            "event_index",
            "event_id",
            "case",
            "case:concept:name",
            "concept:name",
            "lifecycle:transition",
            "lifecycle:state",
            "time:timestamp",
            "operation_end_time",
            "org:resource",
            "SubProcessID",
            "current_task",
        ],
    )
    main_sequence_rows = build_main_event_sequence_rows(main_rows)
    write_csv(
        args.output_dir / "main_event_sequence.csv",
        main_sequence_rows,
        MAIN_EVENT_SEQUENCE_FIELDS,
    )
    main_operation_rows = build_main_operation_rows(main_rows)
    write_csv(
        args.output_dir / "main_operations_clean.csv",
        main_operation_rows,
        MAIN_OPERATION_FIELDS,
    )
    case_main_count = write_case_main_only_jsonl(
        args.output_dir / "case_main_only.jsonl",
        main_operation_rows,
    )

    subprocess_files = [path for path in args.dataset_dir.glob("*.xes") if path.name != "MainProcess.xes"]
    if args.sample_mode == "smallest":
        subprocess_files.sort(key=lambda path: (path.stat().st_size, path.name))
    else:
        subprocess_files.sort()
    if args.require_sensor_points:
        sample_files = []
        for path in subprocess_files:
            if has_sensor_points(path):
                sample_files.append(path)
            if len(sample_files) >= args.sample_size:
                break
    else:
        sample_files = subprocess_files[: args.sample_size]
    sensor_summary: dict[tuple[str, str, str, str], dict[str, object]] = {}
    sub_rows: list[dict[str, object]] = []
    for sample_number, path in enumerate(sample_files, start=1):
        sub_rows.extend(parse_subprocess_file(path, sample_number, sensor_summary))

    write_csv(
        args.output_dir / "sample_sub_events.csv",
        sub_rows,
        [
            "sample_number",
            "subprocess_file",
            "subprocess_id",
            "event_index",
            "concept:name",
            "org:resource",
            "time:timestamp",
            "operation_end_time",
            "sensor_point_count",
            "sensor_system_count",
            "sensor_observation_count",
            "sensor_spec_count",
            "sensor_systems_sample",
            "sensor_observations_sample",
            "sensor_specs_sample",
            "sensor_value_examples",
        ],
    )
    write_csv(
        args.output_dir / "sample_sensor_summary.csv",
        serialize_sensor_summary(sensor_summary),
        [
            "system_short",
            "observation_short",
            "value_shape",
            "point_count",
            "file_count",
            "subprocess_count",
            "event_count",
            "system_type",
            "procedure_type",
            "interaction_type",
            "observation_specification",
            "first_timestamp",
            "last_timestamp",
            "example_values",
            "system",
            "observation",
        ],
    )

    with (args.output_dir / "sample_subprocess_files.txt").open("w", encoding="utf-8") as fh:
        for path in sample_files:
            fh.write(f"{path}\n")

    print(f"main_events={len(main_rows)}")
    print(f"main_event_sequence={len(main_sequence_rows)}")
    print(f"main_operations_clean={len(main_operation_rows)}")
    print(f"case_main_only={case_main_count}")
    print(f"sample_sub_events={len(sub_rows)}")
    print(f"sensor_summary_rows={len(sensor_summary)}")
    print(f"output_dir={args.output_dir}")


if __name__ == "__main__":
    main()
