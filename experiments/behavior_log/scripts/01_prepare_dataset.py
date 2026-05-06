#!/usr/bin/env python3
"""Stage 01: prepare dataset traces, occurrence matrices, and splits."""

from __future__ import annotations

import csv
import json
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from random import Random
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from behavior_log.utils.io import load_yaml, save_json

CONFIG_DIR = ROOT / "configs" / "preparation"
REPO_ROOT = ROOT.parents[1]


@dataclass
class EventRecord:
    sort_key: tuple[Any, ...]
    token: str
    timestamp: float | None
    timestamp_display: str
    label: str
    metadata_values: dict[str, str]


@dataclass
class TraceAccumulator:
    label_votes: Counter[str] = field(default_factory=Counter)
    fallback_label: str = ""
    events: list[EventRecord] = field(default_factory=list)

    def add_event(
        self,
        *,
        sort_key: tuple[Any, ...],
        token: str,
        timestamp: float | None,
        timestamp_display: str,
        label: str,
        count_label_vote: bool,
        metadata_values: dict[str, str],
    ) -> None:
        if label and not self.fallback_label:
            self.fallback_label = label
        if count_label_vote and label:
            self.label_votes[label] += 1
        self.events.append(
            EventRecord(
                sort_key=sort_key,
                token=token,
                timestamp=timestamp,
                timestamp_display=timestamp_display,
                label=label,
                metadata_values=metadata_values,
            )
        )


@dataclass
class PreparedTrace:
    sample_id: str
    label: str
    tokens: list[str]
    latency: float
    event_counts: Counter[str]
    metadata: dict[str, str]
    split: str = ""
    trace_length: int = 0
    start_time: str = ""
    end_time: str = ""


def resolve_config(config_name_or_path: str) -> Path:
    raw = Path(config_name_or_path)
    if raw.suffix in {".yaml", ".yml"}:
        return raw if raw.is_absolute() else ROOT / raw
    return CONFIG_DIR / f"{config_name_or_path}.yaml"


def resolve_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


def normalize_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    return text in {"1", "true", "yes", "y"}


def coerce_sort_value(value: str) -> Any:
    text = value.strip()
    if text == "":
        return ""
    if text.lstrip("-").isdigit():
        return int(text)
    try:
        return float(text)
    except ValueError:
        return text


def build_sort_key(row: dict[str, str], sort_columns: list[str]) -> tuple[Any, ...]:
    return tuple(coerce_sort_value(row[column]) for column in sort_columns)


def parse_timestamp(row: dict[str, str], cfg: dict[str, Any]) -> float | None:
    parser_name = str(cfg.get("timestamp_parser", "none"))
    if parser_name == "none":
        return None
    if parser_name == "hdfs_datetime":
        date_col, time_col = cfg["timestamp_columns"]
        try:
            return datetime.strptime(
                row[date_col].strip() + row[time_col].strip(),
                "%y%m%d%H%M%S",
            ).timestamp()
        except ValueError:
            return None
    if parser_name == "iso8601":
        ts_col = cfg["timestamp_columns"][0]
        try:
            return datetime.fromisoformat(row[ts_col].strip()).timestamp()
        except ValueError:
            return None
    if parser_name == "unix":
        ts_col = cfg["timestamp_columns"][0]
        try:
            return float(row[ts_col].strip())
        except ValueError:
            return None
    raise ValueError(f"Unsupported timestamp_parser: {parser_name}")


def build_timestamp_display(row: dict[str, str], cfg: dict[str, Any]) -> str:
    timestamp_columns = list(cfg.get("timestamp_display_columns", cfg.get("timestamp_columns", [])))
    if not timestamp_columns:
        return ""
    return " ".join(row[column].strip() for column in timestamp_columns).strip()


def should_count_label_vote(row: dict[str, str], cfg: dict[str, Any]) -> bool:
    exclude_column = cfg.get("label_vote_exclude_column")
    if not exclude_column:
        return True

    raw_value = row[str(exclude_column)]
    excluded_values = cfg.get("label_vote_exclude_values", [])
    for excluded in excluded_values:
        if isinstance(excluded, bool):
            if normalize_bool(raw_value) == excluded:
                return False
        elif str(raw_value) == str(excluded):
            return False
    return True


def choose_trace_label(accumulator: TraceAccumulator) -> str:
    if accumulator.label_votes:
        max_count = max(accumulator.label_votes.values())
        winners = sorted(label for label, count in accumulator.label_votes.items() if count == max_count)
        return winners[0]
    return accumulator.fallback_label


def finalize_trace(sample_id: str, accumulator: TraceAccumulator) -> PreparedTrace:
    ordered_events = sorted(accumulator.events, key=lambda event: event.sort_key)
    tokens = [event.token for event in ordered_events]
    timestamps = [event.timestamp for event in ordered_events]
    start_time = ordered_events[0].timestamp_display if ordered_events else ""
    end_time = ordered_events[-1].timestamp_display if ordered_events else ""

    valid_times = [timestamp for timestamp in timestamps if timestamp is not None]

    latency = 0.0
    if valid_times and len(valid_times) == len(timestamps):
        latency = float(max(0.0, valid_times[-1] - valid_times[0]))

    metadata: dict[str, str] = {}
    if ordered_events:
        metadata_keys = sorted({key for event in ordered_events for key in event.metadata_values})
        for key in metadata_keys:
            counts = Counter(event.metadata_values.get(key, "") for event in ordered_events if event.metadata_values.get(key, ""))
            metadata[f"dominant_{key.lower()}"] = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0][0] if counts else ""

    return PreparedTrace(
        sample_id=sample_id,
        label=choose_trace_label(accumulator),
        tokens=tokens,
        trace_length=len(tokens),
        start_time=start_time,
        end_time=end_time,
        latency=latency,
        event_counts=Counter(tokens),
        metadata=metadata,
    )


def sanitize_sample_id_part(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_.:-]+", "_", value.strip())
    return text or "missing"


def choose_window_label(events: list[EventRecord], *, anomaly_label: str, normal_label: str) -> str:
    labels = [event.label for event in events if event.label]
    if any(label == anomaly_label for label in labels):
        return anomaly_label
    if labels:
        max_count = max(Counter(labels).values())
        winners = sorted(label for label, count in Counter(labels).items() if count == max_count)
        return winners[0]
    return normal_label


def finalize_window_trace(
    *,
    sample_id: str,
    events: list[EventRecord],
    anomaly_label: str,
    normal_label: str,
    extra_metadata: dict[str, str] | None = None,
) -> PreparedTrace:
    tokens = [event.token for event in events]
    timestamps = [event.timestamp for event in events]
    start_time = events[0].timestamp_display if events else ""
    end_time = events[-1].timestamp_display if events else ""
    valid_times = [timestamp for timestamp in timestamps if timestamp is not None]
    latency = 0.0
    if valid_times and len(valid_times) == len(timestamps):
        latency = float(max(0.0, valid_times[-1] - valid_times[0]))

    metadata: dict[str, str] = dict(extra_metadata or {})
    if events:
        metadata_keys = sorted({key for event in events for key in event.metadata_values})
        for key in metadata_keys:
            counts = Counter(event.metadata_values.get(key, "") for event in events if event.metadata_values.get(key, ""))
            metadata[f"dominant_{key.lower()}"] = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0][0] if counts else ""

    return PreparedTrace(
        sample_id=sample_id,
        label=choose_window_label(events, anomaly_label=anomaly_label, normal_label=normal_label),
        tokens=tokens,
        trace_length=len(tokens),
        start_time=start_time,
        end_time=end_time,
        latency=latency,
        event_counts=Counter(tokens),
        metadata=metadata,
    )


def load_traces(
    cfg: dict[str, Any],
    *,
    finalize: bool = True,
) -> tuple[dict[str, PreparedTrace] | dict[str, TraceAccumulator], dict[str, Any]]:
    input_path = resolve_path(cfg.get("input_path", cfg.get("input_csv")))
    group_key = str(cfg["group_key"])
    token_column = str(cfg["token_column"])
    label_column = str(cfg["label_column"])
    sort_columns = list(cfg["sort_columns"])
    metadata_columns = list(cfg.get("metadata_mode_columns", []))

    traces: dict[str, TraceAccumulator] = defaultdict(TraceAccumulator)
    total_rows = 0
    missing_group_key_rows = 0
    missing_token_rows = 0

    def process_row(row: dict[str, str]) -> None:
        nonlocal total_rows, missing_group_key_rows, missing_token_rows
        total_rows += 1
        sample_id = row[group_key].strip()
        token = row[token_column].strip()
        if not sample_id:
            missing_group_key_rows += 1
            return
        if not token:
            missing_token_rows += 1
            return

        traces[sample_id].add_event(
            sort_key=build_sort_key(row, sort_columns),
            token=token,
            timestamp=parse_timestamp(row, cfg),
            timestamp_display=build_timestamp_display(row, cfg),
            label=row[label_column].strip(),
            count_label_vote=should_count_label_vote(row, cfg),
            metadata_values={column: row[column].strip() for column in metadata_columns},
        )

    if input_path.suffix.lower() == ".parquet":
        try:
            import pandas as pd
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError("Reading parquet inputs requires pandas.") from exc
        df = pd.read_parquet(input_path)
        required_columns = {group_key, token_column, label_column, *sort_columns, *metadata_columns}
        required_columns.update(cfg.get("timestamp_columns", []))
        exclude_column = cfg.get("label_vote_exclude_column")
        if exclude_column:
            required_columns.add(str(exclude_column))
        missing = required_columns.difference(df.columns.tolist())
        if missing:
            raise ValueError(f"Input parquet is missing required columns: {', '.join(sorted(missing))}")
        for row in df.fillna("").to_dict(orient="records"):
            process_row({key: str(value) for key, value in row.items()})
    else:
        with input_path.open("r", newline="", encoding="utf-8", errors="replace") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise ValueError("Input CSV is empty.")

            required_columns = {group_key, token_column, label_column, *sort_columns, *metadata_columns}
            timestamp_columns = cfg.get("timestamp_columns", [])
            required_columns.update(timestamp_columns)
            exclude_column = cfg.get("label_vote_exclude_column")
            if exclude_column:
                required_columns.add(str(exclude_column))

            missing = required_columns.difference(reader.fieldnames)
            if missing:
                raise ValueError(f"Input CSV is missing required columns: {', '.join(sorted(missing))}")

            for row in reader:
                process_row(row)

    stats = {
        "input_path": str(input_path),
        "total_rows": total_rows,
        "missing_group_key_rows": missing_group_key_rows,
        "missing_token_rows": missing_token_rows,
    }
    if not finalize:
        return dict(traces), stats

    prepared = {
        sample_id: finalize_trace(sample_id, accumulator)
        for sample_id, accumulator in traces.items()
    }
    return prepared, stats


def assign_splits(
    traces: dict[str, PreparedTrace],
    *,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> dict[str, str]:
    if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 1e-9:
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")

    rng = Random(seed)
    by_label: dict[str, list[str]] = defaultdict(list)
    for sample_id, trace in traces.items():
        by_label[trace.label].append(sample_id)

    split_map: dict[str, str] = {}
    for label, sample_ids in by_label.items():
        sample_ids = sorted(sample_ids)
        rng.shuffle(sample_ids)
        n_samples = len(sample_ids)
        n_train = int(n_samples * train_ratio)
        n_val = int(n_samples * val_ratio)
        n_test = n_samples - n_train - n_val

        train_ids = sample_ids[:n_train]
        val_ids = sample_ids[n_train : n_train + n_val]
        test_ids = sample_ids[n_train + n_val : n_train + n_val + n_test]

        for sample_id in train_ids:
            split_map[sample_id] = "train"
        for sample_id in val_ids:
            split_map[sample_id] = "val"
        for sample_id in test_ids:
            split_map[sample_id] = "test"

    return split_map


def attach_splits(traces: dict[str, PreparedTrace], split_map: dict[str, str]) -> None:
    for sample_id, trace in traces.items():
        trace.split = split_map[sample_id]


def build_traces_rows(traces: dict[str, PreparedTrace]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for sample_id in sorted(traces):
        trace = traces[sample_id]
        rows.append(
            {
                "sample_id": trace.sample_id,
                "label": trace.label,
                "split": trace.split,
                "sequence": " ".join(trace.tokens),
                "trace_length": trace.trace_length,
            }
        )
    return rows


def build_occurrence_rows(traces: dict[str, PreparedTrace], tokens: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for sample_id in sorted(traces):
        trace = traces[sample_id]
        row: dict[str, Any] = {
            "sample_id": trace.sample_id,
            "label": trace.label,
            "split": trace.split,
        }
        for token in tokens:
            row[token] = trace.event_counts[token]
        rows.append(row)
    return rows


def build_metadata_rows(traces: dict[str, PreparedTrace]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for sample_id in sorted(traces):
        trace = traces[sample_id]
        row: dict[str, Any] = {
            "sample_id": trace.sample_id,
            "label": trace.label,
            "split": trace.split,
            "start_time": trace.start_time,
            "end_time": trace.end_time,
            "latency": trace.latency,
            "trace_length": trace.trace_length,
        }
        row.update(trace.metadata)
        rows.append(row)
    return rows


def build_splits_rows(traces: dict[str, PreparedTrace]) -> list[dict[str, Any]]:
    return [{"sample_id": traces[sample_id].sample_id, "split": traces[sample_id].split} for sample_id in sorted(traces)]


def write_table(
    path: Path,
    rows: list[dict[str, Any]],
    *,
    output_format: str,
    columns: list[str] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if output_format == "csv":
        if columns is None:
            columns = list(rows[0].keys()) if rows else []
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
        return
    if output_format == "parquet":
        try:
            import pandas as pd
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError("Writing parquet requires pandas.") from exc
        try:
            pd.DataFrame(rows, columns=columns).to_parquet(path, index=False)
        except ImportError as exc:
            raise ImportError(
                "Writing parquet requires a parquet engine such as pyarrow or fastparquet."
            ) from exc
        return
    raise ValueError(f"Unsupported output_format: {output_format}")


def write_split_cache(
    cache_dir: Path,
    *,
    traces_rows: list[dict[str, Any]],
    occurrence_rows: list[dict[str, Any]],
    output_format: str,
) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    trace_columns = list(traces_rows[0].keys()) if traces_rows else ["sample_id", "label", "split", "sequence", "trace_length"]
    occurrence_columns = list(occurrence_rows[0].keys()) if occurrence_rows else ["sample_id", "label", "split"]
    for split_name in ("train", "val", "test"):
        split_traces = [row for row in traces_rows if row["split"] == split_name]
        split_occurrence = [row for row in occurrence_rows if row["split"] == split_name]
        write_table(
            cache_dir / f"traces_{split_name}.{output_format}",
            split_traces,
            output_format=output_format,
            columns=trace_columns,
        )
        write_table(
            cache_dir / f"occurrence_{split_name}.{output_format}",
            split_occurrence,
            output_format=output_format,
            columns=occurrence_columns,
        )


def summarize_dataset(
    cfg: dict[str, Any],
    traces: dict[str, PreparedTrace],
    split_map: dict[str, str],
    load_stats: dict[str, Any],
) -> dict[str, Any]:
    label_counts = Counter(trace.label for trace in traces.values())
    split_counts = Counter(split_map.values())
    trace_lengths = [len(trace.tokens) for trace in traces.values()]
    latencies = [trace.latency for trace in traces.values()]
    token_vocab = sorted({token for trace in traces.values() for token in trace.event_counts}, key=str)

    return {
        "dataset_kind": cfg["dataset_kind"],
        "dataset_name": cfg["dataset_name"],
        "input_path": load_stats["input_path"],
        "output_dir": str(resolve_path(cfg["output_dir"])),
        "output_format": cfg["output_format"],
        "write_split_cache": bool(cfg.get("write_split_cache", False)),
        "group_key": cfg["group_key"],
        "token_column": cfg["token_column"],
        "label_column": cfg["label_column"],
        "sort_columns": cfg["sort_columns"],
        "total_rows": load_stats["total_rows"],
        "missing_group_key_rows": load_stats["missing_group_key_rows"],
        "missing_token_rows": load_stats["missing_token_rows"],
        "n_traces": len(traces),
        "label_counts": dict(sorted(label_counts.items())),
        "split_counts": dict(sorted(split_counts.items())),
        "token_vocab_size": len(token_vocab),
        "trace_length": {
            "mean": sum(trace_lengths) / len(trace_lengths) if trace_lengths else 0.0,
            "max": max(trace_lengths) if trace_lengths else 0,
        },
        "latency": {
            "mean": sum(latencies) / len(latencies) if latencies else 0.0,
            "max": max(latencies) if latencies else 0.0,
        },
    }


def prepare_hdfs(cfg: dict[str, Any]) -> None:
    traces, load_stats = load_traces(cfg)
    split_map = assign_splits(
        traces,
        train_ratio=float(cfg["train_ratio"]),
        val_ratio=float(cfg["val_ratio"]),
        test_ratio=float(cfg["test_ratio"]),
        seed=int(cfg["seed"]),
    )
    attach_splits(traces, split_map)

    output_dir = resolve_path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    token_vocab = sorted({token for trace in traces.values() for token in trace.event_counts}, key=str)
    output_format = str(cfg.get("output_format", "csv")).lower()
    traces_rows = build_traces_rows(traces)
    occurrence_rows = build_occurrence_rows(traces, token_vocab)
    metadata_rows = build_metadata_rows(traces)
    splits_rows = build_splits_rows(traces)

    write_table(output_dir / f"traces.{output_format}", traces_rows, output_format=output_format)
    write_table(
        output_dir / f"occurrence_matrix.{output_format}",
        occurrence_rows,
        output_format=output_format,
    )
    write_table(output_dir / f"metadata.{output_format}", metadata_rows, output_format=output_format)
    write_table(output_dir / f"splits.{output_format}", splits_rows, output_format=output_format)
    if bool(cfg.get("write_split_cache", False)):
        write_split_cache(
            output_dir / "cache",
            traces_rows=traces_rows,
            occurrence_rows=occurrence_rows,
            output_format=output_format,
        )
    summary = summarize_dataset(cfg, traces, split_map, load_stats)
    save_json(summary, output_dir / "summary.json")

    print(f"Prepared {summary['dataset_name']}")
    print(f"Traces: {summary['n_traces']}")
    print(f"Rows consumed: {summary['total_rows']}")
    print(f"Output: {output_dir}")


def prepare_synthetic(cfg: dict[str, Any]) -> None:
    traces, load_stats = load_traces(cfg)
    split_map = assign_splits(
        traces,
        train_ratio=float(cfg["train_ratio"]),
        val_ratio=float(cfg["val_ratio"]),
        test_ratio=float(cfg["test_ratio"]),
        seed=int(cfg["seed"]),
    )
    attach_splits(traces, split_map)

    output_dir = resolve_path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    token_vocab = sorted({token for trace in traces.values() for token in trace.event_counts}, key=str)
    output_format = str(cfg.get("output_format", "csv")).lower()
    traces_rows = build_traces_rows(traces)
    occurrence_rows = build_occurrence_rows(traces, token_vocab)
    metadata_rows = build_metadata_rows(traces)
    splits_rows = build_splits_rows(traces)

    write_table(output_dir / f"traces.{output_format}", traces_rows, output_format=output_format)
    write_table(
        output_dir / f"occurrence_matrix.{output_format}",
        occurrence_rows,
        output_format=output_format,
    )
    write_table(output_dir / f"metadata.{output_format}", metadata_rows, output_format=output_format)
    write_table(output_dir / f"splits.{output_format}", splits_rows, output_format=output_format)
    if bool(cfg.get("write_split_cache", False)):
        write_split_cache(
            output_dir / "cache",
            traces_rows=traces_rows,
            occurrence_rows=occurrence_rows,
            output_format=output_format,
        )
    summary = summarize_dataset(cfg, traces, split_map, load_stats)
    save_json(summary, output_dir / "summary.json")

    print(f"Prepared {summary['dataset_name']}")
    print(f"Traces: {summary['n_traces']}")
    print(f"Rows consumed: {summary['total_rows']}")
    print(f"Output: {output_dir}")


def prepare_bgl(cfg: dict[str, Any]) -> None:
    node_traces, load_stats = load_traces(cfg, finalize=False)
    window_size = int(cfg["window_size"])
    stride = int(cfg["stride"])
    min_window_size = int(cfg.get("min_window_size", window_size))
    drop_incomplete = bool(cfg.get("drop_incomplete", False))
    anomaly_label = str(cfg.get("anomaly_label", "Anomaly"))
    normal_label = str(cfg.get("normal_label", "Normal"))
    if window_size <= 0 or stride <= 0 or min_window_size <= 0:
        raise ValueError("BGL window_size, stride, and min_window_size must be positive.")
    if min_window_size > window_size:
        raise ValueError("BGL min_window_size cannot exceed window_size.")

    traces: dict[str, PreparedTrace] = {}
    skipped_short_nodes = 0
    for node in sorted(node_traces):
        ordered_events = sorted(node_traces[node].events, key=lambda event: event.sort_key)
        if len(ordered_events) < min_window_size:
            skipped_short_nodes += 1
            continue

        local_window_index = 0
        for start in range(0, len(ordered_events), stride):
            end = start + window_size
            if end > len(ordered_events) and drop_incomplete:
                break
            window_events = ordered_events[start:min(end, len(ordered_events))]
            if len(window_events) < min_window_size:
                break

            sample_id = f"{sanitize_sample_id_part(node)}__w{local_window_index:06d}"
            traces[sample_id] = finalize_window_trace(
                sample_id=sample_id,
                events=window_events,
                anomaly_label=anomaly_label,
                normal_label=normal_label,
                extra_metadata={
                    "node": node,
                    "window_index": str(local_window_index),
                },
            )
            local_window_index += 1
            if end >= len(ordered_events):
                break

    split_map = assign_splits(
        traces,
        train_ratio=float(cfg["train_ratio"]),
        val_ratio=float(cfg["val_ratio"]),
        test_ratio=float(cfg["test_ratio"]),
        seed=int(cfg["seed"]),
    )
    attach_splits(traces, split_map)

    output_dir = resolve_path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    token_vocab = sorted({token for trace in traces.values() for token in trace.event_counts}, key=str)
    output_format = str(cfg.get("output_format", "csv")).lower()
    traces_rows = build_traces_rows(traces)
    occurrence_rows = build_occurrence_rows(traces, token_vocab)
    metadata_rows = build_metadata_rows(traces)
    splits_rows = build_splits_rows(traces)

    write_table(output_dir / f"traces.{output_format}", traces_rows, output_format=output_format)
    write_table(
        output_dir / f"occurrence_matrix.{output_format}",
        occurrence_rows,
        output_format=output_format,
    )
    write_table(output_dir / f"metadata.{output_format}", metadata_rows, output_format=output_format)
    write_table(output_dir / f"splits.{output_format}", splits_rows, output_format=output_format)
    if bool(cfg.get("write_split_cache", False)):
        write_split_cache(
            output_dir / "cache",
            traces_rows=traces_rows,
            occurrence_rows=occurrence_rows,
            output_format=output_format,
        )

    summary = summarize_dataset(cfg, traces, split_map, load_stats)
    summary.update(
        {
            "sample_mode": "node_sliding_window",
            "window_size": window_size,
            "stride": stride,
            "min_window_size": min_window_size,
            "drop_incomplete": drop_incomplete,
            "source_nodes": len(node_traces),
            "skipped_short_nodes": skipped_short_nodes,
        }
    )
    save_json(summary, output_dir / "summary.json")

    print(f"Prepared {summary['dataset_name']}")
    print(f"Windows: {summary['n_traces']}")
    print(f"Rows consumed: {summary['total_rows']}")
    print(f"Source nodes: {summary['source_nodes']}")
    print(f"Output: {output_dir}")


def main() -> None:
    config_name = sys.argv[1] if len(sys.argv) > 1 else "hdfs_small"
    cfg = load_yaml(resolve_config(config_name))
    dataset_kind = str(cfg["dataset_kind"]).lower()

    if dataset_kind == "hdfs":
        prepare_hdfs(cfg)
        return
    if dataset_kind == "synthetic":
        prepare_synthetic(cfg)
        return
    if dataset_kind == "bgl":
        prepare_bgl(cfg)
        return
    raise ValueError(f"Unsupported dataset_kind: {cfg['dataset_kind']}")


if __name__ == "__main__":
    main()
