#!/usr/bin/env python3
"""Merge subprocess event/sequence shard outputs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def merge_csv_shards(input_dir: Path, output_path: Path, pattern: str) -> int:
    shard_paths = sorted(input_dir.glob(pattern))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    row_count = 0
    fieldnames: list[str] | None = None

    with output_path.open("w", newline="", encoding="utf-8") as out_fh:
        writer: csv.DictWriter[str] | None = None
        for shard_path in shard_paths:
            with shard_path.open("r", newline="", encoding="utf-8") as in_fh:
                reader = csv.DictReader(in_fh)
                if fieldnames is None:
                    fieldnames = list(reader.fieldnames or [])
                    writer = csv.DictWriter(out_fh, fieldnames=fieldnames)
                    writer.writeheader()
                elif list(reader.fieldnames or []) != fieldnames:
                    raise ValueError(f"CSV header mismatch in {shard_path}")
                assert writer is not None
                for row in reader:
                    writer.writerow(row)
                    row_count += 1
    return row_count


def merge_jsonl_shards(input_dir: Path, output_path: Path, pattern: str) -> int:
    shard_paths = sorted(input_dir.glob(pattern))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    row_count = 0

    with output_path.open("w", encoding="utf-8") as out_fh:
        for shard_path in shard_paths:
            with shard_path.open("r", encoding="utf-8") as in_fh:
                for line in in_fh:
                    if not line.strip():
                        continue
                    json.loads(line)
                    out_fh.write(line)
                    row_count += 1
    return row_count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=Path("/home/lyra/projects/dataset/subprocess_shards"))
    parser.add_argument("--output-dir", type=Path, default=Path("/home/lyra/projects/dataset"))
    parser.add_argument("--events-pattern", default="subprocess_events_clean_shard_*.csv")
    parser.add_argument("--sequences-pattern", default="subprocess_sequences_shard_*.jsonl")
    args = parser.parse_args()

    event_count = merge_csv_shards(
        args.input_dir,
        args.output_dir / "subprocess_events_clean.csv",
        args.events_pattern,
    )
    sequence_count = merge_jsonl_shards(
        args.input_dir,
        args.output_dir / "subprocess_sequences.jsonl",
        args.sequences_pattern,
    )
    print(f"subprocess_events={event_count}")
    print(f"subprocess_sequences={sequence_count}")


if __name__ == "__main__":
    main()
