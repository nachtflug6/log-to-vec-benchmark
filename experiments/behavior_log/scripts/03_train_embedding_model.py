#!/usr/bin/env python3
"""Stage 03: train a learned trace-level embedding model from config."""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from behavior_log.data.trace_data import load_table, tokenize_sequence
from behavior_log.training.trace_contrastive import train_hdfs_contrastive_encoder
from behavior_log.training.trace_training_plan import build_hdfs_component_specs
from behavior_log.utils.io import load_yaml, save_json


def _resolve_config(config_name_or_path: str) -> Path:
    raw = Path(config_name_or_path)
    if raw.suffix in {".yaml", ".yml"}:
        return raw if raw.is_absolute() else ROOT / raw
    return ROOT / "configs" / "training" / f"{config_name_or_path}.yaml"


def _resolve_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT.parents[1] / path


def _detect_table_path(base_dir: Path, stem: str) -> Path:
    parquet_path = base_dir / f"{stem}.parquet"
    csv_path = base_dir / f"{stem}.csv"
    if parquet_path.exists():
        return parquet_path
    if csv_path.exists():
        return csv_path
    raise FileNotFoundError(f"Could not find {stem}.parquet or {stem}.csv under {base_dir}")


def _pick_trace_input(cfg: dict) -> tuple[Path, str]:
    dataset_dir = _resolve_path(cfg["dataset_dir"])
    prefer_cache = bool(cfg.get("prefer_split_cache", True))
    split_name = str(cfg.get("cache_split", "train"))
    if prefer_cache:
        cache_dir = dataset_dir / "cache"
        parquet_path = cache_dir / f"traces_{split_name}.parquet"
        csv_path = cache_dir / f"traces_{split_name}.csv"
        if parquet_path.exists():
            return parquet_path, f"cache:{split_name}"
        if csv_path.exists():
            return csv_path, f"cache:{split_name}"

    return _detect_table_path(dataset_dir, "traces"), "full_table"


def _build_token_vocab(sequences: list[list[str]], *, min_token_count: int) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for tokens in sequences:
        counter.update(tokens)

    vocab = {"[PAD]": 0, "[UNK]": 1, "[MASK]": 2}
    for token, count in sorted(counter.items()):
        if count >= min_token_count:
            vocab[token] = len(vocab)
    return vocab


def main() -> None:
    config_name = sys.argv[1] if len(sys.argv) > 1 else "hdfs_small_ts2vec"
    cfg = load_yaml(_resolve_config(config_name))
    dataset_type = str(cfg["dataset_type"]).lower()
    if dataset_type not in {"hdfs", "behavior_subprocess"}:
        raise NotImplementedError(
            "Stage 03 currently implements dataset_type in "
            f"{{'hdfs', 'behavior_subprocess'}}. Got {cfg['dataset_type']!r}."
        )

    traces_path, source_mode = _pick_trace_input(cfg)
    dataset_dir = _resolve_path(cfg["dataset_dir"])
    summary_path = dataset_dir / "summary.json"
    dataset_summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    traces_df = load_table(traces_path)
    required_columns = {"sample_id", "label", "split", "sequence", "trace_length"}
    missing = required_columns.difference(traces_df.columns)
    if missing:
        raise ValueError(f"Trace input is missing required columns: {', '.join(sorted(missing))}")

    if source_mode == "full_table":
        train_df = traces_df[traces_df["split"].astype(str) == "train"].copy()
        val_df = traces_df[traces_df["split"].astype(str) == "val"].copy()
        test_df = traces_df[traces_df["split"].astype(str) == "test"].copy()
    else:
        train_df = traces_df.copy()
        val_df = None
        test_df = None

    train_sequences = [tokenize_sequence(sequence) for sequence in train_df["sequence"].fillna("").astype(str)]
    train_lengths = [len(tokens) for tokens in train_sequences]
    token_vocab = _build_token_vocab(
        train_sequences,
        min_token_count=int(cfg.get("min_token_count", 1)),
    )

    output_dir = _resolve_path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    token_vocab_path = output_dir / cfg.get("token_vocab_file", "token_vocab.json")
    token_stats_path = output_dir / cfg.get("token_stats_file", "token_stats.json")
    training_plan_path = output_dir / cfg.get("training_plan_file", "training_plan.json")
    training_summary_path = output_dir / cfg.get("training_summary_file", "training_summary.json")

    token_stats = {
        "dataset_type": dataset_type,
        "vocab_size": len(token_vocab),
        "special_tokens": ["[PAD]", "[UNK]", "[MASK]"],
        "train_samples": int(len(train_df)),
        "train_label_counts": train_df["label"].astype(str).value_counts().sort_index().to_dict(),
        "train_trace_length": {
            "mean": (sum(train_lengths) / len(train_lengths)) if train_lengths else 0.0,
            "max": max(train_lengths) if train_lengths else 0,
        },
    }
    component_specs = build_hdfs_component_specs(cfg)

    plan = {
        "run_name": cfg["run_name"],
        "status": "not_started",
        "model_type": cfg["model_type"],
        "dataset_type": dataset_type,
        "dataset_name": cfg.get("dataset_name"),
        "sample_mode": cfg["sample_mode"],
        "input_source_mode": source_mode,
        "dataset_dir": str(dataset_dir),
        "input_traces_path": str(traces_path),
        "dataset_summary_path": str(summary_path) if summary_path.exists() else None,
        "output_dir": str(output_dir),
        "dry_run_only": bool(cfg.get("dry_run_only", True)),
        "split_counts": {
            "train": int(len(train_df)),
            "val": int(len(val_df)) if val_df is not None else None,
            "test": int(len(test_df)) if test_df is not None else None,
        },
        "label_counts": traces_df["label"].astype(str).value_counts().sort_index().to_dict(),
        "token_vocab_path": str(token_vocab_path),
        "max_len": int(cfg["max_len"]),
        "input_type": cfg["input_type"],
        "token_column": cfg["token_column"],
        "training_rule": cfg["training_rule"],
        "augmentation": cfg["augmentation"],
        "objective": cfg.get("objective", {"type": "contrastive"}),
        "encoder": cfg["encoder"],
        "optimization": cfg["optimization"],
        "components": component_specs,
        "dataset_summary": {
            "n_traces": dataset_summary.get("n_traces"),
            "token_vocab_size": dataset_summary.get("token_vocab_size"),
            "trace_length": dataset_summary.get("trace_length"),
            "label_counts": dataset_summary.get("label_counts"),
        },
    }

    token_vocab_path.write_text(json.dumps(token_vocab, indent=2), encoding="utf-8")
    save_json(token_stats, token_stats_path)
    save_json(plan, training_plan_path)

    if not bool(cfg.get("dry_run_only", True)):
        result = train_hdfs_contrastive_encoder(
            train_df=train_df,
            token_vocab=token_vocab,
            cfg=cfg,
            output_dir=output_dir,
        )
        plan["status"] = "completed"
        plan["model_path"] = result.checkpoint_path
        plan["training_summary_path"] = str(training_summary_path)
        save_json(plan, training_plan_path)
        save_json(result.training_summary, training_summary_path)
        print(f"Trained model: {result.checkpoint_path}")
        print(f"Training summary: {training_summary_path}")
        return

    print(f"Prepared training plan for {cfg['run_name']}")
    print(f"Input mode: {source_mode}")
    print(f"Train samples: {len(train_df)}")
    print(f"Token vocab size: {len(token_vocab)}")
    print(f"Saved plan: {training_plan_path}")


if __name__ == "__main__":
    main()
