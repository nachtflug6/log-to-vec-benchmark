#!/usr/bin/env python3
"""Stage 04: extract embeddings from a trained trace-level SSL model."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from behavior_log.data.trace_data import load_table
from behavior_log.training.trace_contrastive import TraceSequenceDataset, build_trace_ssl_model
from behavior_log.utils.io import load_yaml, save_json


def _resolve_config(config_name_or_path: str) -> Path:
    raw = Path(config_name_or_path)
    if raw.suffix in {".yaml", ".yml"}:
        return raw if raw.is_absolute() else ROOT / raw
    return ROOT / "configs" / "extraction" / f"{config_name_or_path}.yaml"


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


def _load_checkpoint(path: Path, device: torch.device) -> dict[str, Any]:
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Checkpoint is not a dictionary: {path}")
    required_keys = {"model_state_dict", "token_vocab", "config"}
    missing = required_keys.difference(checkpoint)
    if missing:
        raise ValueError(f"Checkpoint is missing keys: {', '.join(sorted(missing))}")
    return checkpoint


def _select_device(requested_device: str) -> torch.device:
    if requested_device == "auto":
        requested_device = "cuda" if torch.cuda.is_available() else "cpu"
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA was requested but is not available; falling back to CPU.")
        requested_device = "cpu"
    return torch.device(requested_device)


def _required_trace_columns() -> set[str]:
    return {"sample_id", "label", "split", "sequence", "trace_length"}


def _extract_embeddings(
    *,
    model: torch.nn.Module,
    dataset: TraceSequenceDataset,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    chunks: list[np.ndarray] = []

    model.eval()
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            _, embeddings = model(input_ids, attention_mask)
            chunks.append(embeddings.detach().cpu().numpy().astype(np.float32))

    if not chunks:
        return np.empty((0, 0), dtype=np.float32)
    return np.concatenate(chunks, axis=0)


def main() -> None:
    config_name = sys.argv[1] if len(sys.argv) > 1 else "hdfs_small_ts2vec"
    cfg = load_yaml(_resolve_config(config_name))

    device = _select_device(str(cfg.get("device", "auto")))
    checkpoint_path = _resolve_path(cfg["checkpoint_path"])
    checkpoint = _load_checkpoint(checkpoint_path, device)
    training_cfg = checkpoint["config"]
    token_vocab = {str(token): int(index) for token, index in checkpoint["token_vocab"].items()}

    traces_path = _resolve_path(cfg["traces_path"]) if cfg.get("traces_path") else _detect_table_path(
        _resolve_path(cfg["dataset_dir"]),
        "traces",
    )
    traces_df = load_table(traces_path)
    missing = _required_trace_columns().difference(traces_df.columns)
    if missing:
        raise ValueError(f"Trace input is missing required columns: {', '.join(sorted(missing))}")

    split_filter = cfg.get("split")
    if split_filter and str(split_filter).lower() != "all":
        traces_df = traces_df[traces_df["split"].astype(str) == str(split_filter)].copy()

    traces_df = traces_df.reset_index(drop=True)
    max_len = int(cfg.get("max_len", training_cfg["max_len"]))
    dataset = TraceSequenceDataset(traces_df, token_vocab=token_vocab, max_len=max_len)

    model = build_trace_ssl_model(
        vocab_size=len(token_vocab),
        token_vocab=token_vocab,
        encoder_cfg=training_cfg["encoder"],
        max_len=max_len,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    embeddings = _extract_embeddings(
        model=model,
        dataset=dataset,
        batch_size=int(cfg.get("batch_size", 256)),
        device=device,
    )

    if bool(cfg.get("l2_normalize", False)) and len(embeddings) > 0:
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / np.maximum(norms, 1e-12)

    output_dir = _resolve_path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    embeddings_path = output_dir / cfg.get("embedding_file", "embeddings.npz")
    summary_path = output_dir / cfg.get("summary_file", "embedding_summary.json")

    np.savez_compressed(
        embeddings_path,
        embeddings=embeddings.astype(np.float32),
        sample_id=traces_df["sample_id"].astype(str).to_numpy(),
        label=traces_df["label"].astype(str).to_numpy(),
        split=traces_df["split"].astype(str).to_numpy(),
    )

    summary = {
        "status": "completed",
        "dataset_name": cfg.get("dataset_name", training_cfg.get("dataset_name")),
        "run_name": cfg.get("run_name", training_cfg.get("run_name")),
        "model_type": training_cfg.get("model_type"),
        "objective_type": checkpoint.get("objective_type", training_cfg.get("objective", {}).get("type")),
        "encoder_architecture": checkpoint.get("encoder_architecture", training_cfg.get("encoder", {}).get("architecture")),
        "checkpoint_path": str(checkpoint_path),
        "traces_path": str(traces_path),
        "embedding_file": str(embeddings_path),
        "device": str(device),
        "n_samples": int(len(traces_df)),
        "embedding_dim": int(embeddings.shape[1]) if embeddings.ndim == 2 and len(embeddings) > 0 else 0,
        "split_counts": traces_df["split"].astype(str).value_counts().sort_index().to_dict(),
        "label_counts": traces_df["label"].astype(str).value_counts().sort_index().to_dict(),
        "max_len": max_len,
        "l2_normalize": bool(cfg.get("l2_normalize", False)),
    }
    save_json(summary, summary_path)

    print(f"Extracted embeddings: {summary['n_samples']}")
    print(f"Embedding dim: {summary['embedding_dim']}")
    print(f"Saved: {embeddings_path}")


if __name__ == "__main__":
    main()
