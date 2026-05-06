#!/usr/bin/env python3
"""Stage 02: compute baseline embeddings from prepared trace datasets."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from behavior_log.baselines.trace_representations import TraceRepresentationBuilder
from behavior_log.utils.io import load_yaml, save_json


def _resolve_config(config_name_or_path: str) -> Path:
    raw = Path(config_name_or_path)
    if raw.suffix in {".yaml", ".yml"}:
        return raw if raw.is_absolute() else ROOT / raw
    return ROOT / "configs" / "baselines" / f"{config_name_or_path}.yaml"


def _resolve_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT.parents[1] / path


def _load_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def main() -> None:
    config_name = sys.argv[1] if len(sys.argv) > 1 else "hdfs_small_occurrence"
    cfg = load_yaml(_resolve_config(config_name))

    traces_df = _load_table(_resolve_path(cfg["traces_path"]))
    occurrence_df = _load_table(_resolve_path(cfg["occurrence_matrix_path"]))
    splits_df = _load_table(_resolve_path(cfg["splits_path"]))

    traces_df["sample_id"] = traces_df["sample_id"].astype(str)
    occurrence_df["sample_id"] = occurrence_df["sample_id"].astype(str)
    splits_df["sample_id"] = splits_df["sample_id"].astype(str)

    merged = splits_df[["sample_id", "split"]].merge(
        traces_df,
        on=["sample_id", "split"],
        how="inner",
    )
    merged = merged.merge(
        occurrence_df,
        on=["sample_id", "split"],
        how="inner",
        suffixes=("", "_occ"),
    )

    if "label_occ" in merged.columns:
        label_mismatch = merged["label"].astype(str) != merged["label_occ"].astype(str)
        if label_mismatch.any():
            raise ValueError("Label mismatch detected between traces and occurrence_matrix.")
        merged = merged.drop(columns=["label_occ"])

    if len(merged) != len(splits_df):
        raise ValueError(
            f"Aligned sample count mismatch: splits={len(splits_df)}, merged={len(merged)}. "
            "Check sample_id/label consistency across prepared files."
        )

    traces_aligned = merged[["sample_id", "label", "split", "sequence", "trace_length"]].copy()
    occurrence_feature_columns = [
        column
        for column in occurrence_df.columns
        if column not in {"sample_id", "label"}
    ]
    occurrence_aligned = merged[["sample_id", "label", *occurrence_feature_columns]].copy()

    builder = TraceRepresentationBuilder(
        method=str(cfg["method"]),
        n_components=int(cfg.get("n_components", 32)),
        whiten=bool(cfg.get("whiten", False)),
        ngram_min=int(cfg.get("ngram_min", 1)),
        ngram_max=int(cfg.get("ngram_max", 3)),
    )
    result = builder.fit_transform(
        traces_df=traces_aligned,
        occurrence_df=occurrence_aligned,
        split=merged["split"].astype(str).to_numpy(),
    )

    output_dir = _resolve_path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    embeddings_path = output_dir / cfg.get("embedding_file", "embeddings.npz")
    state_path = output_dir / cfg.get("state_file", "baseline_state.json")
    summary_path = output_dir / cfg.get("summary_file", "embedding_summary.json")

    np.savez_compressed(
        embeddings_path,
        embeddings=result.embeddings.astype(np.float32),
        sample_id=merged["sample_id"].astype(str).to_numpy(),
        label=merged["label"].astype(str).to_numpy(),
        split=merged["split"].astype(str).to_numpy(),
    )
    builder.save(state_path)

    summary = {
        "dataset_name": cfg.get("dataset_name"),
        "method": cfg["method"],
        "n_samples": int(len(merged)),
        "embedding_dim": int(result.embeddings.shape[1]),
        "split_counts": merged["split"].value_counts().sort_index().to_dict(),
        "label_counts": merged["label"].value_counts().sort_index().to_dict(),
        **result.summary,
    }
    save_json(summary, summary_path)

    print(f"Computed {cfg['method']} embeddings")
    print(f"Samples: {summary['n_samples']}")
    print(f"Embedding dim: {summary['embedding_dim']}")
    print(f"Saved: {embeddings_path}")


if __name__ == "__main__":
    main()
