#!/usr/bin/env python3
"""Stage 05: evaluate frozen HDFS block-level embeddings."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from behavior_log.data.trace_data import load_table
from behavior_log.evaluation.trace_embedding_evaluator import HDFSTraceEmbeddingEvaluator
from behavior_log.utils.io import load_yaml, save_json


def _resolve_config(config_name_or_path: str) -> Path:
    raw = Path(config_name_or_path)
    if raw.suffix in {".yaml", ".yml"}:
        return raw if raw.is_absolute() else ROOT / raw
    return ROOT / "configs" / "evaluation" / f"{config_name_or_path}.yaml"


def _resolve_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT.parents[1] / path


def _load_embeddings(path: Path) -> dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    required = {"embeddings", "sample_id", "label", "split"}
    missing = required.difference(data.files)
    if missing:
        raise ValueError(f"Embedding file is missing arrays: {', '.join(sorted(missing))}")
    return {key: data[key] for key in required}


def _align_prepared_tables(
    *,
    embedding_sample_ids: np.ndarray,
    traces_path: Path,
    occurrence_path: Path,
):
    traces_df = load_table(traces_path)
    occurrence_df = load_table(occurrence_path)
    for frame_name, frame in (("traces", traces_df), ("occurrence_matrix", occurrence_df)):
        if "sample_id" not in frame.columns:
            raise ValueError(f"{frame_name} is missing sample_id column.")
        frame["sample_id"] = frame["sample_id"].astype(str)

    wanted = set(embedding_sample_ids.astype(str))
    missing_traces = wanted.difference(traces_df["sample_id"])
    missing_occurrence = wanted.difference(occurrence_df["sample_id"])
    if missing_traces:
        raise ValueError(f"traces file is missing {len(missing_traces)} embedding sample ids.")
    if missing_occurrence:
        raise ValueError(f"occurrence_matrix file is missing {len(missing_occurrence)} embedding sample ids.")
    return traces_df, occurrence_df


def main() -> None:
    config_name = sys.argv[1] if len(sys.argv) > 1 else "hdfs_small_ts2vec"
    cfg = load_yaml(_resolve_config(config_name))
    if str(cfg.get("dataset_type", "hdfs")).lower() != "hdfs":
        raise NotImplementedError("Stage 05 currently implements HDFS trace-level evaluation.")

    embeddings_path = _resolve_path(cfg["embedding_file"])
    traces_path = _resolve_path(cfg["traces_path"])
    occurrence_path = _resolve_path(cfg["occurrence_matrix_path"])
    output_dir = _resolve_path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    embedding_data = _load_embeddings(embeddings_path)
    traces_df, occurrence_df = _align_prepared_tables(
        embedding_sample_ids=embedding_data["sample_id"].astype(str),
        traces_path=traces_path,
        occurrence_path=occurrence_path,
    )

    evaluator = HDFSTraceEmbeddingEvaluator(
        retrieval_ks=list(cfg.get("retrieval_ks", [5, 10])),
        structure_k=int(cfg.get("structure_k", 5)),
        logistic_max_iter=int(cfg.get("logistic_max_iter", 2000)),
        logistic_class_weight=cfg.get("logistic_class_weight", "balanced"),
        standardize_probe_features=bool(cfg.get("standardize_probe_features", True)),
        ngram_min=int(cfg.get("ngram_min", 1)),
        ngram_max=int(cfg.get("ngram_max", 3)),
    )
    result = evaluator.evaluate(
        embeddings=embedding_data["embeddings"].astype(np.float32),
        sample_id=embedding_data["sample_id"].astype(str),
        labels=embedding_data["label"].astype(str),
        split=embedding_data["split"].astype(str),
        traces_df=traces_df,
        occurrence_df=occurrence_df,
    )

    metrics = {
        "method_name": cfg.get("method_name", config_name),
        "dataset_name": cfg.get("dataset_name"),
        "embedding_file": str(embeddings_path),
        "traces_path": str(traces_path),
        "occurrence_matrix_path": str(occurrence_path),
        **result.metrics,
    }
    metrics_path = output_dir / cfg.get("metrics_file", "metrics.json")
    artifacts_path = output_dir / cfg.get("artifacts_file", "evaluation_artifacts.npz")
    save_json(metrics, metrics_path)
    np.savez_compressed(artifacts_path, **result.artifacts)

    print("HDFS embedding evaluation:")
    print(f"  method: {metrics['method_name']}")
    print(f"  samples: {metrics['n_samples']}")
    print(f"  embedding_dim: {metrics['embedding_dim']}")
    print(f"  test_macro_f1: {metrics['linear_probe_test_macro_f1']:.6f}")
    print(f"  test_auroc: {metrics['linear_probe_test_auroc']}")
    print(f"  retrieval_p@5: {metrics.get('retrieval_overall_p_at_5_test')}")
    print(f"  structure_count_cos@5: {metrics.get('structure_event_count_cosine_at_5_test')}")
    print(f"  saved: {metrics_path}")


if __name__ == "__main__":
    main()
