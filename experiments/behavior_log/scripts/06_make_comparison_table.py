#!/usr/bin/env python3
"""Stage 06: make a compact comparison table from HDFS evaluation metrics."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from behavior_log.utils.io import load_yaml, save_json


DEFAULT_METRICS = [
    "linear_probe_test_macro_f1",
    "linear_probe_test_auroc",
    "linear_probe_test_auprc",
    "linear_probe_test_anomaly_recall",
    "linear_probe_test_anomaly_precision",
    "retrieval_overall_p_at_5_test",
    "retrieval_anomaly_p_at_5_test",
    "retrieval_normal_p_at_5_test",
    "retrieval_overall_p_at_10_test",
    "retrieval_anomaly_p_at_10_test",
    "retrieval_normal_p_at_10_test",
    "structure_event_count_cosine_at_5_test",
    "structure_ngram_cosine_at_5_test",
]

DEFAULT_COLUMN_LABELS = {
    "method": "Method",
    "embedding_dim": "Dim",
    "linear_probe_test_macro_f1": "Macro-F1",
    "linear_probe_test_auroc": "AUROC",
    "linear_probe_test_auprc": "AUPRC",
    "linear_probe_test_anomaly_recall": "Anom. Recall",
    "linear_probe_test_anomaly_precision": "Anom. Precision",
    "retrieval_overall_p_at_5_test": "Ret. P@5",
    "retrieval_anomaly_p_at_5_test": "Anom. Ret. P@5",
    "retrieval_normal_p_at_5_test": "Normal Ret. P@5",
    "retrieval_overall_p_at_10_test": "Ret. P@10",
    "retrieval_anomaly_p_at_10_test": "Anom. Ret. P@10",
    "retrieval_normal_p_at_10_test": "Normal Ret. P@10",
    "structure_event_count_cosine_at_5_test": "Count Cos@5",
    "structure_ngram_cosine_at_5_test": "N-gram Cos@5",
}


def _resolve_config(config_name_or_path: str) -> Path:
    raw = Path(config_name_or_path)
    if raw.suffix in {".yaml", ".yml"}:
        return raw if raw.is_absolute() else ROOT / raw
    return ROOT / "configs" / "comparison" / f"{config_name_or_path}.yaml"


def _resolve_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT.parents[1] / path


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _format_metric(value: Any, *, digits: int) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return ""
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _build_rows(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    metrics = list(cfg.get("metrics", DEFAULT_METRICS))
    rows: list[dict[str, Any]] = []
    for method_cfg in cfg["methods"]:
        metrics_path = _resolve_path(method_cfg["metrics_path"])
        if not metrics_path.exists():
            if bool(cfg.get("allow_missing", False)):
                print(f"Skipping missing metrics file: {metrics_path}")
                continue
            raise FileNotFoundError(f"Missing metrics file: {metrics_path}")

        payload = _load_json(metrics_path)
        row: dict[str, Any] = {
            "method": method_cfg.get("display_name", payload.get("method_name", method_cfg.get("name"))),
            "method_key": method_cfg.get("name", payload.get("method_name")),
            "dataset_name": payload.get("dataset_name", cfg.get("dataset_name")),
            "n_samples": payload.get("n_samples"),
            "embedding_dim": payload.get("embedding_dim"),
            "metrics_path": str(metrics_path),
        }
        for metric in metrics:
            row[metric] = payload.get(metric)
        rows.append(row)
    return rows


def _rank_rows(df: pd.DataFrame, primary_metric: str) -> pd.DataFrame:
    if primary_metric not in df.columns:
        return df
    ranked = df.copy()
    ranked["_primary_sort"] = pd.to_numeric(ranked[primary_metric], errors="coerce")
    ranked = ranked.sort_values(["_primary_sort", "method"], ascending=[False, True]).drop(columns=["_primary_sort"])
    return ranked.reset_index(drop=True)


def _write_markdown(
    *,
    df: pd.DataFrame,
    path: Path,
    title: str,
    metrics: list[str],
    digits: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    display_columns = ["method", "embedding_dim", *metrics]
    table_df = df[display_columns].copy()
    table_df = table_df.rename(columns=DEFAULT_COLUMN_LABELS)
    for column in table_df.columns:
        if column == "Method":
            continue
        table_df[column] = table_df[column].map(lambda value: _format_metric(value, digits=digits))

    lines = [f"# {title}", ""]
    if len(table_df) == 0:
        lines.append("_No metrics were available._")
    else:
        lines.append(table_df.to_markdown(index=False))
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    config_name = sys.argv[1] if len(sys.argv) > 1 else "hdfs_small"
    cfg = load_yaml(_resolve_config(config_name))
    output_dir = _resolve_path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics = list(cfg.get("metrics", DEFAULT_METRICS))
    rows = _build_rows(cfg)
    df = pd.DataFrame(rows)
    primary_metric = str(cfg.get("primary_metric", "linear_probe_test_macro_f1"))
    df = _rank_rows(df, primary_metric)

    csv_path = output_dir / cfg.get("csv_file", "comparison_table.csv")
    md_path = output_dir / cfg.get("markdown_file", "comparison_table.md")
    summary_path = output_dir / cfg.get("summary_file", "comparison_summary.json")

    df.to_csv(csv_path, index=False)
    _write_markdown(
        df=df,
        path=md_path,
        title=str(cfg.get("title", "HDFS Embedding Comparison")),
        metrics=metrics,
        digits=int(cfg.get("digits", 4)),
    )

    summary = {
        "dataset_name": cfg.get("dataset_name"),
        "n_methods": int(len(df)),
        "primary_metric": primary_metric,
        "best_method": None if len(df) == 0 else str(df.iloc[0]["method"]),
        "best_primary_metric": None if len(df) == 0 else df.iloc[0].get(primary_metric),
        "csv_file": str(csv_path),
        "markdown_file": str(md_path),
        "methods": df["method"].astype(str).tolist() if "method" in df else [],
        "metrics": metrics,
    }
    save_json(summary, summary_path)

    print(f"Comparison methods: {len(df)}")
    if len(df) > 0:
        print(f"Best by {primary_metric}: {summary['best_method']} = {summary['best_primary_metric']}")
    print(f"Saved CSV: {csv_path}")
    print(f"Saved Markdown: {md_path}")


if __name__ == "__main__":
    main()
