from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict
import argparse
import pandas as pd


# -----------------------------------------------------------------------------
# Load
# -----------------------------------------------------------------------------

def load_summary(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def round_df(df: pd.DataFrame, n: int = 4) -> pd.DataFrame:
    df = df.copy()
    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]):
            df[c] = df[c].round(n)
    return df


# -----------------------------------------------------------------------------
# Tables
# -----------------------------------------------------------------------------

def table_probes(summary):
    rows = []
    for target, models in summary.get("probes", {}).items():
        for model, m in models.items():
            rows.append({
                "Target": target,
                "Model": model,
                "Accuracy": m.get("accuracy"),
                "Balanced Acc": m.get("balanced_accuracy"),
                "MAE": m.get("mae"),
                "RMSE": m.get("rmse"),
                "R2": m.get("r2"),
            })
    df = pd.DataFrame(rows)
    return round_df(df)


def table_retrieval(summary):
    rows = []
    for target, m in summary.get("retrieval", {}).items():
        if target == "num_samples":
            continue
        rows.append({
            "Target": target,
            "R@1": m.get("recall_at_1"),
            "R@5": m.get("recall_at_5"),
            "R@10": m.get("recall_at_10"),
            "Top1 Match": m.get("top1_match_fraction"),
            "Top5 Match": m.get("top5_match_fraction"),
        })
    return round_df(pd.DataFrame(rows))


def table_clustering(summary):
    rows = []
    for target, m in summary.get("clustering", {}).items():
        rows.append({
            "Target": target,
            "Clusters": m.get("num_clusters"),
            "ARI": m.get("ari"),
            "NMI": m.get("nmi"),
            "Purity": m.get("purity"),
            "Silhouette": m.get("silhouette"),
        })
    return round_df(pd.DataFrame(rows))


def table_transition(summary):
    t = summary.get("transition", {})

    row = {
        "Metric": "Mode Classification",
        "Clean Acc": t.get("mode_probe_linear_clean", {}).get("accuracy"),
        "Transition Acc": t.get("mode_probe_linear_transition", {}).get("accuracy"),
    }

    row2 = {
        "Metric": "Retrieval@5",
        "Clean": t.get("retrieval_mode_clean", {}).get("recall_at_5"),
        "Transition": t.get("retrieval_mode_transition", {}).get("recall_at_5"),
    }

    return round_df(pd.DataFrame([row, row2]))


def table_transition_bucket(summary):
    rows = []
    buckets = summary.get("transition", {}).get("transition_distance_buckets", {})
    for name, info in buckets.items():
        probe = info.get("mode_probe_linear", {})
        rows.append({
            "Distance": name,
            "Samples": info.get("num_samples"),
            "Accuracy": probe.get("accuracy"),
            "Balanced Acc": probe.get("balanced_accuracy"),
        })
    return round_df(pd.DataFrame(rows))

def parse_args():
    parser = argparse.ArgumentParser(description="Export evaluation results to markdown tables")

    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to evaluation_suite_summary.json"
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="md_tables",
        help="Directory to save markdown tables"
    )

    return parser.parse_args()

# -----------------------------------------------------------------------------
# Save Markdown
# -----------------------------------------------------------------------------

def save_md(df: pd.DataFrame, path: Path, title: str):
    content = f"## {title}\n\n"
    content += df.to_markdown(index=False)
    path.write_text(content, encoding="utf-8")


def main():
    args = parse_args()

    json_path = Path(args.input)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = load_summary(json_path)

    tables = {
        "probes": (table_probes(summary), "Probe Results"),
        "retrieval": (table_retrieval(summary), "Retrieval Performance"),
        "clustering": (table_clustering(summary), "Clustering Metrics"),
        "transition": (table_transition(summary), "Transition vs Clean"),
        "transition_bucket": (table_transition_bucket(summary), "Transition Difficulty"),
    }

    for name, (df, title) in tables.items():
        save_md(df, out_dir / f"{name}.md", title)

    report = "# Evaluation Results Summary\n\n"
    for name, (df, title) in tables.items():
        report += f"## {title}\n\n"
        report += df.to_markdown(index=False)
        report += "\n\n"

    (out_dir / "full_report.md").write_text(report, encoding="utf-8")

    print(f"Markdown tables saved to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()