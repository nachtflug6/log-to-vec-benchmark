"""Step 6: Aggregate all metrics JSONs into a summary table and report.md.

Usage:
  python 06_build_report.py --metrics_dir metrics --plots_dir plots --output_dir reports
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

PROBLEMS = ["p1_simple_1d", "p2_multichannel", "p3_hard_noisy"]
MODELS = ["fft", "moment", "ts2vec"]

METRIC_DISPLAY = {
    "mode_separability_index": ("MSI", "↑"),
    "loop_consistency_mean": ("Loop Consistency (DTW)", "↓"),
    "transition_sharpness": ("Transition Sharpness (windows)", "↓"),
    "pca_loop_compactness_mean": ("PCA Compactness", "↓"),
    "centroid_stability_mean": ("Centroid Stability", "↓"),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--metrics_dir", type=str, default="metrics")
    p.add_argument("--plots_dir", type=str, default="plots")
    p.add_argument("--output_dir", type=str, default="reports")
    return p.parse_args()


def _fmt(val: Any, decimals: int = 3) -> str:
    if val is None:
        return "—"
    try:
        f = float(val)
        if f != f:  # nan
            return "—"
        if f == float("inf"):
            return "∞"
        return f"{f:.{decimals}f}"
    except (TypeError, ValueError):
        return str(val)


def _load_metrics(metrics_dir: Path) -> Dict[str, dict]:
    results = {}
    for problem in PROBLEMS:
        for model in MODELS:
            path = metrics_dir / f"{problem}_{model}.json"
            if path.exists():
                with open(path) as f:
                    results[f"{problem}/{model}"] = json.load(f)
    return results


def _best_per_col(rows: List[dict], metric_key: str, direction: str) -> Optional[str]:
    """Return the cell key with best value for this metric."""
    best_key = None
    best_val = None
    for key, row in rows:
        v = row.get(metric_key)
        if v is None:
            continue
        try:
            v = float(v)
        except (TypeError, ValueError):
            continue
        if v != v or v == float("inf"):
            continue
        if best_val is None:
            best_val, best_key = v, key
        elif direction == "↑" and v > best_val:
            best_val, best_key = v, key
        elif direction == "↓" and v < best_val:
            best_val, best_key = v, key
    return best_key


def _build_table(results: Dict[str, dict], problem: str) -> str:
    header = "| Model | " + " | ".join(f"{name} {arrow}" for name, arrow in METRIC_DISPLAY.values()) + " |"
    sep = "| --- |" + " --- |" * len(METRIC_DISPLAY)

    rows_for_problem = [
        (model, results.get(f"{problem}/{model}", {}))
        for model in MODELS
    ]

    best_per_metric = {
        mkey: _best_per_col(
            [(model, r) for model, r in rows_for_problem],
            mkey,
            arrow,
        )
        for mkey, (_, arrow) in METRIC_DISPLAY.items()
    }

    lines = [header, sep]
    for model, row in rows_for_problem:
        cells = [f"**{model}**"]
        for mkey, (_, arrow) in METRIC_DISPLAY.items():
            val = _fmt(row.get(mkey))
            is_best = best_per_metric.get(mkey) == model and val != "—"
            cells.append(f"**{val}**" if is_best else val)
        lines.append("| " + " | ".join(cells) + " |")

    return "\n".join(lines)


def _plot_ref(plots_dir: Path, problem: str, model: str, plot_type: str) -> str:
    rel = plots_dir / problem / model / f"{problem}_{model}_{plot_type}.png"
    if rel.exists():
        return f"![{plot_type}]({rel})"
    return f"*(not generated)*"


def main() -> None:
    args = parse_args()
    metrics_dir = Path(args.metrics_dir)
    plots_dir = Path(args.plots_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = _load_metrics(metrics_dir)
    if not results:
        print("[warn] No metrics found. Run 04_compute_trace_metrics.py first.")

    lines = [
        "# RQ2 — Trace Comparison in Embedding Space: Report",
        "",
        "## Overview",
        "",
        "Three synthetic periodic-mode datasets (CNC-analogy: parts A, B, C).",
        "Each mode is a characteristic multi-channel sine wave pattern.",
        "Embeddings compared: FFT baseline (deterministic), MOMENT (pretrained), TS2Vec (trained).",
        "Key question: **does the embedding produce geometrically coherent, mode-separable traces?**",
        "",
        "Metrics:",
        "- **MSI** (Mode Separability Index) — inter-mode distance / intra-mode spread. ↑ better.",
        "- **Loop Consistency** — DTW distance between repeated traces of the same mode. ↓ better.",
        "- **Transition Sharpness** — windows to cross midpoint after a mode change. ↓ better.",
        "- **PCA Compactness** — convex hull area of mode loop in PC1–PC2. ↓ better.",
        "- **Centroid Stability** — std of per-run mode centroids across test trajectories. ↓ better.",
        "",
    ]

    problem_labels = {
        "p1_simple_1d": "Problem 1 — Simple 1D (1 channel, clearly separated frequencies)",
        "p2_multichannel": "Problem 2 — Multi-channel (4ch, cross-channel frequency mixing)",
        "p3_hard_noisy": "Problem 3 — Hard / Noisy (4ch, similar frequencies, σ=0.20)",
    }

    for problem in PROBLEMS:
        lines.append(f"## {problem_labels[problem]}")
        lines.append("")
        lines.append(_build_table(results, problem))
        lines.append("")
        lines.append("**Selected plots (worm plot, one test trajectory):**")
        lines.append("")
        for model in MODELS:
            cell_dir = plots_dir / problem / model
            worm = cell_dir / f"{problem}_{model}_worm.png"
            loops = cell_dir / f"{problem}_{model}_mode_loops.png"
            if worm.exists():
                lines.append(f"- `{model}` worm: `{worm}`")
            if loops.exists():
                lines.append(f"- `{model}` loops: `{loops}`")
        lines.append("")

    lines += [
        "## Key Findings",
        "",
        "*(Fill in after running experiments.)*",
        "",
        "Expected sanity checks:",
        "- P1 FFT should show very high MSI (frequencies trivially separable by spectrum)",
        "- P3 should have lowest MSI across all models (similar freqs + high noise)",
        "- PCA worm for P1/FFT should show visually distinct closed loops per mode",
        "- TS2Vec trained on P3 should improve over FFT on MSI, demonstrating learned robustness",
        "",
    ]

    report_path = out_dir / "report.md"
    report_path.write_text("\n".join(lines))
    print(f"[report] written to {report_path}")

    # Also dump the full metrics as a flat JSON table for programmatic use
    flat = []
    for key, m in results.items():
        problem, model = key.split("/")
        row = {"problem": problem, "model": model}
        for mkey in METRIC_DISPLAY:
            row[mkey] = m.get(mkey)
        flat.append(row)

    summary_path = out_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(flat, f, indent=2)
    print(f"[report] summary JSON -> {summary_path}")


if __name__ == "__main__":
    main()
