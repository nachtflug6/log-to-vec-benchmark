"""Step 5: Build markdown report from RQ7 metrics.

Usage:
  python 05_build_report.py --metrics_dir ../results/metrics \
      --output_dir ../results/reports
"""
from __future__ import annotations

import argparse, json
from pathlib import Path
from typing import List

MODELS = ["fft", "moment"]


def _load_metrics(metrics_dir: Path) -> List[dict]:
    rows = []
    for f in sorted(metrics_dir.glob("*.json")):
        try:
            rows.append(json.loads(f.read_text()))
        except Exception:
            pass
    return rows


def _metric_table(rows: List[dict], scenarios: List[str]) -> str:
    header = ("| Scenario | Dataset | Model | Pseudo | MSI ↑ | Silhouette ↑ "
              "| Loop DTW ↓ | Centroid Stab ↓ | PCA Var % |")
    sep    = "|---|---|---|---|---|---|---|---|---|"
    lines  = [header, sep]
    for sid in scenarios:
        sc_rows = [r for r in rows if r.get("scenario_id") == sid]
        dataset = sc_rows[0]["dataset"] if sc_rows else "?"
        for model in MODELS:
            r = next((x for x in sc_rows if x.get("model") == model), None)
            pseudo_tag = "yes*" if (r and r.get("pseudo_runs")) else "no"
            if r is None:
                lines.append(f"| {sid} | {dataset} | {model} | — | — | — | — | — | — |")
                continue
            lines.append(
                f"| {sid} | {dataset} | {model} | {pseudo_tag} "
                f"| {r.get('mode_separability_index', float('nan')):.2f} "
                f"| {r.get('silhouette_score', float('nan')):.3f} "
                f"| {r.get('loop_consistency_mean', float('nan')):.2f} "
                f"| {r.get('centroid_stability_mean', float('nan')):.4f} "
                f"| {r.get('pca_variance_explained', float('nan'))*100:.1f}% |"
            )
    return "\n".join(lines)


def build_report(metrics_dir: Path, output_dir: Path) -> None:
    rows      = _load_metrics(metrics_dir)
    scenarios = sorted(set(r["scenario_id"] for r in rows))
    output_dir.mkdir(parents=True, exist_ok=True)

    lines = [
        "# RQ7 — Real-World Dataset Evaluation\n",
        "## Overview\n",
        "Three real-world datasets evaluated with two embedding methods (FFT, MOMENT).\n",
        "Datasets:\n",
        "- **SWaT** (Secure Water Treatment): 51 sensors, binary normal/attack labels\n",
        "- **MSL** (Mars Science Laboratory): 55 telemetry channels, binary anomaly labels\n",
        "- **TEP** (Tennessee Eastman Process, Reinartz 2021): 52 process variables, "
        "6 production modes or 9 fault classes\n",
        "\n*Pseudo-runs (yes\\*): single recording split into K=20 equal time-chunks "
        "as proxy independent runs. Loop DTW is indicative-only for these datasets.*\n",
        "\n## Results\n",
        _metric_table(rows, scenarios),
        "",
        "\n## Scenario Notes\n",
        "| Scenario | Description | Channels | Modes | Runs |",
        "|---|---|---|---|---|",
        "| swat_binary | SWaT test set, normal vs attack | 51 | 2 | 20 pseudo |",
        "| msl_binary  | MSL test set, normal vs anomaly | 55 | 2 | 20 pseudo |",
        "| tep_modes   | TEP fault-free, 6 production modes | 52 | 6 | 500 genuine |",
        "| tep_faults  | TEP IDV0-IDV8 fault types | 52 | 9 | 500 per fault |",
        "",
        "\n## Key Findings\n",
        "*(fill in after running experiments)*\n",
        "\n## Comparison with Synthetic Results (RQ3)\n",
        "*(compare MSI and silhouette values against RQ3 sig_step / sig_mixed baselines)*\n",
    ]

    report = "\n".join(lines)
    (output_dir / "report.md").write_text(report)

    summary = [
        {
            "scenario_id":            r["scenario_id"],
            "model":                  r["model"],
            "dataset":                r.get("dataset", ""),
            "pseudo_runs":            r.get("pseudo_runs", False),
            "silhouette_score":       r.get("silhouette_score"),
            "mode_separability_index":r.get("mode_separability_index"),
            "loop_consistency_mean":  r.get("loop_consistency_mean"),
            "pca_variance_explained": r.get("pca_variance_explained"),
        }
        for r in rows
    ]
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"Report → {output_dir / 'report.md'}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--metrics_dir", default="../results/metrics")
    p.add_argument("--output_dir",  default="../results/reports")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_report(Path(args.metrics_dir), Path(args.output_dir))
