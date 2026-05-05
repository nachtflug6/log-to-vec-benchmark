"""Step 6: Build markdown report from RQ4 metrics.

Generates one report covering all scenarios × models × both label perspectives.

Usage:
  python 06_build_report.py --metrics_dir ../results/metrics \
      --output_dir ../results/reports
"""
from __future__ import annotations

import argparse, json
from pathlib import Path
from typing import Dict, List

MODELS    = ["fft", "moment"]
LABELS    = [("step", "Program Step"), ("opstate", "Operating State")]


def _load_metrics(metrics_dir: Path) -> List[dict]:
    rows = []
    for f in sorted(metrics_dir.glob("*.json")):
        try:
            rows.append(json.loads(f.read_text()))
        except Exception:
            pass
    return rows


def _metric_table(rows: List[dict], scenarios: List[str], label_suffix: str) -> str:
    header = ("| Scenario | Model | MSI ↑ | Silhouette ↑ | Loop DTW ↓ "
              "| Trans. Sharp ↓ | Centroid Stab ↓ | PCA Var % |")
    sep    = "|---|---|---|---|---|---|---|---|"
    lines  = [header, sep]
    for sid in scenarios:
        for model in MODELS:
            r = next((x for x in rows if x.get("scenario_id") == sid
                      and x.get("model") == model
                      and x.get("label", "").endswith(label_suffix)), None)
            if r is None:
                lines.append(f"| {sid} | {model} | — | — | — | — | — | — |")
                continue
            lines.append(
                f"| {sid} | {model} "
                f"| {r.get('mode_separability_index', float('nan')):.2f} "
                f"| {r.get('silhouette_score', float('nan')):.3f} "
                f"| {r.get('loop_consistency_mean', float('nan')):.2f} "
                f"| {r.get('transition_sharpness', float('nan')):.2f} "
                f"| {r.get('centroid_stability_mean', float('nan')):.4f} "
                f"| {r.get('pca_variance_explained', float('nan'))*100:.1f}% |"
            )
    return "\n".join(lines)


def build_report(metrics_dir: Path, output_dir: Path) -> None:
    rows      = _load_metrics(metrics_dir)
    scenarios = sorted(set(r["scenario_id"] for r in rows))
    output_dir.mkdir(parents=True, exist_ok=True)

    lines = [
        "# RQ4 — CNC-Analogous Mixed-Signal Benchmark\n",
        "## Overview\n",
        "Seven scenarios with two-level label structure:\n",
        "- **program_step_id** — which G-code block is executing (fine-grained, industrial KPI)\n",
        "- **op_state_id** — operating state: IDLE/HOMING/RAPID/CUTTING/DWELL/FAULT (coarse)\n",
        "\n10 channels: axis_x, axis_y, axis_z, spindle_speed, feed_rate, "
        "vibration_x, vibration_y, temperature, tool_id (discrete), alarm_code (discrete)\n",
        "\nKey question: *Can embeddings separate program steps using coordinate + vibration "
        "structure? And does the same step cluster consistently across repeated part runs?*\n",
    ]

    for suffix, label_name in LABELS:
        lines.append(f"\n## Results — {label_name} Labels\n")
        lines.append(_metric_table(rows, scenarios, suffix))
        lines.append("")

    lines += [
        "\n## Scenario Descriptions\n",
        "| Scenario | Programs | Runs | Channels | Key variation |",
        "|---|---|---|---|---|",
        "| cnc_part_simple  | Square contour     | 20 | 6  | Baseline, easy |",
        "| cnc_part_full    | Square contour     | 20 | 10 | Full channel set |",
        "| cnc_two_parts    | Square + Triangle  | 20 | 10 | Cross-part separation |",
        "| cnc_tool_wear    | Square contour     | 20 | 10 | Vib. amplitude +4%/run |",
        "| cnc_noisy        | Square contour     | 20 | 10 | σ_noise = 0.25 |",
        "| cnc_pocket       | Pocket milling     | 20 | 10 | Z-varying, 11 steps |",
        "| cnc_faults       | Square + faults    | 20 | 10 | ~8% fault injection |",
        "",
        "\n## Key Findings\n",
        "*(fill in after running experiments)*\n",
    ]

    report = "\n".join(lines)
    (output_dir / "report.md").write_text(report)

    summary = [{"scenario_id": r["scenario_id"], "model": r["model"],
                "label": r.get("label", ""),
                "silhouette_score": r.get("silhouette_score"),
                "mode_separability_index": r.get("mode_separability_index"),
                "loop_consistency_mean": r.get("loop_consistency_mean"),
                "pca_variance_explained": r.get("pca_variance_explained")}
               for r in rows]
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
