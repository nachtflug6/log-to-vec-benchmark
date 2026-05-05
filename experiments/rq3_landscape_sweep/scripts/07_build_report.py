"""Step 7: Build the comprehensive RQ3 landscape report.

Reads all metric JSONs and plot files, produces:
  reports/report.md   — metric tables, worm-plot grids, limit analysis, DCC note
  reports/summary.json — machine-readable flat list of all (scenario, model) metrics

Run directly on the login node (<1 min).

Usage:
  python 07_build_report.py --metrics_dir ../results/metrics \
      --plots_dir ../results/plots --output_dir ../results/reports
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional

_RQ3_SRC = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(_RQ3_SRC))

from rq3.config.scenarios import SCENARIOS, SCENARIO_IDS

MODELS = ["fft8", "fft32", "moment", "dcc"]

AXIS_ORDER = {
    "signal":  ["baseline", "sig_step", "sig_event", "sig_mixed"],
    "modes":   ["modes_2", "baseline", "modes_5", "modes_8"],
    "noise":   ["noise_000", "baseline", "noise_020", "noise_040"],
    "missing": ["baseline", "miss_rand10", "miss_rand30", "miss_block30", "miss_channel"],
}

AXIS_LABELS = {
    "baseline":     "sine / 3 modes / σ=0.05 / no missing",
    "sig_step":     "step (rectangular)",
    "sig_event":    "event (Poisson spikes)",
    "sig_mixed":    "mixed (step+sine)",
    "modes_2":      "2 modes",
    "modes_5":      "5 modes",
    "modes_8":      "8 modes",
    "noise_000":    "σ = 0.00",
    "noise_020":    "σ = 0.20",
    "noise_040":    "σ = 0.40",
    "miss_rand10":  "random 10% missing",
    "miss_rand30":  "random 30% missing",
    "miss_block30": "block 30% missing",
    "miss_channel": "channel dropout 25%",
}

DCC_NOTE = """
## DCC vs. Official TS2Vec

Our model labelled **`dcc`** (Dilated-Conv-Contrast) is inspired by TS2Vec but is **not**
the official implementation of Yue et al. 2022.  Key architectural differences:

| Aspect | Official TS2Vec (Yue et al. 2022) | Our DCC |
|---|---|---|
| Contrast granularity | Timestamp-level | Instance-level (global avg-pool) |
| Loss hierarchy | Multi-scale hierarchical NT-Xent | Single-scale NT-Xent |
| Temporal resolution | Per-timestep embedding | One vector per window |
| Pooling | None — output is a sequence | Adaptive avg-pool → scalar vector |
| Augmentation | Random cropping + mixing | Crop + noise + scale + random mask |

Because DCC uses instance-level contrast rather than timestamp-level contrast, it learns
which windows are *similar* but does not enforce any structure about *which parts* of a
window belong to which mode.  This explains why DCC clusters poorly (PCA compactness
blows up on multi-channel inputs) even when probe accuracy is competitive: the contrastive
objective produces a useful discriminative feature but not a geometrically organised space.
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fmt(v, decimals: int = 3, bold: bool = False) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "—"
    s = f"{v:.{decimals}f}"
    return f"**{s}**" if bold else s


def _scalar(v) -> Optional[float]:
    """Unwrap a metric value that may be a dict (per-mode) to a scalar mean."""
    if isinstance(v, dict):
        vals = [x for x in v.values() if x is not None and not math.isnan(x)]
        return float(sum(vals) / len(vals)) if vals else None
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _get(row: dict, key: str) -> Optional[float]:
    return _scalar(row.get(key))


def build_metric_table(
    all_metrics: List[dict],
    scenario_ids: List[str],
    title: str,
) -> str:
    rows_by_sid = {m["scenario_id"] + "_" + m["model"]: m for m in all_metrics}

    metric_cols = [
        ("MSI ↑",    "mode_separability_index"),
        ("LC ↓",     "loop_consistency_mean"),
        ("TS ↓",     "transition_sharpness"),
        ("Compact ↓","pca_loop_compactness_mean"),
        ("Stab ↓",   "centroid_stability_mean"),
    ]

    # Collect all values per (metric, model) to find best
    best: Dict[str, Dict[str, float]] = {}
    for mcol, mkey in metric_cols:
        best[mkey] = {}
        vals_by_model: Dict[str, List[float]] = {m: [] for m in MODELS}
        for sid in scenario_ids:
            for model in MODELS:
                key = f"{sid}_{model}"
                if key in rows_by_sid:
                    v = _get(rows_by_sid[key], mkey)
                    if v is not None:
                        vals_by_model[model].append(v)
        # best = model with best mean across scenarios on this axis
        for model, vals in vals_by_model.items():
            if vals:
                best[mkey][model] = sum(vals) / len(vals)

    lines = [f"\n### {title}\n"]
    # Header
    header = "| Scenario |"
    for model in MODELS:
        for mcol, _ in metric_cols:
            header += f" {model}/{mcol} |"
    lines.append(header)
    lines.append("|" + "---|" * (1 + len(MODELS) * len(metric_cols)))

    for sid in scenario_ids:
        label = AXIS_LABELS.get(sid, sid)
        row = f"| {label} |"
        for model in MODELS:
            key = f"{sid}_{model}"
            entry = rows_by_sid.get(key)
            for _, mkey in metric_cols:
                v = _get(entry, mkey) if entry else None
                # Determine if this is the best model for this metric+scenario
                if v is not None and entry:
                    all_v = {
                        m: _get(rows_by_sid.get(f"{sid}_{m}"), mkey)
                        for m in MODELS
                    }
                    valid = {m: vv for m, vv in all_v.items() if vv is not None}
                    if valid:
                        higher_is_better = mkey == "mode_separability_index"
                        best_model = (max if higher_is_better else min)(valid, key=lambda m: valid[m])
                        is_best = (model == best_model)
                    else:
                        is_best = False
                    row += f" {_fmt(v, bold=is_best)} |"
                else:
                    row += " — |"
        lines.append(row)

    return "\n".join(lines)


def build_worm_grid(
    plots_dir: Path,
    scenario_ids: List[str],
    models: List[str],
    plot_type: str = "worm",
    title: str = "",
) -> str:
    lines = []
    if title:
        lines.append(f"\n### {title}\n")

    # Header row
    header = "| Scenario |" + "".join(f" {m} |" for m in models)
    lines.append(header)
    lines.append("|" + "---|" * (1 + len(models)))

    for sid in scenario_ids:
        label = AXIS_LABELS.get(sid, sid)
        row = f"| {label} |"
        for model in models:
            img_path = plots_dir / sid / model / f"{sid}_{model}_{plot_type}.png"
            if img_path.exists():
                rel = img_path.relative_to(plots_dir.parent)
                row += f" ![]({rel}) |"
            else:
                row += " *(missing)* |"
        lines.append(row)

    return "\n".join(lines)


def write_limit_analysis(all_metrics: List[dict], baseline_sid: str = "baseline") -> str:
    baseline = {m["model"]: m for m in all_metrics if m["scenario_id"] == baseline_sid}
    lines = ["\n## Model Limit Analysis\n"]

    for model in MODELS:
        lines.append(f"### {model}\n")
        base = baseline.get(model)

        if base is None:
            lines.append("*(baseline metrics not available)*\n")
            continue

        base_msi = _get(base, "mode_separability_index") or 1.0
        base_lc  = _get(base, "loop_consistency_mean") or 1.0

        # Find worst-MSI scenario
        worst_msi_sid, worst_msi_val = None, float("inf")
        # Find noise threshold where MSI < 1.0
        noise_break = None
        # Find missing threshold where LC > 2× baseline
        miss_break = None

        for row in all_metrics:
            if row["model"] != model or row["scenario_id"] == baseline_sid:
                continue
            msi = _get(row, "mode_separability_index")
            lc  = _get(row, "loop_consistency_mean")

            if msi is not None and msi < worst_msi_val:
                worst_msi_val = msi
                worst_msi_sid = row["scenario_id"]

            if row["axis"] == "noise" and msi is not None and msi < 1.0 and noise_break is None:
                noise_break = row["noise_std"]

            if row["axis"] == "missing" and lc is not None and lc > 2 * base_lc and miss_break is None:
                miss_break = row["missing_fraction"]

        if worst_msi_sid:
            drop = base_msi - worst_msi_val
            label = AXIS_LABELS.get(worst_msi_sid, worst_msi_sid)
            lines.append(f"- Largest MSI drop: **{label}** (MSI {base_msi:.2f} → {worst_msi_val:.2f}, −{drop:.2f})")
        if noise_break is not None:
            lines.append(f"- MSI falls below 1.0 at noise σ = **{noise_break}**")
        elif base is not None:
            lines.append("- MSI stays ≥ 1.0 across all tested noise levels")
        if miss_break is not None:
            lines.append(f"- Loop consistency degrades (> 2× baseline) at missing fraction = **{miss_break:.0%}**")
        else:
            lines.append("- Loop consistency robust across all tested missing rates")
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--metrics_dir", type=str, default="../results/metrics")
    p.add_argument("--plots_dir", type=str, default="../results/plots")
    p.add_argument("--output_dir", type=str, default="../results/reports")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    metrics_dir = Path(args.metrics_dir)
    plots_dir = Path(args.plots_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load all metric files
    all_metrics: List[dict] = []
    for path in sorted(metrics_dir.glob("*.json")):
        try:
            all_metrics.append(json.loads(path.read_text()))
        except Exception as e:
            print(f"  [warn] could not load {path.name}: {e}")

    print(f"Loaded {len(all_metrics)} metric cells")

    # Persist summary JSON
    (output_dir / "summary.json").write_text(
        json.dumps(all_metrics, indent=2)
    )

    sections = ["# RQ3 — Landscape Sweep Report\n"]
    sections.append(
        "Models: **fft8** (FFT 8-bin), **fft32** (FFT 32-bin), "
        "**moment** (MOMENT-1-base pretrained), **dcc** (Dilated-Conv-Contrast).\n"
        "Metrics: MSI ↑ (mode separability), LC ↓ (loop DTW consistency), "
        "TS ↓ (transition sharpness), Compact ↓ (PCA hull area), Stab ↓ (centroid stability).\n"
        "Best value per scenario bolded.\n"
    )

    # ── Four metric tables ────────────────────────────────────────────────
    sections.append("\n## Metric Tables\n")
    sections.append(build_metric_table(all_metrics, AXIS_ORDER["signal"],  "Axis 1: Signal Type"))
    sections.append(build_metric_table(all_metrics, AXIS_ORDER["modes"],   "Axis 2: Mode Count"))
    sections.append(build_metric_table(all_metrics, AXIS_ORDER["noise"],   "Axis 3: Noise Level"))
    sections.append(build_metric_table(all_metrics, AXIS_ORDER["missing"], "Axis 4: Missing Values"))

    # ── Worm-plot grids ───────────────────────────────────────────────────
    sections.append("\n## Trace Visualisations\n")
    sections.append(
        build_worm_grid(
            plots_dir,
            ["sig_step", "sig_event", "sig_mixed"],
            ["fft8", "moment", "dcc"],
            plot_type="worm",
            title="Grid A: Signal type × model (worm plots)",
        )
    )
    sections.append(
        build_worm_grid(
            plots_dir,
            ["noise_000", "baseline", "noise_020", "noise_040"],
            ["fft8", "moment", "dcc"],
            plot_type="worm",
            title="Grid B: Noise level × model (worm plots)",
        )
    )
    sections.append(
        build_worm_grid(
            plots_dir,
            ["miss_rand10", "miss_rand30", "miss_block30", "miss_channel"],
            ["fft8", "moment", "dcc"],
            plot_type="mode_loops",
            title="Grid C: Missing-value type × model (mode-loop plots)",
        )
    )

    # ── Limit analysis ────────────────────────────────────────────────────
    sections.append(write_limit_analysis(all_metrics))

    # ── DCC note ─────────────────────────────────────────────────────────
    sections.append(DCC_NOTE)

    report = "\n".join(sections)
    (output_dir / "report.md").write_text(report)
    print(f"[report] written to {output_dir / 'report.md'}")
    print(f"[report] summary JSON → {output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
