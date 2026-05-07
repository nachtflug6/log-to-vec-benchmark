"""Step 5: Build the RQ6 signal-ablation report.

Reads all metric JSONs, produces:
  reports/report.md    — two metric tables (signal type × channel ratio)
  reports/summary.json — machine-readable flat list of all cells

Run directly on the login node (<1 min).

Usage:
  python 05_build_report.py --metrics_dir ../results/metrics \
      --output_dir ../results/reports
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional

_RQ6_SRC = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(_RQ6_SRC))

from rq6.config.scenarios import SCENARIOS, SCENARIO_IDS

MODELS = ["fft8", "fft32", "moment"]

AXIS_ORDER = {
    "signal": ["ratio_0step", "sig_sawtooth", "sig_damped",  "sig_chirp"],
    "ratio":  ["ratio_0step", "ratio_1step",  "ratio_2step", "ratio_3step", "ratio_4step"],
}

AXIS_LABELS = {
    "ratio_0step":  "0/4 step ch  (pure sine)",
    "sig_sawtooth": "sawtooth",
    "sig_damped":   "damped sine",
    "sig_chirp":    "linear chirp",
    "ratio_1step":  "1/4 step ch",
    "ratio_2step":  "2/4 step ch  (= RQ3 mixed)",
    "ratio_3step":  "3/4 step ch",
    "ratio_4step":  "4/4 step ch  (pure step)",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fmt(v, decimals: int = 3, bold: bool = False) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "—"
    s = f"{v:.{decimals}f}"
    return f"**{s}**" if bold else s


def _scalar(v) -> Optional[float]:
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
    return _scalar(row.get(key)) if row else None


def build_metric_table(
    all_metrics: List[dict],
    scenario_ids: List[str],
    title: str,
) -> str:
    rows_by_sid = {m["scenario_id"] + "_" + m["model"]: m for m in all_metrics}

    metric_cols = [
        ("MSI ↑",     "mode_separability_index"),
        ("LC ↓",      "loop_consistency_mean"),
        ("TS ↓",      "transition_sharpness"),
        ("Compact ↓", "pca_loop_compactness_mean"),
        ("Stab ↓",    "centroid_stability_mean"),
    ]

    lines = [f"\n### {title}\n"]
    header = "| Scenario |"
    for model in MODELS:
        for mcol, _ in metric_cols:
            header += f" {model}/{mcol} |"
    lines.append(header)
    lines.append("|" + "---|" * (1 + len(MODELS) * len(metric_cols)))

    for sid in scenario_ids:
        label = AXIS_LABELS.get(sid, sid)
        row   = f"| {label} |"
        for model in MODELS:
            key   = f"{sid}_{model}"
            entry = rows_by_sid.get(key)
            for _, mkey in metric_cols:
                v = _get(entry, mkey)
                if v is not None:
                    all_v = {
                        m: _get(rows_by_sid.get(f"{sid}_{m}"), mkey)
                        for m in MODELS
                    }
                    valid = {m: vv for m, vv in all_v.items() if vv is not None}
                    if valid:
                        higher_is_better = (mkey == "mode_separability_index")
                        best_model = (max if higher_is_better else min)(valid, key=lambda m: valid[m])
                        row += f" {_fmt(v, bold=(model == best_model))} |"
                    else:
                        row += f" {_fmt(v)} |"
                else:
                    row += " — |"
        lines.append(row)

    return "\n".join(lines)


def build_msi_trend(all_metrics: List[dict]) -> str:
    """Simple MSI-only table for the ratio axis to show the step-vs-sine breakpoint."""
    rows = {m["scenario_id"] + "_" + m["model"]: m for m in all_metrics}
    ratio_ids = AXIS_ORDER["ratio"]

    lines = ["\n### MSI trend across channel-ratio sweep\n"]
    header = "| Step channels |" + "".join(f" {m}/MSI |" for m in MODELS)
    lines.append(header)
    lines.append("|" + "---|" * (1 + len(MODELS)))

    for sid in ratio_ids:
        n_step = sid.replace("ratio_", "").replace("step", "")
        label  = f"{n_step} / 4"
        row    = f"| {label} |"
        for model in MODELS:
            v = _get(rows.get(f"{sid}_{model}"), "mode_separability_index")
            row += f" {_fmt(v, decimals=2)} |"
        lines.append(row)

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--metrics_dir", type=str, default="../results/metrics")
    p.add_argument("--output_dir",  type=str, default="../results/reports")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    metrics_dir = Path(args.metrics_dir)
    output_dir  = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_metrics: List[dict] = []
    for path in sorted(metrics_dir.glob("*.json")):
        try:
            all_metrics.append(json.loads(path.read_text()))
        except Exception as e:
            print(f"  [warn] could not load {path.name}: {e}")

    print(f"Loaded {len(all_metrics)} metric cells")

    (output_dir / "summary.json").write_text(json.dumps(all_metrics, indent=2))

    sections = ["# RQ6 — Signal Ablation Report\n"]
    sections.append(
        "**Research question:** Where does MOMENT's advantage over FFT disappear "
        "as signals become less periodic?\n\n"
        "**Axis A** — Signal type: ratio_0step (pure sine anchor), sawtooth, damped sine, chirp.\n"
        "**Axis B** — Channel ratio: 0–4 out of 4 channels carrying step signal "
        "(0 = pure sine, 4 = pure step).\n\n"
        "Models: **fft8** (FFT 8-bin), **fft32** (FFT 32-bin), "
        "**moment** (MOMENT-1-base pretrained).\n"
        "Metrics: MSI ↑, LC ↓, TS ↓, Compact ↓, Stab ↓.  Best per row bolded.\n"
    )

    sections.append("\n## Metric Tables\n")
    sections.append(
        build_metric_table(all_metrics, AXIS_ORDER["signal"], "Axis A: Signal Type")
    )
    sections.append(
        build_metric_table(all_metrics, AXIS_ORDER["ratio"],  "Axis B: Channel Ratio (step vs sine)")
    )
    sections.append(build_msi_trend(all_metrics))

    sections.append(
        "\n## Interpretation\n\n"
        "- **Axis A** tests whether MOMENT's advantage is specific to *pure sine* or "
        "extends to other periodic/oscillatory signal families (sawtooth, damped, chirp).\n"
        "- **Axis B** locates the *breakpoint* in step-channel fraction where FFT overtakes "
        "MOMENT on MSI.  The RQ3 result (sig_step: FFT MSI 9.6, MOMENT MSI 2.8) anchors "
        "the pure-step end; this sweep fills in the intermediate cases.\n"
    )

    report = "\n".join(sections)
    (output_dir / "report.md").write_text(report)
    print(f"[report] written to {output_dir / 'report.md'}")
    print(f"[report] summary JSON → {output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
