"""Step 5: Plot metrics vs window size for both models.

Produces two figures (one per label perspective):
  sweep_step.png   — metrics with program_step_id labels
  sweep_opstate.png — metrics with op_state_id labels

Each figure is a 2×3 panel grid:
  Row 1: MSI ↑ | Silhouette ↑ | PCA Variance %
  Row 2: Loop DTW ↓ | Transition Sharpness ↓ | Centroid Stability ↓

X-axis is window size (log2 scale), one line per model.
A secondary x-axis shows approximate window duration (at 100 Hz sample rate).

Usage:
  python 05_plot_sweep.py --metrics_dir ../results/metrics \
      --output_dir ../results/plots
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

_HERE    = Path(__file__).resolve().parent
_RQ5_SRC = _HERE.parent / "src"
_RQ4_SRC = _HERE.parent.parent / "rq4_cnc_benchmark" / "src"
for _p in [str(_RQ5_SRC), str(_RQ4_SRC)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from rq5.config.sweep import WINDOW_SIZES, MODELS

_SAMPLE_RATE_HZ = 100  # assumed sample rate for duration annotation

_MODEL_STYLE = {
    "fft":    {"color": "#1f77b4", "marker": "o", "label": "FFT"},
    "moment": {"color": "#ff7f0e", "marker": "s", "label": "MOMENT"},
}

_PANELS = [
    # (metric_key, title, ylabel, higher_is_better)
    ("mode_separability_index", "Mode Separability Index", "MSI", True),
    ("silhouette_score",        "Silhouette Score",        "Silhouette", True),
    ("pca_variance_explained",  "PCA Variance (top-2 PCs)", "Fraction", True),
    ("loop_consistency_mean",   "Loop DTW Consistency",    "Mean DTW ↓", False),
    ("transition_sharpness",    "Transition Sharpness",    "Windows ↓", False),
    ("centroid_stability_mean", "Centroid Stability",      "Mean Std ↓", False),
]


def _load_metrics(metrics_dir: Path) -> List[dict]:
    rows = []
    for f in sorted(metrics_dir.glob("*.json")):
        try:
            rows.append(json.loads(f.read_text()))
        except Exception:
            pass
    return rows


def _build_series(rows: List[dict], label_suffix: str
                  ) -> Dict[str, Dict[int, Dict[str, Optional[float]]]]:
    """Returns {model: {window_length: {metric: value}}}."""
    series: Dict[str, Dict[int, Dict[str, Optional[float]]]] = {
        m: {ws: {} for ws in WINDOW_SIZES} for m in MODELS
    }
    for r in rows:
        if r.get("label") != label_suffix:
            continue
        ws    = r.get("window_length")
        model = r.get("model")
        if ws not in WINDOW_SIZES or model not in MODELS:
            continue
        for key, *_ in _PANELS:
            v = r.get(key)
            if v is not None and not (isinstance(v, float) and np.isnan(v)):
                series[model][ws][key] = v
    return series


def _plot_figure(series: Dict, label_name: str, out_path: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    fig.suptitle(f"RQ5 — Window Size Sweep  ({label_name} labels)", fontsize=13, y=1.01)

    x_log = np.log2(WINDOW_SIZES)

    for ax, (key, title, ylabel, higher) in zip(axes.flat, _PANELS):
        arrow = "↑" if higher else "↓"
        ax.set_title(f"{title}  {arrow}", fontsize=10)
        ax.set_xlabel("Window length (samples)")
        ax.set_ylabel(ylabel, fontsize=9)

        any_data = False
        for model in MODELS:
            style = _MODEL_STYLE[model]
            ys = [series[model][ws].get(key) for ws in WINDOW_SIZES]
            valid_x = [x_log[i] for i, v in enumerate(ys) if v is not None]
            valid_y = [v        for v in ys          if v is not None]
            if valid_y:
                ax.plot(valid_x, valid_y,
                        color=style["color"], marker=style["marker"],
                        linewidth=1.8, markersize=7, label=style["label"])
                any_data = True

        ax.set_xticks(x_log)
        ax.set_xticklabels([str(ws) for ws in WINDOW_SIZES], fontsize=8)

        # Secondary x-axis: duration in ms
        ax2 = ax.twiny()
        ax2.set_xlim(ax.get_xlim())
        ax2.set_xticks(x_log)
        ax2.set_xticklabels(
            [f"{ws*1000//_SAMPLE_RATE_HZ}ms" for ws in WINDOW_SIZES], fontsize=7
        )
        ax2.tick_params(axis="x", length=2)

        if any_data:
            ax.legend(fontsize=8, framealpha=0.6)
        ax.grid(True, alpha=0.3)

        # Shade "better" half with very light background
        ylims = ax.get_ylim()
        if higher:
            ax.fill_between(x_log, ylims[1] * 0.75, ylims[1],
                            alpha=0.04, color="green")
        else:
            ax.fill_between(x_log, ylims[0], ylims[0] + (ylims[1]-ylims[0]) * 0.25,
                            alpha=0.04, color="green")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out_path}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--metrics_dir", default="../results/metrics")
    p.add_argument("--output_dir",  default="../results/plots")
    return p.parse_args()


def main():
    args        = parse_args()
    metrics_dir = Path(args.metrics_dir)
    out_dir     = Path(args.output_dir)
    rows        = _load_metrics(metrics_dir)

    if not rows:
        print("No metric JSON files found — nothing to plot.")
        return

    for suffix, label_name in [("step", "Program Step"), ("opstate", "Operating State")]:
        series = _build_series(rows, suffix)
        _plot_figure(series, label_name, out_dir / f"sweep_{suffix}.png")


if __name__ == "__main__":
    main()
