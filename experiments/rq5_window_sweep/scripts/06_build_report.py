"""Step 6: Build markdown report from RQ5 window-sweep metrics.

Usage:
  python 06_build_report.py --metrics_dir ../results/metrics \
      --output_dir ../results/reports
"""
from __future__ import annotations

import argparse, json
from pathlib import Path
from typing import List

import sys
_HERE    = Path(__file__).resolve().parent
_RQ5_SRC = _HERE.parent / "src"
_RQ4_SRC = _HERE.parent.parent / "rq4_cnc_benchmark" / "src"
for _p in [str(_RQ5_SRC), str(_RQ4_SRC)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from rq5.config.sweep import WINDOW_SIZES, MODELS


def _load_metrics(metrics_dir: Path) -> List[dict]:
    rows = []
    for f in sorted(metrics_dir.glob("*.json")):
        try:
            rows.append(json.loads(f.read_text()))
        except Exception:
            pass
    return rows


def _table(rows: List[dict], label_suffix: str) -> str:
    header = ("| Window | Stride | Model | MSI ↑ | Silhouette ↑ | "
              "Loop DTW ↓ | Trans. Sharp ↓ | Centroid Stab ↓ | PCA Var % |")
    sep    = "|---|---|---|---|---|---|---|---|---|"
    lines  = [header, sep]
    for ws in WINDOW_SIZES:
        sid = f"cnc_ws_{ws:03d}"
        for model in MODELS:
            r = next((x for x in rows if x.get("scenario_id") == sid
                      and x.get("model") == model
                      and x.get("label") == label_suffix), None)
            stride = ws // 8
            if r is None:
                lines.append(f"| {ws} | {stride} | {model} | — | — | — | — | — | — |")
                continue
            lines.append(
                f"| {ws} | {stride} | {model} "
                f"| {r.get('mode_separability_index', float('nan')):.2f} "
                f"| {r.get('silhouette_score', float('nan')):.3f} "
                f"| {r.get('loop_consistency_mean', float('nan')):.2f} "
                f"| {r.get('transition_sharpness', float('nan')):.2f} "
                f"| {r.get('centroid_stability_mean', float('nan')):.4f} "
                f"| {r.get('pca_variance_explained', float('nan'))*100:.1f}% |"
            )
    return "\n".join(lines)


def build_report(metrics_dir: Path, output_dir: Path) -> None:
    rows = _load_metrics(metrics_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    lines = [
        "# RQ5 — Window Size Sweep\n",
        "## Setup\n",
        "Base scenario: `cnc_part_full` (square contour, 10 channels, 20 runs, σ=0.05)\n",
        f"Window sizes tested: {WINDOW_SIZES}  (stride = window_length // 8)\n",
        "Models: FFT amplitude spectrum, MOMENT (pads to 512 for short windows)\n",
        "\nKey questions:\n",
        "- Does increasing window size improve MSI / silhouette by capturing "
        "more periodic structure?\n",
        "- Does a larger window hurt transition sharpness by spanning mode "
        "boundaries?\n",
        "- Is there a sweet spot that balances spectral richness with "
        "temporal precision?\n",
        "- How does MOMENT's mandatory 512-sample padding affect short-window "
        "quality?\n",
        "\n![Sweep overview (program step)](../plots/sweep_step.png)\n",
        "\n![Sweep overview (operating state)](../plots/sweep_opstate.png)\n",
        "\n## Results — Program Step Labels\n",
        _table(rows, "step"),
        "\n## Results — Operating State Labels\n",
        _table(rows, "opstate"),
        "\n## Findings\n",
        "*(fill in after running experiments)*\n",
    ]

    (output_dir / "report.md").write_text("\n".join(lines))
    print(f"Report → {output_dir / 'report.md'}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--metrics_dir", default="../results/metrics")
    p.add_argument("--output_dir",  default="../results/reports")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_report(Path(args.metrics_dir), Path(args.output_dir))
