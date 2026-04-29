"""Step 4: Compute trace metrics for all (problem, model) experiment cells.

For each cell, loads the embedding .npz and the trajectory timelines, then
computes: MSI, loop consistency (DTW), transition sharpness, PCA loop
compactness, centroid stability.

Usage:
  python 04_compute_trace_metrics.py --data_dir data --embeddings_dir embeddings --output_dir metrics
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from rq2.evaluation.trace_metrics import (
    compute_all_metrics,
    _extract_mode_segments,
)

PROBLEMS = ["p1_simple_1d", "p2_multichannel", "p3_hard_noisy"]
MODELS = ["fft", "moment", "ts2vec"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, default="data")
    p.add_argument("--embeddings_dir", type=str, default="embeddings")
    p.add_argument("--output_dir", type=str, default="metrics")
    p.add_argument("--stride", type=int, default=12)
    p.add_argument("--problems", nargs="+", default=PROBLEMS)
    p.add_argument("--models", nargs="+", default=MODELS)
    return p.parse_args()


def _load_timelines(data_dir: Path, problem: str) -> Dict[int, dict]:
    """Return {traj_id: {change_points, mode_sequence}} from saved timeline JSONs."""
    traj_dir = data_dir / problem / "trajectories"
    timelines = {}
    for f in sorted(traj_dir.glob("traj_*_timeline.json")):
        with open(f) as fh:
            d = json.load(fh)
        timelines[d["trajectory_id"]] = d
    return timelines


def _change_points_to_windows(change_points: List[int], stride: int) -> List[int]:
    """Convert timestep-level change points to approximate window indices."""
    return [cp // stride for cp in change_points if cp > 0]


def _save_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    def _convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {_convert(k): _convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_convert(v) for v in obj]
        return obj

    with open(path, "w") as f:
        json.dump(_convert(data), f, indent=2)


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_dir)
    emb_root = Path(args.embeddings_dir)
    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    for problem in args.problems:
        timelines = _load_timelines(data_root, problem)

        for model in args.models:
            emb_path = emb_root / f"{problem}_{model}.npz"
            if not emb_path.exists():
                print(f"[skip] {problem} × {model}: embedding not found at {emb_path}")
                continue

            print(f"[metrics] {problem} × {model}  loading {emb_path}")
            data = np.load(emb_path)
            embeddings = data["embeddings"]
            mode_ids = data["mode_id"]
            traj_ids = data["trajectory_id"]
            window_starts = data["window_start"]

            # --- Collect per-trajectory data for multi-trajectory metrics ---
            per_traj_emb: List[np.ndarray] = []
            per_traj_mode: List[np.ndarray] = []
            all_cp_windows: List[int] = []

            # Build per-mode segment dict (aggregated across all trajectories)
            agg_segments: Dict[int, List[np.ndarray]] = {}

            for traj_id, tl in timelines.items():
                mask = traj_ids == traj_id
                if not mask.any():
                    continue
                t_emb = embeddings[mask]
                t_mode = mode_ids[mask]
                t_starts = window_starts[mask]

                per_traj_emb.append(t_emb)
                per_traj_mode.append(t_mode)

                cp_windows = _change_points_to_windows(tl["change_points"], args.stride)
                all_cp_windows.extend(
                    # Convert to global-ish index (not used for TS on global trace)
                    # We use per-trajectory CP windows for transition sharpness
                    cp_windows
                )

                # Extract contiguous mode segments for this trajectory
                segs = _extract_mode_segments(t_emb, t_mode, tl["change_points"], args.stride)
                for m, seg_list in segs.items():
                    agg_segments.setdefault(m, []).extend(seg_list)

            # --- Compute metrics on the full (concatenated) embedding set ---
            # Use first test trajectory for worm-plot-style metrics
            unique_trajs = sorted(timelines.keys())
            test_trajs = unique_trajs[int(0.85 * len(unique_trajs)):]  # last 15%

            test_emb_parts = []
            test_mode_parts = []
            test_cp_windows = []

            for traj_id in test_trajs:
                mask = traj_ids == traj_id
                if not mask.any():
                    continue
                test_emb_parts.append(embeddings[mask])
                test_mode_parts.append(mode_ids[mask])
                tl = timelines[traj_id]
                test_cp_windows.extend(
                    _change_points_to_windows(tl["change_points"], args.stride)
                )

            if not test_emb_parts:
                print(f"  [warn] no test trajectories found, using all data")
                test_emb = embeddings
                test_mode = mode_ids
                test_cp_windows = sorted(set(all_cp_windows))
            else:
                test_emb = np.concatenate(test_emb_parts, axis=0)
                test_mode = np.concatenate(test_mode_parts, axis=0)

            metrics = compute_all_metrics(
                embeddings=test_emb,
                mode_ids=test_mode,
                change_point_windows=test_cp_windows,
                per_trajectory_embeddings=per_traj_emb,
                per_trajectory_mode_ids=per_traj_mode,
                per_trajectory_segments=agg_segments,
                window_stride=args.stride,
            )

            metrics["problem"] = problem
            metrics["model"] = model
            metrics["num_windows_total"] = int(len(embeddings))
            metrics["num_windows_eval"] = int(len(test_emb))
            metrics["embedding_dim"] = int(embeddings.shape[1])

            out_path = out_root / f"{problem}_{model}.json"
            _save_json(out_path, metrics)

            msi = metrics.get("mode_separability_index", float("nan"))
            lc = metrics.get("loop_consistency_mean", float("nan"))
            ts_val = metrics.get("transition_sharpness", float("nan"))
            print(f"  MSI={msi:.3f}  LC={lc:.3f}  TS={ts_val:.3f}")


if __name__ == "__main__":
    main()
