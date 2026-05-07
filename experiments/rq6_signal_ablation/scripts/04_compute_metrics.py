"""Step 4: Compute trace metrics for all (scenario, model) cells in RQ6.

Run directly on the login node (~2 min for all 24 cells).

Usage:
  python 04_compute_metrics.py --data_dir ../results/data \
      --embeddings_dir ../results/embeddings --output_dir ../results/metrics
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np

_RQ6_SRC  = Path(__file__).parent.parent / "src"
_REPO_ROOT = Path(__file__).parent.parent.parent.parent   # log-to-vec-benchmark/
_RQ2_SRC  = _REPO_ROOT / "experiments" / "rq2_trace_comparison" / "src"

for _p in [str(_RQ6_SRC), str(_RQ2_SRC)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from rq6.config.scenarios import SCENARIOS, SCENARIO_IDS, get_scenario
from rq2.evaluation.trace_metrics import compute_all_metrics

MODELS = ["fft8", "fft32", "moment"]


def _load_embeddings(emb_path: Path) -> Dict:
    data = np.load(emb_path)
    return {k: data[k] for k in data.files}


def _build_trajectory_data(emb_data: Dict, stride: int):
    embeddings   = emb_data["embeddings"]
    mode_ids     = emb_data["mode_id"]
    traj_ids     = emb_data["trajectory_id"]
    window_starts = emb_data["window_start"]

    unique_trajs = np.unique(traj_ids)
    per_traj_embs:  List[np.ndarray] = []
    per_traj_modes: List[np.ndarray] = []
    per_traj_segs:  Dict[int, List]  = {}

    for tid in unique_trajs:
        mask   = traj_ids == tid
        t_emb  = embeddings[mask]
        t_modes = mode_ids[mask]
        t_starts = window_starts[mask]

        sort_idx = np.argsort(t_starts)
        t_emb    = t_emb[sort_idx]
        t_modes  = t_modes[sort_idx]

        per_traj_embs.append(t_emb)
        per_traj_modes.append(t_modes)

        segs: Dict[int, List[np.ndarray]] = {}
        current_mode = t_modes[0]
        seg_start = 0
        for i in range(1, len(t_modes)):
            if t_modes[i] != current_mode:
                seg_emb = t_emb[seg_start:i]
                if len(seg_emb) >= 3:
                    segs.setdefault(int(current_mode), []).append(seg_emb)
                current_mode = t_modes[i]
                seg_start = i
        seg_emb = t_emb[seg_start:]
        if len(seg_emb) >= 3:
            segs.setdefault(int(current_mode), []).append(seg_emb)

        for m, seg_list in segs.items():
            per_traj_segs.setdefault(m, []).extend(seg_list)

    all_modes = mode_ids
    change_point_windows = [
        i for i in range(1, len(all_modes)) if all_modes[i] != all_modes[i - 1]
    ]
    return embeddings, all_modes, change_point_windows, per_traj_embs, per_traj_modes, per_traj_segs


def compute_scenario_model(
    scenario_id: str,
    model: str,
    data_dir: Path,
    emb_dir: Path,
    stride: int = 12,
) -> dict:
    emb_path = emb_dir / f"{scenario_id}_{model}.npz"
    if not emb_path.exists():
        raise FileNotFoundError(f"Embeddings not found: {emb_path}")

    emb_data = _load_embeddings(emb_path)
    cfg = get_scenario(scenario_id)

    (all_emb, all_modes, change_point_windows,
     per_traj_embs, per_traj_modes, per_traj_segs) = _build_trajectory_data(emb_data, stride)

    metrics = compute_all_metrics(
        embeddings=all_emb,
        mode_ids=all_modes,
        change_point_windows=change_point_windows,
        per_trajectory_embeddings=per_traj_embs,
        per_trajectory_mode_ids=per_traj_modes,
        per_trajectory_segments=per_traj_segs,
        window_stride=stride,
    )

    result = {
        "scenario_id":    scenario_id,
        "model":          model,
        "axis":           cfg.axis,
        "signal_type":    cfg.signal_type,
        "step_channels":  cfg.step_channels,
        "num_modes":      cfg.num_modes,
        "noise_std":      cfg.noise_std,
        "missing_type":   cfg.missing_type,
        "missing_fraction": cfg.missing_fraction,
    }
    result.update(metrics)
    return result


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir",       type=str, default="../results/data")
    p.add_argument("--embeddings_dir", type=str, default="../results/embeddings")
    p.add_argument("--output_dir",     type=str, default="../results/metrics")
    p.add_argument("--scenarios",      nargs="+", default=None)
    p.add_argument("--models",         nargs="+", default=None)
    p.add_argument("--stride",         type=int, default=12)
    p.add_argument("--force",          action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    emb_dir  = Path(args.embeddings_dir)
    out_dir  = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    scenario_ids = args.scenarios if args.scenarios else SCENARIO_IDS
    models       = args.models    if args.models    else MODELS

    total = len(scenario_ids) * len(models)
    print(f"Computing metrics: {len(scenario_ids)} scenarios × {len(models)} models = {total} cells")

    done, skipped, failed = 0, 0, 0
    t0_total = time.time()

    for sid in scenario_ids:
        for model in models:
            out_path = out_dir / f"{sid}_{model}.json"
            if out_path.exists() and not args.force:
                skipped += 1
                continue
            try:
                t0     = time.time()
                result = compute_scenario_model(sid, model, data_dir, emb_dir, args.stride)
                out_path.write_text(json.dumps(result, indent=2))
                msi = result.get("mode_separability_index", float("nan"))
                lc  = result.get("loop_consistency_mean",  float("nan"))
                print(f"  [{sid:20s} × {model:6s}]  MSI={msi:.2f}  LC={lc:.2f}  {time.time()-t0:.1f}s")
                done += 1
            except FileNotFoundError as e:
                print(f"  [MISSING] {sid} × {model}: {e}")
                failed += 1
            except Exception as e:
                print(f"  [ERROR]   {sid} × {model}: {e}")
                failed += 1

    print(f"\nDone: {done}  Skipped: {skipped}  Failed: {failed}  ({time.time()-t0_total:.1f}s total)")


if __name__ == "__main__":
    main()
