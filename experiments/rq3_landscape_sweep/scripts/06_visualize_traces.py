"""Step 6: Generate worm-plot suites for all (scenario, model) cells.

Run directly on the login node (~3 min total).

Usage:
  python 06_visualize_traces.py --data_dir ../results/data \
      --embeddings_dir ../results/embeddings --output_dir ../results/plots
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

_RQ3_SRC = Path(__file__).parent.parent / "src"
_RQ2_SRC = Path(__file__).parent.parent.parent / "rq2_trace_comparison" / "src"
sys.path.insert(0, str(_RQ3_SRC))
sys.path.insert(0, str(_RQ2_SRC))

from rq3.config.scenarios import SCENARIO_IDS, get_scenario
from rq2.visualization.worm_plots import save_all_plots

MODELS = ["fft8", "fft32", "moment", "dcc"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, default="../results/data")
    p.add_argument("--embeddings_dir", type=str, default="../results/embeddings")
    p.add_argument("--output_dir", type=str, default="../results/plots")
    p.add_argument("--scenarios", nargs="+", default=None)
    p.add_argument("--models", nargs="+", default=None)
    p.add_argument("--stride", type=int, default=12)
    p.add_argument("--trajectory_id", type=int, default=0)
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    emb_dir = Path(args.embeddings_dir)
    out_dir = Path(args.output_dir)

    scenario_ids = args.scenarios if args.scenarios else SCENARIO_IDS
    models = args.models if args.models else MODELS

    t0_total = time.time()
    done, skipped, failed = 0, 0, 0

    for sid in scenario_ids:
        for model in models:
            cell_dir = out_dir / sid / model
            # Check if worm plot already exists as a proxy for completion
            if (cell_dir / f"{sid}_{model}_worm.png").exists() and not args.force:
                skipped += 1
                continue

            emb_path = emb_dir / f"{sid}_{model}.npz"
            if not emb_path.exists():
                failed += 1
                continue

            try:
                data = np.load(emb_path)
                embeddings = data["embeddings"]
                mode_ids = data["mode_id"]
                traj_ids = data["trajectory_id"]
                window_starts = data["window_start"]

                # Use the requested trajectory only
                tid_mask = traj_ids == args.trajectory_id
                if not tid_mask.any():
                    tid_mask = traj_ids == np.unique(traj_ids)[0]

                t_emb = embeddings[tid_mask]
                t_modes = mode_ids[tid_mask]
                t_starts = window_starts[tid_mask]

                sort_idx = np.argsort(t_starts)
                t_emb = t_emb[sort_idx]
                t_modes = t_modes[sort_idx]

                # Derive change-point window indices
                change_point_windows = [
                    i for i in range(1, len(t_modes)) if t_modes[i] != t_modes[i - 1]
                ]

                cell_dir.mkdir(parents=True, exist_ok=True)
                save_all_plots(
                    embeddings=t_emb,
                    mode_ids=t_modes,
                    change_point_windows=change_point_windows,
                    output_dir=cell_dir,
                    prefix=f"{sid}_{model}",
                )
                print(f"  [done] {sid:20s} × {model}")
                done += 1
            except Exception as e:
                print(f"  [ERROR] {sid} × {model}: {e}")
                failed += 1

    print(f"\nDone: {done}  Skipped: {skipped}  Failed: {failed}  ({time.time()-t0_total:.1f}s)")


if __name__ == "__main__":
    main()
