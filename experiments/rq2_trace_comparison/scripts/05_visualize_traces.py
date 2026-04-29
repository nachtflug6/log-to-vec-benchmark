"""Step 5: Generate all 5 plot types for every (problem, model) experiment cell.

Usage:
  python 05_visualize_traces.py --data_dir data --embeddings_dir embeddings --output_dir plots
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from rq2.visualization.worm_plots import save_all_plots

PROBLEMS = ["p1_simple_1d", "p2_multichannel", "p3_hard_noisy"]
MODELS = ["fft", "moment", "ts2vec"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, default="data")
    p.add_argument("--embeddings_dir", type=str, default="embeddings")
    p.add_argument("--output_dir", type=str, default="plots")
    p.add_argument("--stride", type=int, default=12)
    p.add_argument("--trajectory_id", type=int, default=0,
                   help="Which trajectory to use for the worm/loop plots (picks one full run)")
    p.add_argument("--problems", nargs="+", default=PROBLEMS)
    p.add_argument("--models", nargs="+", default=MODELS)
    return p.parse_args()


def _load_timeline(data_dir: Path, problem: str, traj_id: int) -> dict:
    path = data_dir / problem / "trajectories" / f"traj_{traj_id:03d}_timeline.json"
    if not path.exists():
        return {"change_points": [], "mode_sequence": []}
    with open(path) as f:
        return json.load(f)


def _cp_to_windows(change_points: List[int], stride: int) -> List[int]:
    return [cp // stride for cp in change_points if cp > 0]


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_dir)
    emb_root = Path(args.embeddings_dir)
    out_root = Path(args.output_dir)

    for problem in args.problems:
        for model in args.models:
            emb_path = emb_root / f"{problem}_{model}.npz"
            if not emb_path.exists():
                print(f"[skip] {problem} × {model}: embedding not found")
                continue

            data = np.load(emb_path)
            embeddings = data["embeddings"]
            mode_ids = data["mode_id"]
            traj_ids = data["trajectory_id"]

            # Select one representative trajectory for the trace plots
            traj_id = args.trajectory_id
            mask = traj_ids == traj_id
            if not mask.any():
                available = np.unique(traj_ids)
                traj_id = int(available[0])
                mask = traj_ids == traj_id

            traj_emb = embeddings[mask]
            traj_mode = mode_ids[mask]

            tl = _load_timeline(data_root, problem, traj_id)
            cp_windows = _cp_to_windows(tl.get("change_points", []), args.stride)

            prefix = f"{problem}_{model}"
            cell_dir = out_root / problem / model
            print(f"[viz] {problem} × {model}  ({len(traj_emb)} windows, traj {traj_id})")

            save_all_plots(
                embeddings=traj_emb,
                mode_ids=traj_mode,
                change_point_windows=cp_windows,
                output_dir=cell_dir,
                prefix=prefix,
            )


if __name__ == "__main__":
    main()
