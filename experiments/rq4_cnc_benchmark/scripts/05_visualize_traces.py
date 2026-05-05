"""Step 5: Generate worm-plot suites for all RQ4 (scenario, model) cells.

Produces both PCA and UMAP/t-SNE plots, coloured by program_step_id.
A second set coloured by op_state_id is saved with the _opstate suffix.

Run on login node.

Usage:
  python 05_visualize_traces.py --embeddings_dir ../results/embeddings \
      --output_dir ../results/plots [--trajectory_id 0]
"""
from __future__ import annotations

import argparse, sys, time
from pathlib import Path

import numpy as np

_HERE    = Path(__file__).resolve().parent
_RQ4_SRC = _HERE.parent / "src"
_RQ2_SRC = _HERE.parent.parent / "rq2_trace_comparison" / "src"
for _p in [str(_RQ4_SRC), str(_RQ2_SRC)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from rq4.config.scenarios import SCENARIO_IDS
from rq2.visualization.worm_plots import save_all_plots

MODELS = ["fft", "moment"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--embeddings_dir", default="../results/embeddings")
    p.add_argument("--output_dir",     default="../results/plots")
    p.add_argument("--scenarios",      nargs="+", default=None)
    p.add_argument("--models",         nargs="+", default=None)
    p.add_argument("--trajectory_id",  type=int,  default=0)
    p.add_argument("--force",          action="store_true")
    return p.parse_args()


def _plots_for_label(emb_data, traj_id_req, label_key, label_suffix, out_dir, prefix, force):
    traj_ids   = emb_data["trajectory_id"]
    win_starts = emb_data["window_start"]
    embeddings = emb_data["embeddings"]
    mode_ids   = emb_data[label_key]

    mask = traj_ids == traj_id_req
    if not mask.any():
        mask = traj_ids == np.unique(traj_ids)[0]

    t_emb   = embeddings[mask]
    t_modes = mode_ids[mask]
    t_starts = win_starts[mask]
    idx_sort = np.argsort(t_starts)
    t_emb, t_modes = t_emb[idx_sort], t_modes[idx_sort]

    cps = [i for i in range(1, len(t_modes)) if t_modes[i] != t_modes[i-1]]

    cell_dir = out_dir / label_suffix
    if (cell_dir / f"{prefix}_{label_suffix}_worm.png").exists() and not force:
        return

    cell_dir.mkdir(parents=True, exist_ok=True)
    save_all_plots(t_emb, t_modes, cps, cell_dir, prefix=f"{prefix}_{label_suffix}")


def main():
    args    = parse_args()
    emb_dir = Path(args.embeddings_dir)
    out_dir = Path(args.output_dir)
    ids     = args.scenarios or SCENARIO_IDS
    models  = args.models    or MODELS

    done = skipped = failed = 0
    t0 = time.time()

    for sid in ids:
        for model in models:
            emb_path = emb_dir / f"{sid}_{model}.npz"
            if not emb_path.exists():
                failed += 1
                continue
            try:
                emb_data = {k: v for k, v in np.load(emb_path).items()}
                base_dir = out_dir / sid / model

                _plots_for_label(emb_data, args.trajectory_id,
                                 "program_step_id", "step",   base_dir, f"{sid}_{model}", args.force)
                _plots_for_label(emb_data, args.trajectory_id,
                                 "op_state_id",     "opstate", base_dir, f"{sid}_{model}", args.force)
                print(f"  [done] {sid:30s} × {model}")
                done += 1
            except Exception as e:
                print(f"  [err ] {sid} × {model}: {e}")
                failed += 1

    print(f"\nDone {done}  Failed {failed}  ({time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
