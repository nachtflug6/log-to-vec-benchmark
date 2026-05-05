"""Step 4: Compute trace metrics for all (window_size, model) cells.

Runs metrics twice per cell: program_step_id labels and op_state_id labels.

Usage:
  python 04_compute_metrics.py --data_dir ../results/data \
      --embeddings_dir ../results/embeddings --output_dir ../results/metrics
"""
from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path
from typing import Dict, List

import numpy as np

_HERE    = Path(__file__).resolve().parent
_RQ5_SRC = _HERE.parent / "src"
_RQ4_SRC = _HERE.parent.parent / "rq4_cnc_benchmark" / "src"
_RQ2_SRC = _HERE.parent.parent / "rq2_trace_comparison" / "src"
for _p in [str(_RQ5_SRC), str(_RQ4_SRC), str(_RQ2_SRC)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from rq5.config.sweep import SCENARIO_IDS, MODELS, WINDOW_SIZES, get_scenario
from rq2.evaluation.trace_metrics import compute_all_metrics

LABEL_PAIRS = [("program_step_id", "step"), ("op_state_id", "opstate")]


def _build_metric_inputs(emb_data: dict, label_key: str):
    embeddings = emb_data["embeddings"]
    mode_ids   = emb_data[label_key]
    traj_ids   = emb_data["trajectory_id"]
    win_starts = emb_data["window_start"]

    per_traj_embs, per_traj_modes, per_traj_segs = [], [], {}
    for tid in np.unique(traj_ids):
        mask     = traj_ids == tid
        t_emb    = embeddings[mask]
        t_modes  = mode_ids[mask]
        idx_sort = np.argsort(win_starts[mask])
        t_emb, t_modes = t_emb[idx_sort], t_modes[idx_sort]
        per_traj_embs.append(t_emb)
        per_traj_modes.append(t_modes)
        segs: Dict[int, List] = {}
        cur, seg_start = t_modes[0], 0
        for i in range(1, len(t_modes)):
            if t_modes[i] != cur:
                s = t_emb[seg_start:i]
                if len(s) >= 3:
                    segs.setdefault(int(cur), []).append(s)
                cur, seg_start = t_modes[i], i
        s = t_emb[seg_start:]
        if len(s) >= 3:
            segs.setdefault(int(cur), []).append(s)
        for m, sl in segs.items():
            per_traj_segs.setdefault(m, []).extend(sl)

    all_modes = mode_ids
    cps = [i for i in range(1, len(all_modes)) if all_modes[i] != all_modes[i-1]]
    return embeddings, all_modes, cps, per_traj_embs, per_traj_modes, per_traj_segs


def compute_cell(sid, model, emb_dir, out_dir, force):
    emb_path = emb_dir / f"{sid}_{model}.npz"
    if not emb_path.exists():
        raise FileNotFoundError(emb_path)
    emb_data = {k: v for k, v in np.load(emb_path).items()}
    cfg      = get_scenario(sid)

    for label_key, suffix in LABEL_PAIRS:
        out_path = out_dir / f"{sid}_{model}_{suffix}.json"
        if out_path.exists() and not force:
            continue
        embs, modes, cps, pt_embs, pt_modes, pt_segs = _build_metric_inputs(emb_data, label_key)
        metrics = compute_all_metrics(
            embeddings=embs, mode_ids=modes, change_point_windows=cps,
            per_trajectory_embeddings=pt_embs, per_trajectory_mode_ids=pt_modes,
            per_trajectory_segments=pt_segs,
        )
        result = {"scenario_id": sid, "model": model, "label": suffix,
                  "window_length": cfg.window_length, "stride": cfg.stride}
        result.update(metrics)
        out_path.write_text(json.dumps(result, indent=2))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir",       default="../results/data")
    p.add_argument("--embeddings_dir", default="../results/embeddings")
    p.add_argument("--output_dir",     default="../results/metrics")
    p.add_argument("--scenarios",      nargs="+", default=None)
    p.add_argument("--models",         nargs="+", default=None)
    p.add_argument("--force",          action="store_true")
    return p.parse_args()


def main():
    args    = parse_args()
    emb_dir = Path(args.embeddings_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ids     = args.scenarios or SCENARIO_IDS
    models  = args.models    or MODELS
    done = failed = 0
    t0 = time.time()

    for sid in ids:
        for model in models:
            try:
                t1 = time.time()
                compute_cell(sid, model, emb_dir, out_dir, args.force)
                step_path = out_dir / f"{sid}_{model}_step.json"
                if step_path.exists():
                    d = json.loads(step_path.read_text())
                    print(f"  [{sid:20s} × {model:6s}]  ws={d['window_length']:3d}  "
                          f"MSI={d.get('mode_separability_index', float('nan')):.2f}  "
                          f"sil={d.get('silhouette_score', float('nan')):.3f}  "
                          f"{time.time()-t1:.1f}s")
                done += 1
            except FileNotFoundError as e:
                print(f"  [miss] {sid} × {model}: {e}")
                failed += 1
            except Exception as e:
                print(f"  [err ] {sid} × {model}: {e}")
                failed += 1

    print(f"\nDone {done}  Failed {failed}  ({time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
