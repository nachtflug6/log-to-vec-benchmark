"""Step 1: Generate CNC datasets for all RQ4 scenarios.

Usage (from repo root or on Alvis):
  python experiments/rq4_cnc_benchmark/scripts/01_generate_scenarios.py \
      --output_dir experiments/rq4_cnc_benchmark/results/data
"""
from __future__ import annotations

import argparse, sys, time
from pathlib import Path

_HERE    = Path(__file__).resolve().parent
_RQ4_SRC = _HERE.parent / "src"
_RQ2_SRC = _HERE.parent.parent / "rq2_trace_comparison" / "src"
for _p in [str(_RQ4_SRC), str(_RQ2_SRC)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from rq4.config.scenarios import SCENARIOS, SCENARIO_IDS, get_scenario
from rq4.generation.cnc_generator import generate_cnc_dataset


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--output_dir", default="../results/data")
    p.add_argument("--scenarios", nargs="+", default=None)
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def main():
    args    = parse_args()
    out_dir = Path(args.output_dir)
    ids     = args.scenarios or SCENARIO_IDS

    print(f"Generating {len(ids)} RQ4 CNC scenario(s) → {out_dir}")
    t0 = time.time()

    for sid in ids:
        dest = out_dir / sid
        if (dest / "windows.npz").exists() and not args.force:
            print(f"  [skip] {sid}  (already exists)")
            continue
        cfg = get_scenario(sid)
        t1  = time.time()
        generate_cnc_dataset(cfg, dest)
        import numpy as np
        meta_path = dest / "metadata.json"
        import json
        meta = json.loads(meta_path.read_text())
        print(f"  [done] {sid:30s}  {meta['num_windows']:5d} windows  "
              f"{meta['num_runs']} runs  {time.time()-t1:.1f}s")

    print(f"\nAll done in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
