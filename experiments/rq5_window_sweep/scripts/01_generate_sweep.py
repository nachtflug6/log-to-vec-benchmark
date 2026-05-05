"""Step 1: Generate CNC datasets for all window-size variants.

Usage:
  python 01_generate_sweep.py --output_dir ../results/data
"""
from __future__ import annotations

import argparse, sys, time
from pathlib import Path

_HERE    = Path(__file__).resolve().parent
_RQ5_SRC = _HERE.parent / "src"
_RQ4_SRC = _HERE.parent.parent / "rq4_cnc_benchmark" / "src"
for _p in [str(_RQ5_SRC), str(_RQ4_SRC)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from rq4.generation.cnc_generator import generate_cnc_dataset
from rq5.config.sweep import SWEEP_SCENARIOS


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--output_dir", default="../results/data")
    p.add_argument("--force",      action="store_true")
    return p.parse_args()


def main():
    args     = parse_args()
    out_dir  = Path(args.output_dir)
    t0       = time.time()
    print(f"Generating {len(SWEEP_SCENARIOS)} window-size variant(s) → {out_dir}")

    for cfg in SWEEP_SCENARIOS:
        dest = out_dir / cfg.scenario_id
        if (dest / "windows.npz").exists() and not args.force:
            print(f"  [skip] {cfg.scenario_id}  (already exists)")
            continue
        t1 = time.time()
        generate_cnc_dataset(cfg, dest)
        import numpy as np, json as _json
        meta = _json.loads((dest / "metadata.json").read_text())
        print(f"  [done] {cfg.scenario_id:20s}  ws={cfg.window_length:3d}  "
              f"stride={cfg.stride:2d}  {meta['num_windows']:5d} windows  "
              f"{time.time()-t1:.1f}s")

    print(f"\nAll done in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
