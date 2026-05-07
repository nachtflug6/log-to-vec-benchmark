"""Step 1: Generate all RQ6 scenario datasets.

Generates 8 scenarios across two axes:
  Axis A — signal type: sawtooth, damped sine, linear chirp (vs sine anchor ratio_0step)
  Axis B — channel ratio: 0–4 step channels out of 4 (ratio_0step … ratio_4step)

Run directly on the login node (no GPU needed, <1 min total).

Usage:
  python 01_generate_scenarios.py --output_dir ../results/data
  python 01_generate_scenarios.py --output_dir ../results/data --scenarios sig_sawtooth
  python 01_generate_scenarios.py --output_dir ../results/data --force
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_SRC = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(_SRC))

from rq6.config.scenarios import SCENARIOS, SCENARIO_IDS, get_scenario
from rq6.generation.rq6_generator import generate_rq6_dataset


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--output_dir", type=str, default="../results/data")
    p.add_argument("--scenarios", nargs="+", default=None)
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)

    scenario_ids = args.scenarios if args.scenarios else SCENARIO_IDS
    for sid in scenario_ids:
        get_scenario(sid)  # validate

    print(f"Generating {len(scenario_ids)} scenario(s) → {output_dir}")
    t0_total = time.time()

    import numpy as np
    for sid in scenario_ids:
        cfg = get_scenario(sid)
        dest = output_dir / sid
        marker = dest / "windows.npz"

        if marker.exists() and not args.force:
            N = np.load(marker)["X"].shape[0]
            print(f"  [skip] {sid:20s}  (N={N} windows already exist)")
            continue

        t0 = time.time()
        generate_rq6_dataset(cfg, dest)

        npz = np.load(marker)
        N   = npz["X"].shape[0]
        obs = float(npz["mask"].mean())
        print(
            f"  [done] {sid:20s}  N={N:5d} windows  "
            f"observed={obs:.2%}  {time.time()-t0:.1f}s"
        )

    print(f"\nAll done in {time.time()-t0_total:.1f}s")


if __name__ == "__main__":
    main()
