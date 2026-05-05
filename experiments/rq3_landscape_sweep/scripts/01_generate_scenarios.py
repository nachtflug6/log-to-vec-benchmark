"""Step 1: Generate all RQ3 scenario datasets.

Run directly on the login node (no GPU needed, ~2 min total).

Usage:
  python 01_generate_scenarios.py --output_dir ../results/data
  python 01_generate_scenarios.py --output_dir ../results/data --scenarios baseline sig_step
  python 01_generate_scenarios.py --output_dir ../results/data --force
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Make rq3 importable
_SRC = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(_SRC))

from rq3.config.scenarios import SCENARIOS, SCENARIO_IDS, get_scenario
from rq3.generation.rq3_generator import generate_rq3_dataset


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--output_dir", type=str, default="../results/data")
    p.add_argument(
        "--scenarios", nargs="+", default=None,
        help="Subset of scenario IDs to generate (default: all)",
    )
    p.add_argument("--force", action="store_true", help="Regenerate even if output exists")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)

    scenario_ids = args.scenarios if args.scenarios else SCENARIO_IDS
    # Validate
    for sid in scenario_ids:
        get_scenario(sid)  # raises KeyError if unknown

    print(f"Generating {len(scenario_ids)} scenario(s) → {output_dir}")
    t0_total = time.time()

    for sid in scenario_ids:
        cfg = get_scenario(sid)
        dest = output_dir / sid
        marker = dest / "windows.npz"

        if marker.exists() and not args.force:
            data = dict(zip(["N"], [str(__import__("numpy").load(marker)["X"].shape[0])]))
            print(f"  [skip] {sid:20s}  (N={data['N']} windows already exist)")
            continue

        t0 = time.time()
        generate_rq3_dataset(cfg, dest)
        elapsed = time.time() - t0

        npz = __import__("numpy").load(marker)
        N = npz["X"].shape[0]
        obs = float(npz["mask"].mean())
        print(
            f"  [done] {sid:20s}  N={N:5d} windows  "
            f"observed={obs:.2%}  {elapsed:.1f}s"
        )

    print(f"\nAll done in {time.time() - t0_total:.1f}s")


if __name__ == "__main__":
    main()
