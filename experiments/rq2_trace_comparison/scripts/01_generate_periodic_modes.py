"""Step 1: Generate all three periodic-mode datasets (P1, P2, P3).

Usage:
  python 01_generate_periodic_modes.py --output_dir experiments/rq2_trace_comparison/data
  python 01_generate_periodic_modes.py --output_dir ... --num_trajectories 4 --dry_run
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from rq2.generation.periodic_mode_generator import PeriodicModeConfig, generate_dataset


PROBLEMS = ["p1_simple_1d", "p2_multichannel", "p3_hard_noisy"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--output_dir", type=str, default="data")
    p.add_argument("--num_trajectories", type=int, default=20)
    p.add_argument("--periods_per_segment", type=int, default=5)
    p.add_argument("--window_length", type=int, default=48)
    p.add_argument("--stride", type=int, default=12)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--problems", nargs="+", default=PROBLEMS, choices=PROBLEMS)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_root = Path(args.output_dir)

    for problem in args.problems:
        cfg = PeriodicModeConfig(
            problem=problem,
            seed=args.seed,
            num_trajectories=args.num_trajectories,
            periods_per_segment=args.periods_per_segment,
            window_length=args.window_length,
            stride=args.stride,
        )
        problem_dir = out_root / problem
        generate_dataset(cfg, output_dir=problem_dir)
        print(f"  -> saved to {problem_dir}")


if __name__ == "__main__":
    main()
