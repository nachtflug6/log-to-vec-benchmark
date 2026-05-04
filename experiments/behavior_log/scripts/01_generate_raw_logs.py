"""Stage 01: generate behavior-driven raw logs."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from behavior_log.generation.raw_log_generator import BehaviorDrivenLogGenerator
from behavior_log.utils.io import load_yaml, save_json


def main() -> None:
    cfg = load_yaml(ROOT / "configs" / "datasets" / "default.yaml")
    generator = BehaviorDrivenLogGenerator(seed=int(cfg["seed"]))

    dataset = generator.generate_dataset(
        num_trajectories=int(cfg["num_trajectories"]),
        events_per_trajectory=int(cfg["events_per_trajectory"]),
        behavior_ids=cfg["behavior_ids"],
        behavior_probs=cfg["behavior_probs"],
        trajectory_spacing_seconds=float(cfg["trajectory_spacing_seconds"]),
        step_interval_seconds=(float(cfg["min_step_seconds"]), float(cfg["max_step_seconds"])),
        noise_event_prob=float(cfg["noise_event_prob"]),
    )
    metadata = generator.build_metadata(
        dataset=dataset,
        requested_num_trajectories=int(cfg["num_trajectories"]),
        events_per_trajectory=int(cfg["events_per_trajectory"]),
        behavior_ids=cfg["behavior_ids"],
        behavior_probs=cfg["behavior_probs"],
    )
    saved = generator.save_dataset(
        dataset=dataset,
        metadata=metadata,
        output_dir=cfg["raw_output_dir"],
        output_name=cfg["raw_output_name"],
    )
    save_json(metadata, Path(cfg["raw_output_dir"]) / "dataset_manifest.json")

    print(f"Generated {len(dataset)} events across {dataset['trajectory_id'].nunique()} trajectories")
    print(f"Raw logs: {saved['csv_path']}")
    print(f"Metadata: {saved['metadata_path']}")


if __name__ == "__main__":
    main()
