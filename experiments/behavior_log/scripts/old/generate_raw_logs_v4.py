"""Stage 01 (v4): generate latent-dynamic behavior raw logs."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from behavior_log.generation.generator_v4 import LatentBehaviorLogGeneratorV4
from behavior_log.utils.io import load_yaml, save_json


def main() -> None:
    config_name = sys.argv[1] if len(sys.argv) > 1 else "v4"
    cfg = load_yaml(ROOT / "configs" / "datasets" / f"{config_name}.yaml")
    generator = LatentBehaviorLogGeneratorV4(seed=int(cfg["seed"]))

    dataset = generator.generate_dataset(
        num_trajectories=int(cfg["num_trajectories"]),
        events_per_trajectory=int(cfg["events_per_trajectory"]),
        behavior_ids=cfg["behavior_ids"],
        behavior_probs=cfg["behavior_probs"],
        trajectory_spacing_seconds=float(cfg["trajectory_spacing_seconds"]),
        step_interval_seconds=(float(cfg["min_step_seconds"]), float(cfg["max_step_seconds"])),
        segment_length_range=(int(cfg["segment_min_events"]), int(cfg["segment_max_events"])),
        transition_length_range=(int(cfg["transition_min_events"]), int(cfg["transition_max_events"])),
        irrelevant_noise_prob=float(cfg["irrelevant_noise_prob"]),
        severity_noise_prob=float(cfg["severity_noise_prob"]),
        numeric_noise_std=float(cfg["numeric_noise_std"]),
        component_noise_prob=float(cfg["component_noise_prob"]),
        dominant_behavior_threshold=float(cfg["dominant_behavior_threshold"]),
        state_delay_steps_range=(int(cfg["state_delay_min_steps"]), int(cfg["state_delay_max_steps"])),
        state_noise_prob=float(cfg["state_noise_prob"]),
        state_update_prob=float(cfg["state_update_prob"]),
        false_negative_prob=float(cfg["false_negative_prob"]),
        false_positive_prob=float(cfg["false_positive_prob"]),
        critical_prob_cap=float(cfg["critical_prob_cap"]),
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
