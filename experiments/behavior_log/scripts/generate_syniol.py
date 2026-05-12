"""Generate SynIOL synthetic industrial operational logs."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from behavior_log.generation import SynIOLGenerator
from behavior_log.utils.io import load_yaml

SYNIOL_CONFIGS = {
    "syniol_clean",
    "syniol_moderate_noise",
    "syniol_high_noise",
}


def main() -> None:
    if len(sys.argv) != 2 or sys.argv[1] not in SYNIOL_CONFIGS:
        valid = ", ".join(sorted(SYNIOL_CONFIGS))
        raise SystemExit(f"Usage: python generate_syniol.py <{valid}>")
    config_name = sys.argv[1]
    cfg = load_yaml(ROOT / "configs" / "datasets" / f"{config_name}.yaml")
    output_dir = Path(cfg["raw_output_dir"])
    if not output_dir.is_absolute():
        output_dir = REPO_ROOT / output_dir
    generator = SynIOLGenerator(seed=int(cfg["seed"]))

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
        output_dir=output_dir,
        output_name=cfg["raw_output_name"],
    )
    print(f"Generated {len(dataset)} events across {dataset['trajectory_id'].nunique()} trajectories")
    print(f"Observable logs: {saved['observable_csv_path']}")
    print(f"Hidden labels: {saved['labels_csv_path']}")
    print(f"Full logs: {saved['full_csv_path']}")
    print(f"Manifest: {saved['metadata_path']}")
    print(f"Description: {saved['description_path']}")


if __name__ == "__main__":
    main()
