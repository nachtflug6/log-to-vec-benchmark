from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "experiments" / "behavior_log" / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from behavior_log.generation.generator_v4 import LatentBehaviorLogGeneratorV4


def test_v4_generator_produces_requested_shape() -> None:
    generator = LatentBehaviorLogGeneratorV4(seed=7)
    dataset = generator.generate_dataset(
        num_trajectories=3,
        events_per_trajectory=120,
        segment_length_range=(24, 36),
        transition_length_range=(6, 8),
    )

    assert len(dataset) == 360
    counts = dataset.groupby("trajectory_id").size()
    assert counts.tolist() == [120, 120, 120]
    assert {"risk_level", "load_level", "recovery_progress", "is_transition", "segment_id"}.issubset(dataset.columns)


def test_v4_generator_emits_real_transition_events() -> None:
    generator = LatentBehaviorLogGeneratorV4(seed=11)
    dataset = generator.generate_dataset(
        num_trajectories=2,
        events_per_trajectory=160,
        segment_length_range=(32, 48),
        transition_length_range=(8, 10),
    )

    transition_rows = dataset[dataset["is_transition"]]
    assert not transition_rows.empty
    assert transition_rows["source_behavior_id"].ne(transition_rows["target_behavior_id"]).any()
    assert transition_rows["transition_progress"].between(0.0, 1.0).all()


def test_v4_generator_uses_shared_event_vocabulary_across_components() -> None:
    generator = LatentBehaviorLogGeneratorV4(seed=13)
    dataset = generator.generate_dataset(
        num_trajectories=6,
        events_per_trajectory=120,
        segment_length_range=(24, 36),
        transition_length_range=(6, 8),
    )

    status_components = dataset.loc[dataset["event_type"] == "STATUS_CHECK", "component"].unique().tolist()
    error_components = dataset.loc[dataset["event_type"] == "ERROR_DETECTED", "component"].unique().tolist()

    assert len(status_components) >= 3
    assert len(error_components) >= 2


def test_v4_metadata_build_includes_catalog_and_transition_count() -> None:
    generator = LatentBehaviorLogGeneratorV4(seed=17)
    dataset = generator.generate_dataset(
        num_trajectories=2,
        events_per_trajectory=100,
        segment_length_range=(20, 30),
        transition_length_range=(6, 8),
    )

    metadata = generator.build_metadata(
        dataset=dataset,
        requested_num_trajectories=2,
        events_per_trajectory=100,
        behavior_ids=[0, 1, 2, 3],
        behavior_probs=[0.25, 0.25, 0.25, 0.25],
    )

    assert metadata["generated_num_events"] == len(dataset)
    assert metadata["generated_num_trajectories"] == 2
    assert metadata["transition_event_count"] > 0
    assert len(metadata["catalog"]) == 4
