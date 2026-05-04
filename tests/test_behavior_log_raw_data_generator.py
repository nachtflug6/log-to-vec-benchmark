"""Tests for the behavior_log raw generator and preprocessing flow."""

from pathlib import Path
import sys

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.behavior_log.src.behavior_log.generation.raw_log_generator import (  # noqa: E402
    BEHAVIOR_CATALOG,
    BehaviorDrivenLogGenerator,
)
from experiments.behavior_log.src.behavior_log.preprocessing.event_preprocessor import (  # noqa: E402
    EventLogPreprocessor,
)


def test_generate_cycle_matches_catalog_pattern():
    """A chosen cycle should expand into its exact canonical pattern."""

    generator = BehaviorDrivenLogGenerator(seed=7)
    spec = BEHAVIOR_CATALOG[1]
    variant = spec.variants[0]

    rows, _ = generator.generate_cycle(
        behavior_id=spec.behavior_id,
        trajectory_id=0,
        cycle_id=0,
        start_time=pd.Timestamp("2026-01-01T00:00:00").to_pydatetime(),
        step_interval_seconds=(1.0, 1.0),
        variant_index=0,
    )

    assert [row["event_type"] for row in rows] == list(variant.log_pattern)
    assert all(row["behavior_id"] == spec.behavior_id for row in rows)
    assert all(row["cycle_id"] == 0 for row in rows)
    assert all(row["cycle_behavior_id"] == spec.behavior_id for row in rows)
    assert all("sensor_value" in row for row in rows)
    assert all("control_value" in row for row in rows)
    assert all("is_transition" in row for row in rows)
    assert all(row["is_transition"] is False for row in rows)


def test_generate_dataset_single_behavior_creates_expected_cycles():
    """Sampling from one behavior should only produce that behavior across all cycles."""

    generator = BehaviorDrivenLogGenerator(seed=11)
    dataset = generator.generate_dataset(
        num_trajectories=3,
        events_per_trajectory=24,
        behavior_ids=[2],
        behavior_probs=[1.0],
        step_interval_seconds=(1.0, 1.0),
        noise_event_prob=0.0,
    )

    expected_events = {
        event_type
        for variant in BEHAVIOR_CATALOG[2].variants
        for event_type in variant.log_pattern
    }
    assert dataset["trajectory_id"].nunique() == 3
    assert len(dataset) == 3 * 24
    assert set(dataset["behavior_id"]) == {2}
    assert dataset["cycle_id"].nunique() == 3

    for _, trajectory_df in dataset.groupby("trajectory_id"):
        assert set(trajectory_df["event_type"].unique()).issubset(expected_events)


def test_behavior_prob_length_validation():
    """Probability vector must align with selected behavior ids."""

    generator = BehaviorDrivenLogGenerator(seed=1)

    with pytest.raises(ValueError, match="length must match"):
        generator.generate_dataset(
            num_trajectories=5,
            events_per_trajectory=24,
            behavior_ids=[0, 1],
            behavior_probs=[1.0],
            noise_event_prob=0.0,
        )


def test_metadata_reports_requested_trajectory_count():
    """Metadata should expose the trajectory-level ground truth summary."""

    generator = BehaviorDrivenLogGenerator(seed=5)
    dataset = generator.generate_dataset(
        num_trajectories=4,
        events_per_trajectory=24,
        behavior_ids=[3],
        behavior_probs=[1.0],
        step_interval_seconds=(1.0, 1.0),
        noise_event_prob=0.0,
    )
    metadata = generator.build_metadata(
        dataset=dataset,
        requested_num_trajectories=4,
        events_per_trajectory=24,
        behavior_ids=[3],
        behavior_probs=[1.0],
    )

    assert metadata["requested_num_trajectories"] == 4
    assert metadata["events_per_trajectory"] == 24
    assert metadata["cycles_per_trajectory"] == 3
    assert metadata["generated_num_trajectories"] == 4
    assert metadata["generated_num_events"] == len(dataset)
    assert metadata["behavior_trajectory_counts"][0]["behavior_id"] == 3
    assert metadata["behavior_trajectory_counts"][0]["trajectory_count"] == 4
    assert metadata["behavior_cycle_counts"][0]["cycle_count"] == 12


def test_event_preprocessor_creates_numeric_features():
    """The preprocessor should emit dense numeric event vectors without labels inside features."""

    generator = BehaviorDrivenLogGenerator(seed=3)
    dataset = generator.generate_dataset(
        num_trajectories=2,
        events_per_trajectory=24,
        behavior_ids=[0, 1],
        behavior_probs=[0.5, 0.5],
        step_interval_seconds=(1.0, 1.0),
        noise_event_prob=0.0,
    )

    preprocessor = EventLogPreprocessor()
    bundle = preprocessor.fit_transform(dataset)

    assert bundle.X.shape[0] == len(dataset)
    assert bundle.X.shape[1] == len(bundle.feature_names)
    assert bundle.X.dtype.kind == "f"
    assert "behavior_id" not in bundle.feature_names.tolist()
    assert bundle.feature_names.tolist() == [
        "event_type_id",
        "component_id",
        "severity_id",
        "state_id",
        "sensor_value",
        "control_value",
    ]


def test_noise_injection_preserves_target_event_count():
    """Noise insertion should still keep the total trajectory length at the requested size."""

    generator = BehaviorDrivenLogGenerator(seed=9)
    dataset = generator.generate_dataset(
        num_trajectories=2,
        events_per_trajectory=200,
        behavior_ids=[0, 1, 2, 3],
        behavior_probs=[0.25, 0.25, 0.25, 0.25],
        noise_event_prob=0.15,
    )

    assert len(dataset) == 400
    assert dataset.groupby("trajectory_id").size().tolist() == [200, 200]
