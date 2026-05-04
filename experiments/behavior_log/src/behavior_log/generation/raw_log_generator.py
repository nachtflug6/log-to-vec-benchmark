"""Behavior-driven raw log generator v3: hard behavior sequences."""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import pandas as pd


@dataclass(frozen=True)
class BehaviorVariant:
    """One long-range dependency variant of a behavior."""

    variant_id: int
    variant_name: str
    log_pattern: tuple[str, ...]
    state_pattern: tuple[str, ...]


@dataclass(frozen=True)
class BehaviorSpec:
    """Defines one behavior family and its variants."""

    behavior_id: int
    behavior_name: str
    variants: tuple[BehaviorVariant, ...]


BEHAVIOR_CATALOG: tuple[BehaviorSpec, ...] = (
    BehaviorSpec(
        behavior_id=0,
        behavior_name="Successful Recovery",
        variants=(
            BehaviorVariant(
                variant_id=0,
                variant_name="successful_recovery_a",
                log_pattern=(
                    "SENSOR_READ",
                    "ALARM_TRIGGER",
                    "ACTUATOR_CMD",
                    "SENSOR_READ",
                    "WATCHDOG_TICK",
                    "ALARM_TRIGGER",
                    "WATCHDOG_TICK",
                    "ALARM_CLEAR",
                ),
                state_pattern=(
                    "RUNNING",
                    "WARNING",
                    "RECOVERY",
                    "RECOVERY",
                    "RECOVERY",
                    "RECOVERY",
                    "RECOVERY",
                    "RUNNING",
                ),
            ),
            BehaviorVariant(
                variant_id=1,
                variant_name="successful_recovery_b",
                log_pattern=(
                    "SENSOR_READ",
                    "ALARM_TRIGGER",
                    "WATCHDOG_TICK",
                    "ACTUATOR_CMD",
                    "SENSOR_READ",
                    "ALARM_TRIGGER",
                    "WATCHDOG_TICK",
                    "ALARM_CLEAR",
                ),
                state_pattern=(
                    "RUNNING",
                    "WARNING",
                    "WARNING",
                    "RECOVERY",
                    "RECOVERY",
                    "RECOVERY",
                    "RECOVERY",
                    "RUNNING",
                ),
            ),
            BehaviorVariant(
                variant_id=2,
                variant_name="successful_recovery_c",
                log_pattern=(
                    "WATCHDOG_TICK",
                    "SENSOR_READ",
                    "ALARM_TRIGGER",
                    "ACTUATOR_CMD",
                    "SENSOR_READ",
                    "WATCHDOG_TICK",
                    "ALARM_TRIGGER",
                    "ALARM_CLEAR",
                ),
                state_pattern=(
                    "RUNNING",
                    "RUNNING",
                    "WARNING",
                    "RECOVERY",
                    "RECOVERY",
                    "RECOVERY",
                    "RECOVERY",
                    "RUNNING",
                ),
            ),
        ),
    ),
    BehaviorSpec(
        behavior_id=1,
        behavior_name="Failed Recovery",
        variants=(
            BehaviorVariant(
                variant_id=0,
                variant_name="failed_recovery_a",
                log_pattern=(
                    "SENSOR_READ",
                    "ALARM_TRIGGER",
                    "ACTUATOR_CMD",
                    "SENSOR_READ",
                    "WATCHDOG_TICK",
                    "ALARM_CLEAR",
                    "WATCHDOG_TICK",
                    "ALARM_TRIGGER",
                ),
                state_pattern=(
                    "RUNNING",
                    "WARNING",
                    "RECOVERY",
                    "RECOVERY",
                    "RECOVERY",
                    "RECOVERY",
                    "RECOVERY",
                    "WARNING",
                ),
            ),
            BehaviorVariant(
                variant_id=1,
                variant_name="failed_recovery_b",
                log_pattern=(
                    "SENSOR_READ",
                    "ALARM_TRIGGER",
                    "WATCHDOG_TICK",
                    "ACTUATOR_CMD",
                    "SENSOR_READ",
                    "ALARM_CLEAR",
                    "WATCHDOG_TICK",
                    "ALARM_TRIGGER",
                ),
                state_pattern=(
                    "RUNNING",
                    "WARNING",
                    "WARNING",
                    "RECOVERY",
                    "RECOVERY",
                    "RECOVERY",
                    "RECOVERY",
                    "WARNING",
                ),
            ),
            BehaviorVariant(
                variant_id=2,
                variant_name="failed_recovery_c",
                log_pattern=(
                    "WATCHDOG_TICK",
                    "SENSOR_READ",
                    "ALARM_TRIGGER",
                    "ACTUATOR_CMD",
                    "SENSOR_READ",
                    "ALARM_CLEAR",
                    "WATCHDOG_TICK",
                    "ALARM_TRIGGER",
                ),
                state_pattern=(
                    "RUNNING",
                    "RUNNING",
                    "WARNING",
                    "RECOVERY",
                    "RECOVERY",
                    "RECOVERY",
                    "RECOVERY",
                    "WARNING",
                ),
            ),
        ),
    ),
    BehaviorSpec(
        behavior_id=2,
        behavior_name="Successful Communication Recovery",
        variants=(
            BehaviorVariant(
                variant_id=0,
                variant_name="successful_comm_a",
                log_pattern=(
                    "SENSOR_READ",
                    "COMMUNICATION_ERROR",
                    "WATCHDOG_TICK",
                    "SENSOR_READ",
                    "COMMUNICATION_OK",
                    "WATCHDOG_TICK",
                    "COMMUNICATION_ERROR",
                    "COMMUNICATION_OK",
                ),
                state_pattern=(
                    "RUNNING",
                    "WARNING",
                    "WARNING",
                    "WARNING",
                    "RECOVERY",
                    "RECOVERY",
                    "RECOVERY",
                    "RUNNING",
                ),
            ),
            BehaviorVariant(
                variant_id=1,
                variant_name="successful_comm_b",
                log_pattern=(
                    "WATCHDOG_TICK",
                    "SENSOR_READ",
                    "COMMUNICATION_ERROR",
                    "SENSOR_READ",
                    "WATCHDOG_TICK",
                    "COMMUNICATION_OK",
                    "COMMUNICATION_ERROR",
                    "COMMUNICATION_OK",
                ),
                state_pattern=(
                    "RUNNING",
                    "RUNNING",
                    "WARNING",
                    "WARNING",
                    "RECOVERY",
                    "RECOVERY",
                    "RECOVERY",
                    "RUNNING",
                ),
            ),
            BehaviorVariant(
                variant_id=2,
                variant_name="successful_comm_c",
                log_pattern=(
                    "SENSOR_READ",
                    "COMMUNICATION_ERROR",
                    "SENSOR_READ",
                    "WATCHDOG_TICK",
                    "COMMUNICATION_OK",
                    "COMMUNICATION_ERROR",
                    "WATCHDOG_TICK",
                    "COMMUNICATION_OK",
                ),
                state_pattern=(
                    "RUNNING",
                    "WARNING",
                    "WARNING",
                    "WARNING",
                    "RECOVERY",
                    "RECOVERY",
                    "RECOVERY",
                    "RUNNING",
                ),
            ),
        ),
    ),
    BehaviorSpec(
        behavior_id=3,
        behavior_name="Persistent Communication Failure",
        variants=(
            BehaviorVariant(
                variant_id=0,
                variant_name="persistent_comm_a",
                log_pattern=(
                    "SENSOR_READ",
                    "COMMUNICATION_ERROR",
                    "WATCHDOG_TICK",
                    "SENSOR_READ",
                    "COMMUNICATION_OK",
                    "WATCHDOG_TICK",
                    "COMMUNICATION_OK",
                    "COMMUNICATION_ERROR",
                ),
                state_pattern=(
                    "RUNNING",
                    "WARNING",
                    "WARNING",
                    "WARNING",
                    "RECOVERY",
                    "RECOVERY",
                    "RECOVERY",
                    "WARNING",
                ),
            ),
            BehaviorVariant(
                variant_id=1,
                variant_name="persistent_comm_b",
                log_pattern=(
                    "WATCHDOG_TICK",
                    "SENSOR_READ",
                    "COMMUNICATION_ERROR",
                    "SENSOR_READ",
                    "WATCHDOG_TICK",
                    "COMMUNICATION_OK",
                    "COMMUNICATION_OK",
                    "COMMUNICATION_ERROR",
                ),
                state_pattern=(
                    "RUNNING",
                    "RUNNING",
                    "WARNING",
                    "WARNING",
                    "RECOVERY",
                    "RECOVERY",
                    "RECOVERY",
                    "WARNING",
                ),
            ),
            BehaviorVariant(
                variant_id=2,
                variant_name="persistent_comm_c",
                log_pattern=(
                    "SENSOR_READ",
                    "COMMUNICATION_ERROR",
                    "SENSOR_READ",
                    "WATCHDOG_TICK",
                    "COMMUNICATION_OK",
                    "WATCHDOG_TICK",
                    "COMMUNICATION_OK",
                    "COMMUNICATION_ERROR",
                ),
                state_pattern=(
                    "RUNNING",
                    "WARNING",
                    "WARNING",
                    "WARNING",
                    "RECOVERY",
                    "RECOVERY",
                    "RECOVERY",
                    "WARNING",
                ),
            ),
        ),
    ),
)

BEHAVIOR_PROB_MAP: Dict[int, float] = {
    0: 0.25,
    1: 0.25,
    2: 0.25,
    3: 0.25,
}

NOISE_EVENT_POOL: tuple[str, ...] = (
    "WATCHDOG_TICK",
    "COMMUNICATION_OK",
    "SENSOR_READ",
)

COMPONENT_BY_EVENT: Dict[str, str] = {
    "SENSOR_READ": "SensorUnit",
    "ALARM_TRIGGER": "SensorUnit",
    "ALARM_CLEAR": "SensorUnit",
    "ACTUATOR_CMD": "ControlUnit",
    "WATCHDOG_TICK": "ControlUnit",
    "COMMUNICATION_OK": "NetworkUnit",
    "COMMUNICATION_ERROR": "NetworkUnit",
}

MESSAGE_BY_EVENT: Dict[str, str] = {
    "SENSOR_READ": "Sensor unit sampled the process while system state is {state}.",
    "ALARM_TRIGGER": "Sensor unit raised an alarm while system state is {state}.",
    "ALARM_CLEAR": "Sensor unit cleared the alarm while system state is {state}.",
    "ACTUATOR_CMD": "Control unit issued an actuator command while system state is {state}.",
    "WATCHDOG_TICK": "Control unit emitted a watchdog tick while system state is {state}.",
    "COMMUNICATION_OK": "Network unit reported healthy communication while system state is {state}.",
    "COMMUNICATION_ERROR": "Network unit reported a communication error while system state is {state}.",
}


def get_behavior_catalog() -> List[dict]:
    """Return the catalog as JSON-serializable rows."""

    return [asdict(spec) for spec in BEHAVIOR_CATALOG]


class BehaviorDrivenLogGenerator:
    """Generate trajectories whose labels depend on long-range outcomes."""

    def __init__(self, seed: int = 42) -> None:
        self.seed = seed
        self.rng = random.Random(seed)
        self.behaviors = {spec.behavior_id: spec for spec in BEHAVIOR_CATALOG}

    def generate_cycle(
        self,
        behavior_id: int,
        trajectory_id: int,
        cycle_id: int,
        start_time: datetime,
        step_interval_seconds: tuple[float, float] = (1.0, 3.0),
        variant_index: int | None = None,
    ) -> tuple[List[dict], datetime]:
        """Expand one sampled behavior into one long-range dependency cycle."""

        if behavior_id not in self.behaviors:
            raise ValueError(f"Unknown behavior_id: {behavior_id}")

        low, high = step_interval_seconds
        if low <= 0 or high < low:
            raise ValueError("step_interval_seconds must satisfy 0 < low <= high.")

        spec = self.behaviors[behavior_id]
        variant = spec.variants[variant_index] if variant_index is not None else self.rng.choice(spec.variants)
        timestamp = start_time
        rows: List[dict] = []

        for event_index, (event_type, state) in enumerate(zip(variant.log_pattern, variant.state_pattern)):
            rows.append(
                self._build_row(
                    timestamp=timestamp,
                    trajectory_id=trajectory_id,
                    cycle_id=cycle_id,
                    behavior_id=behavior_id,
                    event_type=event_type,
                    state=state,
                    event_index=event_index,
                    cycle_length=len(variant.log_pattern),
                )
            )
            timestamp += timedelta(seconds=self.rng.uniform(low, high))

        return rows, timestamp

    def generate_dataset(
        self,
        num_trajectories: int,
        events_per_trajectory: int,
        behavior_ids: Sequence[int] | None = None,
        behavior_probs: Sequence[float] | None = None,
        start_time: datetime | None = None,
        trajectory_spacing_seconds: float = 600.0,
        step_interval_seconds: tuple[float, float] = (1.0, 3.0),
        noise_event_prob: float = 0.15,
    ) -> pd.DataFrame:
        """Generate an entire raw log dataset."""

        cycle_length = 8
        if num_trajectories <= 0:
            raise ValueError("num_trajectories must be positive.")
        if events_per_trajectory <= 0:
            raise ValueError("events_per_trajectory must be positive.")
        if events_per_trajectory % cycle_length != 0:
            raise ValueError("events_per_trajectory must be divisible by 8 for v3.")
        if trajectory_spacing_seconds <= 0:
            raise ValueError("trajectory_spacing_seconds must be positive.")
        if not 0.0 <= noise_event_prob <= 0.5:
            raise ValueError("noise_event_prob must be in [0.0, 0.5].")

        if start_time is None:
            start_time = datetime(2026, 1, 1, 0, 0, 0)

        if behavior_ids is None:
            behavior_ids = sorted(self.behaviors)
        else:
            behavior_ids = list(behavior_ids)

        self._validate_behavior_ids(behavior_ids)
        weights = self._normalize_behavior_probs(behavior_ids, behavior_probs)

        base_cycles = max(1, int(round((events_per_trajectory * (1.0 - noise_event_prob)) / cycle_length)))
        base_event_count = base_cycles * cycle_length
        noise_events_per_trajectory = max(0, events_per_trajectory - base_event_count)

        rows: List[dict] = []
        current_start = start_time
        for trajectory_id in range(num_trajectories):
            trajectory_rows, _ = self.generate_trajectory(
                trajectory_id=trajectory_id,
                num_cycles=base_cycles,
                noise_events=noise_events_per_trajectory,
                behavior_ids=behavior_ids,
                behavior_probs=weights,
                start_time=current_start,
                step_interval_seconds=step_interval_seconds,
            )
            rows.extend(trajectory_rows)
            current_start += timedelta(seconds=trajectory_spacing_seconds)

        return pd.DataFrame(rows)

    def generate_trajectory(
        self,
        trajectory_id: int,
        num_cycles: int,
        noise_events: int,
        behavior_ids: Sequence[int],
        behavior_probs: Sequence[float],
        start_time: datetime,
        step_interval_seconds: tuple[float, float] = (1.0, 3.0),
    ) -> tuple[List[dict], datetime]:
        """Generate one trajectory as multiple sampled cycles plus inserted noise events."""

        if num_cycles <= 0:
            raise ValueError("num_cycles must be positive.")
        if noise_events < 0:
            raise ValueError("noise_events cannot be negative.")

        rows: List[dict] = []
        current_time = start_time
        cycle_length = 8
        gap_count = num_cycles * cycle_length + 1
        gap_noise_counts = self._sample_noise_distribution(noise_events, gap_count)
        gap_index = 0

        for cycle_id in range(num_cycles):
            behavior_id = self.rng.choices(behavior_ids, weights=behavior_probs, k=1)[0]
            spec = self.behaviors[behavior_id]
            variant = self.rng.choice(spec.variants)
            cycle_rows, current_time = self._expand_variant_rows(
                variant=variant,
                trajectory_id=trajectory_id,
                cycle_id=cycle_id,
                behavior_id=behavior_id,
                start_time=current_time,
                step_interval_seconds=step_interval_seconds,
                gap_noise_counts=gap_noise_counts,
                starting_gap_index=gap_index,
            )
            rows.extend(cycle_rows)
            gap_index += len(variant.log_pattern)

        for _ in range(gap_noise_counts[-1]):
            noise_event = self.rng.choice(NOISE_EVENT_POOL)
            rows.append(
                self._build_row(
                    timestamp=current_time,
                    trajectory_id=trajectory_id,
                    cycle_id=num_cycles - 1,
                    behavior_id=rows[-1]["behavior_id"],
                    event_type=noise_event,
                    state=rows[-1]["state"],
                    event_index=0,
                    cycle_length=cycle_length,
                )
            )
            current_time += timedelta(seconds=self.rng.uniform(*step_interval_seconds))

        return rows, current_time

    def build_metadata(
        self,
        dataset: pd.DataFrame,
        requested_num_trajectories: int,
        events_per_trajectory: int,
        behavior_ids: Sequence[int],
        behavior_probs: Sequence[float],
    ) -> dict:
        """Build a reproducibility summary."""

        behavior_counts = (
            dataset.groupby(["behavior_id"])["trajectory_id"]
            .nunique()
            .reset_index(name="trajectory_count")
            .sort_values("behavior_id")
        )
        behavior_counts["behavior_name"] = behavior_counts["behavior_id"].map(
            {spec.behavior_id: spec.behavior_name for spec in BEHAVIOR_CATALOG}
        )
        behavior_counts = behavior_counts[["behavior_id", "behavior_name", "trajectory_count"]]

        cycle_counts = (
            dataset[["trajectory_id", "cycle_id", "cycle_behavior_id"]]
            .drop_duplicates()
            .groupby("cycle_behavior_id")["cycle_id"]
            .count()
            .reset_index(name="cycle_count")
            .rename(columns={"cycle_behavior_id": "behavior_id"})
            .sort_values("behavior_id")
        )

        return {
            "seed": self.seed,
            "requested_num_trajectories": requested_num_trajectories,
            "events_per_trajectory": events_per_trajectory,
            "cycles_per_trajectory": int(dataset.groupby("trajectory_id")["cycle_id"].nunique().iloc[0]),
            "generated_num_events": int(len(dataset)),
            "generated_num_trajectories": int(dataset["trajectory_id"].nunique()),
            "selected_behavior_ids": list(behavior_ids),
            "selected_behavior_probs": list(behavior_probs),
            "behavior_trajectory_counts": behavior_counts.to_dict(orient="records"),
            "behavior_cycle_counts": cycle_counts.to_dict(orient="records"),
            "catalog": get_behavior_catalog(),
        }

    def save_dataset(self, dataset: pd.DataFrame, metadata: dict, output_dir: str | Path, output_name: str) -> dict:
        """Save CSV and metadata JSON."""

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        csv_path = output_dir / output_name
        metadata_path = output_dir / f"{Path(output_name).stem}_metadata.json"
        dataset.to_csv(csv_path, index=False)
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        return {"csv_path": str(csv_path), "metadata_path": str(metadata_path)}

    def _expand_variant_rows(
        self,
        variant: BehaviorVariant,
        trajectory_id: int,
        cycle_id: int,
        behavior_id: int,
        start_time: datetime,
        step_interval_seconds: tuple[float, float],
        gap_noise_counts: List[int],
        starting_gap_index: int,
    ) -> tuple[List[dict], datetime]:
        timestamp = start_time
        rows: List[dict] = []
        current_gap = starting_gap_index

        for event_index, (event_type, state) in enumerate(zip(variant.log_pattern, variant.state_pattern)):
            for _ in range(gap_noise_counts[current_gap]):
                noise_event = self.rng.choice(NOISE_EVENT_POOL)
                rows.append(
                    self._build_row(
                        timestamp=timestamp,
                        trajectory_id=trajectory_id,
                        cycle_id=cycle_id,
                        behavior_id=behavior_id,
                        event_type=noise_event,
                        state=state,
                        event_index=event_index,
                        cycle_length=len(variant.log_pattern),
                    )
                )
                timestamp += timedelta(seconds=self.rng.uniform(*step_interval_seconds))

            rows.append(
                self._build_row(
                    timestamp=timestamp,
                    trajectory_id=trajectory_id,
                    cycle_id=cycle_id,
                    behavior_id=behavior_id,
                    event_type=event_type,
                    state=state,
                    event_index=event_index,
                    cycle_length=len(variant.log_pattern),
                )
            )
            timestamp += timedelta(seconds=self.rng.uniform(*step_interval_seconds))
            current_gap += 1

        return rows, timestamp

    def _build_row(
        self,
        timestamp: datetime,
        trajectory_id: int,
        cycle_id: int,
        behavior_id: int,
        event_type: str,
        state: str,
        event_index: int,
        cycle_length: int,
    ) -> dict:
        severity = self._sample_severity(event_type)
        sensor_value, control_value = self._generate_numeric_fields(
            behavior_id=behavior_id,
            event_type=event_type,
            event_index=event_index,
            cycle_length=cycle_length,
        )
        return {
            "timestamp": timestamp.isoformat(),
            "trajectory_id": trajectory_id,
            "cycle_id": cycle_id,
            "cycle_behavior_id": behavior_id,
            "event_type": event_type,
            "component": COMPONENT_BY_EVENT[event_type],
            "severity": severity,
            "state": state,
            "message": MESSAGE_BY_EVENT[event_type].format(state=state),
            "behavior_id": behavior_id,
            "sensor_value": sensor_value,
            "control_value": control_value,
            "is_transition": False,
        }

    def _sample_severity(self, event_type: str) -> str:
        if event_type in {"ALARM_TRIGGER", "COMMUNICATION_ERROR"}:
            return "ERROR" if self.rng.random() < 0.08 else "WARNING"
        return "INFO"

    def _generate_numeric_fields(
        self,
        behavior_id: int,
        event_type: str,
        event_index: int,
        cycle_length: int,
    ) -> tuple[float, float]:
        """Generate trend-based numeric columns that avoid a simple mean shortcut."""

        sensor_value = 0.0
        control_value = 0.0

        if event_type == "SENSOR_READ":
            progress = event_index / max(1, cycle_length - 1)
            if behavior_id == 0:  # successful recovery
                baseline = 82.0 - 28.0 * progress
            elif behavior_id == 1:  # failed recovery
                baseline = 82.0 - 12.0 * progress if progress < 0.7 else 80.0 + 10.0 * (progress - 0.7)
            elif behavior_id == 2:  # successful comm recovery
                baseline = 74.0 - 16.0 * progress
            else:  # persistent comm failure
                baseline = 74.0 - 8.0 * progress if progress < 0.7 else 70.0 + 8.0 * (progress - 0.7)
            sensor_value = round(baseline + self.rng.uniform(-4.0, 4.0), 2)
        elif event_type == "ALARM_TRIGGER":
            sensor_value = round(86.0 + self.rng.uniform(-3.0, 5.0), 2)
        elif event_type == "ACTUATOR_CMD":
            control_value = round(45.0 + self.rng.uniform(-12.0, 12.0), 2)

        return sensor_value, control_value

    def _sample_noise_distribution(self, noise_events: int, gap_count: int) -> List[int]:
        counts = [0] * gap_count
        for _ in range(noise_events):
            counts[self.rng.randrange(gap_count)] += 1
        return counts

    def _validate_behavior_ids(self, behavior_ids: Iterable[int]) -> None:
        invalid = [behavior_id for behavior_id in behavior_ids if behavior_id not in self.behaviors]
        if invalid:
            raise ValueError(f"Unknown behavior_ids: {invalid}")

    @staticmethod
    def _normalize_behavior_probs(
        behavior_ids: Sequence[int],
        behavior_probs: Sequence[float] | None,
    ) -> List[float]:
        if behavior_probs is None:
            probs = [BEHAVIOR_PROB_MAP[behavior_id] for behavior_id in behavior_ids]
        else:
            probs = list(behavior_probs)
            if len(probs) != len(behavior_ids):
                raise ValueError("behavior_probs length must match behavior_ids length.")
            if any(prob < 0 for prob in probs):
                raise ValueError("behavior_probs cannot contain negative values.")

        total = sum(probs)
        if total <= 0:
            raise ValueError("behavior_probs must sum to a positive value.")
        return [prob / total for prob in probs]
