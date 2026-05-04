"""Behavior-driven raw log generator v4: latent dynamic behavior sequences."""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import pandas as pd


COMPONENTS: tuple[str, ...] = (
    "SensorUnit",
    "ControlUnit",
    "NetworkUnit",
    "ProcessingUnit",
)

EVENT_VOCAB: tuple[str, ...] = (
    "STATUS_CHECK",
    "VALUE_READ",
    "ERROR_DETECTED",
    "WARNING_RAISED",
    "DIAGNOSIS_START",
    "RECOVERY_ACTION",
    "HEALTH_OK",
    "RETRY_OPERATION",
    "COMM_DELAY",
    "ESCALATION",
)

BASE_EVENT_PROBS: Dict[str, float] = {
    "STATUS_CHECK": 0.15,
    "VALUE_READ": 0.15,
    "ERROR_DETECTED": 0.12,
    "WARNING_RAISED": 0.10,
    "DIAGNOSIS_START": 0.10,
    "RECOVERY_ACTION": 0.12,
    "RETRY_OPERATION": 0.10,
    "HEALTH_OK": 0.10,
    "COMM_DELAY": 0.08,
    "ESCALATION": 0.08,
}

STATE_VOCAB: tuple[str, ...] = (
    "RUNNING",
    "WARNING",
    "RECOVERY",
    "DEGRADED",
)

SEVERITY_VOCAB: tuple[str, ...] = (
    "INFO",
    "WARNING",
    "ERROR",
    "CRITICAL",
)

EVENT_COMPONENT_WEIGHTS: Dict[str, Dict[str, float]] = {
    "STATUS_CHECK": {component: 1.0 for component in COMPONENTS},
    "VALUE_READ": {"SensorUnit": 1.0, "ProcessingUnit": 0.9, "ControlUnit": 0.4, "NetworkUnit": 0.3},
    "ERROR_DETECTED": {"SensorUnit": 0.9, "ControlUnit": 0.8, "NetworkUnit": 0.8, "ProcessingUnit": 0.7},
    "WARNING_RAISED": {"SensorUnit": 0.9, "ControlUnit": 0.8, "NetworkUnit": 0.8, "ProcessingUnit": 0.7},
    "DIAGNOSIS_START": {"ProcessingUnit": 1.0, "ControlUnit": 0.8, "NetworkUnit": 0.6, "SensorUnit": 0.5},
    "RECOVERY_ACTION": {"ControlUnit": 1.0, "ProcessingUnit": 0.8, "NetworkUnit": 0.7, "SensorUnit": 0.3},
    "HEALTH_OK": {component: 1.0 for component in COMPONENTS},
    "RETRY_OPERATION": {"ControlUnit": 0.9, "NetworkUnit": 0.9, "ProcessingUnit": 0.7, "SensorUnit": 0.4},
    "COMM_DELAY": {"NetworkUnit": 1.0, "ProcessingUnit": 0.7, "ControlUnit": 0.5, "SensorUnit": 0.3},
    "ESCALATION": {"ControlUnit": 0.8, "NetworkUnit": 0.8, "ProcessingUnit": 0.8, "SensorUnit": 0.7},
}

MESSAGE_TEMPLATES: Dict[str, str] = {
    "STATUS_CHECK": "{component} performed a status check while state was {state}.",
    "VALUE_READ": "{component} recorded a measurement while state was {state}.",
    "ERROR_DETECTED": "{component} detected an abnormal condition while state was {state}.",
    "WARNING_RAISED": "{component} raised a warning while state was {state}.",
    "DIAGNOSIS_START": "{component} started diagnosis while state was {state}.",
    "RECOVERY_ACTION": "{component} applied a recovery action while state was {state}.",
    "HEALTH_OK": "{component} reported healthy status while state was {state}.",
    "RETRY_OPERATION": "{component} retried an operation while state was {state}.",
    "COMM_DELAY": "{component} observed communication delay while state was {state}.",
    "ESCALATION": "{component} observed the issue escalate while state was {state}.",
}

COMPONENT_BIAS: Dict[str, float] = {
    "SensorUnit": 0.05,
    "ControlUnit": 0.00,
    "NetworkUnit": 0.07,
    "ProcessingUnit": 0.03,
}


@dataclass(frozen=True)
class BehaviorVariantSpec:
    """Latent-process parameterization for one behavior family."""

    variant_id: int
    variant_name: str
    params: Dict[str, float | int | Sequence[str]]


@dataclass(frozen=True)
class BehaviorSpec:
    """Dynamic behavior family defined by latent process evolution."""

    behavior_id: int
    behavior_name: str
    variants: tuple[BehaviorVariantSpec, ...]


BEHAVIOR_CATALOG_V4: tuple[BehaviorSpec, ...] = (
    BehaviorSpec(
        behavior_id=0,
        behavior_name="Fast Stable Recovery",
        variants=(
            BehaviorVariantSpec(0, "fast_recovery_a", {"shock": 0.30, "recovery_rate": 0.16, "health_recovery": 0.13}),
            BehaviorVariantSpec(1, "fast_recovery_b", {"shock": 0.36, "recovery_rate": 0.18, "health_recovery": 0.14}),
            BehaviorVariantSpec(2, "fast_recovery_c", {"shock": 0.28, "recovery_rate": 0.15, "health_recovery": 0.12}),
        ),
    ),
    BehaviorSpec(
        behavior_id=1,
        behavior_name="Delayed Recovery",
        variants=(
            BehaviorVariantSpec(0, "delayed_recovery_a", {"shock": 0.32, "recovery_rate": 0.07, "pause_prob": 0.18, "health_recovery": 0.06}),
            BehaviorVariantSpec(1, "delayed_recovery_b", {"shock": 0.36, "recovery_rate": 0.08, "pause_prob": 0.25, "health_recovery": 0.07}),
            BehaviorVariantSpec(2, "delayed_recovery_c", {"shock": 0.30, "recovery_rate": 0.06, "pause_prob": 0.30, "health_recovery": 0.05}),
        ),
    ),
    BehaviorSpec(
        behavior_id=2,
        behavior_name="Oscillating Instability",
        variants=(
            BehaviorVariantSpec(0, "oscillating_instability_a", {"shock": 0.30, "oscillation_amplitude": 0.11, "oscillation_period": 3, "recovery_rate": 0.08}),
            BehaviorVariantSpec(1, "oscillating_instability_b", {"shock": 0.34, "oscillation_amplitude": 0.14, "oscillation_period": 4, "recovery_rate": 0.07}),
            BehaviorVariantSpec(2, "oscillating_instability_c", {"shock": 0.28, "oscillation_amplitude": 0.10, "oscillation_period": 5, "recovery_rate": 0.09}),
        ),
    ),
    BehaviorSpec(
        behavior_id=3,
        behavior_name="Cascading Failure",
        variants=(
            BehaviorVariantSpec(0, "cascading_failure_a", {"shock": 0.34, "spread_speed": 3, "damage_rate": 0.10, "spread_path": ("SensorUnit", "ProcessingUnit", "ControlUnit", "NetworkUnit")}),
            BehaviorVariantSpec(1, "cascading_failure_b", {"shock": 0.36, "spread_speed": 4, "damage_rate": 0.09, "spread_path": ("NetworkUnit", "ProcessingUnit", "ControlUnit")}),
            BehaviorVariantSpec(2, "cascading_failure_c", {"shock": 0.32, "spread_speed": 2, "damage_rate": 0.11, "spread_path": ("ControlUnit", "ProcessingUnit", "NetworkUnit", "SensorUnit")}),
        ),
    ),
)


@dataclass
class SegmentPlan:
    """One non-transition behavior segment inside a trajectory."""

    segment_id: int
    behavior_id: int
    behavior_name: str
    variant_id: int
    variant_name: str
    segment_length: int
    load_anchor: float
    initial_component: str
    variant_params: Dict[str, float | int | Sequence[str]]


class LatentBehaviorLogGeneratorV4:
    """Generate trajectories from latent system dynamics and observable emissions."""

    def __init__(self, seed: int = 42) -> None:
        self.seed = seed
        self.rng = random.Random(seed)
        self.behaviors = {spec.behavior_id: spec for spec in BEHAVIOR_CATALOG_V4}
        self.runtime_cfg: dict[str, float | int | tuple[int, int]] = {}

    def generate_dataset(
        self,
        num_trajectories: int,
        events_per_trajectory: int,
        behavior_ids: Sequence[int] | None = None,
        behavior_probs: Sequence[float] | None = None,
        start_time: datetime | None = None,
        trajectory_spacing_seconds: float = 900.0,
        step_interval_seconds: tuple[float, float] = (1.0, 3.0),
        segment_length_range: tuple[int, int] = (48, 96),
        transition_length_range: tuple[int, int] = (8, 14),
        irrelevant_noise_prob: float = 0.08,
        severity_noise_prob: float = 0.04,
        numeric_noise_std: float = 0.03,
        component_noise_prob: float = 0.03,
        dominant_behavior_threshold: float = 0.7,
        state_delay_steps_range: tuple[int, int] = (2, 5),
        state_noise_prob: float = 0.12,
        state_update_prob: float = 0.5,
        false_negative_prob: float = 0.10,
        false_positive_prob: float = 0.08,
        critical_prob_cap: float = 0.25,
    ) -> pd.DataFrame:
        """Generate a full v4 dataset with real transition regions and hidden metadata."""

        del dominant_behavior_threshold
        if num_trajectories <= 0:
            raise ValueError("num_trajectories must be positive.")
        if events_per_trajectory <= 0:
            raise ValueError("events_per_trajectory must be positive.")
        if trajectory_spacing_seconds <= 0:
            raise ValueError("trajectory_spacing_seconds must be positive.")
        if start_time is None:
            start_time = datetime(2026, 1, 1, 0, 0, 0)
        self.runtime_cfg = {
            "state_delay_steps_range": state_delay_steps_range,
            "state_noise_prob": float(state_noise_prob),
            "state_update_prob": float(state_update_prob),
            "severity_noise_prob": float(severity_noise_prob),
            "false_negative_prob": float(false_negative_prob),
            "false_positive_prob": float(false_positive_prob),
            "critical_prob_cap": float(critical_prob_cap),
        }

        behavior_ids = sorted(self.behaviors) if behavior_ids is None else list(behavior_ids)
        self._validate_behavior_ids(behavior_ids)
        weights = self._normalize_behavior_probs(behavior_ids, behavior_probs)

        rows: list[dict] = []
        current_start = start_time
        for trajectory_id in range(num_trajectories):
            trajectory_rows, _ = self.generate_trajectory(
                trajectory_id=trajectory_id,
                events_per_trajectory=events_per_trajectory,
                behavior_ids=behavior_ids,
                behavior_probs=weights,
                start_time=current_start,
                step_interval_seconds=step_interval_seconds,
                segment_length_range=segment_length_range,
                transition_length_range=transition_length_range,
                irrelevant_noise_prob=irrelevant_noise_prob,
                severity_noise_prob=severity_noise_prob,
                numeric_noise_std=numeric_noise_std,
                component_noise_prob=component_noise_prob,
            )
            rows.extend(trajectory_rows)
            current_start += timedelta(seconds=trajectory_spacing_seconds)

        return pd.DataFrame(rows)

    def generate_trajectory(
        self,
        trajectory_id: int,
        events_per_trajectory: int,
        behavior_ids: Sequence[int],
        behavior_probs: Sequence[float],
        start_time: datetime,
        step_interval_seconds: tuple[float, float] = (1.0, 3.0),
        segment_length_range: tuple[int, int] = (48, 96),
        transition_length_range: tuple[int, int] = (8, 14),
        irrelevant_noise_prob: float = 0.08,
        severity_noise_prob: float = 0.04,
        numeric_noise_std: float = 0.03,
        component_noise_prob: float = 0.03,
    ) -> tuple[list[dict], datetime]:
        """Generate one trajectory with multiple segments and explicit transitions."""

        segment_plans = self._sample_segment_plans(
            total_events=events_per_trajectory,
            behavior_ids=behavior_ids,
            behavior_probs=behavior_probs,
            segment_length_range=segment_length_range,
            transition_length_range=transition_length_range,
        )
        rows: list[dict] = []
        timestamp = start_time
        event_index = 0
        previous_final_state: dict | None = None

        for idx, plan in enumerate(segment_plans):
            if idx > 0:
                source_plan = segment_plans[idx - 1]
                transition_length = int(source_plan.variant_params.get("transition_length", 0))
                transition_rows, transition_final_state, timestamp, event_index = self._generate_transition_rows(
                    trajectory_id=trajectory_id,
                    segment_id=plan.segment_id,
                    source_plan=source_plan,
                    target_plan=plan,
                    previous_state=previous_final_state,
                    transition_length=transition_length,
                    start_time=timestamp,
                    event_start_index=event_index,
                    step_interval_seconds=step_interval_seconds,
                    irrelevant_noise_prob=irrelevant_noise_prob,
                    severity_noise_prob=severity_noise_prob,
                    numeric_noise_std=numeric_noise_std,
                    component_noise_prob=component_noise_prob,
                )
                rows.extend(transition_rows)
                previous_final_state = transition_final_state

            segment_rows, previous_final_state, timestamp, event_index = self._generate_segment_rows(
                trajectory_id=trajectory_id,
                plan=plan,
                previous_state=previous_final_state,
                start_time=timestamp,
                event_start_index=event_index,
                step_interval_seconds=step_interval_seconds,
                irrelevant_noise_prob=irrelevant_noise_prob,
                severity_noise_prob=severity_noise_prob,
                numeric_noise_std=numeric_noise_std,
                component_noise_prob=component_noise_prob,
            )
            rows.extend(segment_rows)

        return rows, timestamp

    def build_metadata(
        self,
        dataset: pd.DataFrame,
        requested_num_trajectories: int,
        events_per_trajectory: int,
        behavior_ids: Sequence[int],
        behavior_probs: Sequence[float],
    ) -> dict:
        """Build reproducibility and latent-process metadata."""

        behavior_counts = (
            dataset.groupby("behavior_id")["trajectory_id"]
            .nunique()
            .reset_index(name="trajectory_count")
            .sort_values("behavior_id")
        )
        behavior_counts["behavior_name"] = behavior_counts["behavior_id"].map(
            {spec.behavior_id: spec.behavior_name for spec in BEHAVIOR_CATALOG_V4}
        )

        segment_counts = (
            dataset.loc[~dataset["is_transition"], ["trajectory_id", "segment_id", "behavior_id"]]
            .drop_duplicates()
            .groupby("behavior_id")["segment_id"]
            .count()
            .reset_index(name="segment_count")
            .sort_values("behavior_id")
        )

        transition_count = int(dataset["is_transition"].sum())
        return {
            "seed": self.seed,
            "requested_num_trajectories": requested_num_trajectories,
            "events_per_trajectory": events_per_trajectory,
            "generated_num_events": int(len(dataset)),
            "generated_num_trajectories": int(dataset["trajectory_id"].nunique()),
            "selected_behavior_ids": list(behavior_ids),
            "selected_behavior_probs": list(behavior_probs),
            "transition_event_count": transition_count,
            "behavior_trajectory_counts": behavior_counts.to_dict(orient="records"),
            "behavior_segment_counts": segment_counts.to_dict(orient="records"),
            "catalog": self.get_behavior_catalog(),
        }

    def save_dataset(self, dataset: pd.DataFrame, metadata: dict, output_dir: str | Path, output_name: str) -> dict:
        """Save dataset CSV plus metadata JSON."""

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        csv_path = output_dir / output_name
        metadata_path = output_dir / f"{Path(output_name).stem}_metadata.json"
        dataset.to_csv(csv_path, index=False)
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        return {"csv_path": str(csv_path), "metadata_path": str(metadata_path)}

    def get_behavior_catalog(self) -> list[dict]:
        """Return catalog in JSON-serializable form."""

        return [asdict(spec) for spec in BEHAVIOR_CATALOG_V4]

    def _sample_segment_plans(
        self,
        total_events: int,
        behavior_ids: Sequence[int],
        behavior_probs: Sequence[float],
        segment_length_range: tuple[int, int],
        transition_length_range: tuple[int, int],
    ) -> list[SegmentPlan]:
        min_seg, max_seg = segment_length_range
        min_trans, max_trans = transition_length_range
        if min_seg <= 0 or max_seg < min_seg:
            raise ValueError("segment_length_range must satisfy 0 < min <= max.")
        if min_trans < 0 or max_trans < min_trans:
            raise ValueError("transition_length_range must satisfy 0 <= min <= max.")

        plans: list[SegmentPlan] = []
        remaining = total_events
        next_segment_id = 0
        previous_behavior_id: int | None = None

        while remaining > 0:
            if remaining <= max_seg:
                seg_len = remaining
            else:
                max_seg_for_now = min(max_seg, remaining - min_trans - min_seg)
                if max_seg_for_now < min_seg:
                    seg_len = remaining
                else:
                    seg_len = self.rng.randint(min_seg, max_seg_for_now)
            candidate_ids = [bid for bid in behavior_ids if bid != previous_behavior_id] or list(behavior_ids)
            candidate_weights = self._normalize_behavior_probs(
                candidate_ids,
                [behavior_probs[behavior_ids.index(bid)] for bid in candidate_ids],
            )
            behavior_id = self.rng.choices(candidate_ids, weights=candidate_weights, k=1)[0]
            spec = self.behaviors[behavior_id]
            variant = self.rng.choice(spec.variants)
            load_anchor = self.rng.uniform(0.20, 0.85)
            initial_component = self.rng.choice(COMPONENTS)
            params = dict(variant.params)
            plans.append(
                SegmentPlan(
                    segment_id=next_segment_id,
                    behavior_id=behavior_id,
                    behavior_name=spec.behavior_name,
                    variant_id=variant.variant_id,
                    variant_name=variant.variant_name,
                    segment_length=seg_len,
                    load_anchor=load_anchor,
                    initial_component=initial_component,
                    variant_params=params,
                )
            )
            remaining -= seg_len
            next_segment_id += 1
            previous_behavior_id = behavior_id
            if remaining <= 0:
                break
            max_transition = min(max_trans, remaining - min_seg)
            if max_transition < min_trans:
                transition_len = 0
            else:
                transition_len = self.rng.randint(min_trans, max_transition)
            plans[-1].variant_params["transition_length"] = transition_len
            remaining -= transition_len

        if plans:
            plans[-1].variant_params["transition_length"] = 0
        return plans

    def _generate_segment_rows(
        self,
        trajectory_id: int,
        plan: SegmentPlan,
        previous_state: dict | None,
        start_time: datetime,
        event_start_index: int,
        step_interval_seconds: tuple[float, float],
        irrelevant_noise_prob: float,
        severity_noise_prob: float,
        numeric_noise_std: float,
        component_noise_prob: float,
    ) -> tuple[list[dict], dict, datetime, int]:
        state = self._initialize_latent_state(plan, previous_state)
        rows: list[dict] = []
        timestamp = start_time
        event_index = event_start_index
        for local_step in range(plan.segment_length):
            state = self._update_latent_state(
                state=state,
                behavior_id=plan.behavior_id,
                params=plan.variant_params,
                load_anchor=plan.load_anchor,
                step_index=local_step,
                segment_length=plan.segment_length,
            )
            row = self._emit_row(
                timestamp=timestamp,
                trajectory_id=trajectory_id,
                segment_id=plan.segment_id,
                event_index=event_index,
                behavior_id=plan.behavior_id,
                behavior_name=plan.behavior_name,
                variant_id=plan.variant_id,
                variant_name=plan.variant_name,
                state=state,
                local_phase=local_step / max(plan.segment_length - 1, 1),
                is_transition=False,
                source_behavior_id=plan.behavior_id,
                target_behavior_id=plan.behavior_id,
                transition_progress=0.0,
                irrelevant_noise_prob=irrelevant_noise_prob,
                severity_noise_prob=severity_noise_prob,
                numeric_noise_std=numeric_noise_std,
                component_noise_prob=component_noise_prob,
            )
            rows.append(row)
            timestamp += timedelta(seconds=self.rng.uniform(*step_interval_seconds))
            event_index += 1
        return rows, state, timestamp, event_index

    def _generate_transition_rows(
        self,
        trajectory_id: int,
        segment_id: int,
        source_plan: SegmentPlan,
        target_plan: SegmentPlan,
        previous_state: dict | None,
        transition_length: int,
        start_time: datetime,
        event_start_index: int,
        step_interval_seconds: tuple[float, float],
        irrelevant_noise_prob: float,
        severity_noise_prob: float,
        numeric_noise_std: float,
        component_noise_prob: float,
    ) -> tuple[list[dict], dict, datetime, int]:
        if transition_length <= 0 or previous_state is None:
            target_state = self._initialize_latent_state(target_plan, previous_state)
            return [], target_state, start_time, event_start_index

        target_seed_state = self._initialize_latent_state(target_plan, previous_state)
        source_state = self._clone_state(previous_state)
        rows: list[dict] = []
        timestamp = start_time
        event_index = event_start_index
        transition_state = self._clone_state(previous_state)

        for step in range(transition_length):
            progress = (step + 1) / transition_length
            transition_state = self._blend_states(source_state, target_seed_state, progress)
            transition_state["load_level"] = self._clip(
                transition_state["load_level"] + self.rng.uniform(-0.03, 0.03)
            )
            dominant_behavior_id = source_plan.behavior_id if progress < 0.5 else target_plan.behavior_id
            dominant_behavior_name = source_plan.behavior_name if progress < 0.5 else target_plan.behavior_name
            variant_id = source_plan.variant_id if progress < 0.5 else target_plan.variant_id
            variant_name = source_plan.variant_name if progress < 0.5 else target_plan.variant_name

            row = self._emit_row(
                timestamp=timestamp,
                trajectory_id=trajectory_id,
                segment_id=segment_id,
                event_index=event_index,
                behavior_id=dominant_behavior_id,
                behavior_name=dominant_behavior_name,
                variant_id=variant_id,
                variant_name=variant_name,
                state=transition_state,
                local_phase=progress,
                is_transition=True,
                source_behavior_id=source_plan.behavior_id,
                target_behavior_id=target_plan.behavior_id,
                transition_progress=progress,
                irrelevant_noise_prob=irrelevant_noise_prob,
                severity_noise_prob=severity_noise_prob,
                numeric_noise_std=numeric_noise_std,
                component_noise_prob=component_noise_prob,
            )
            rows.append(row)
            timestamp += timedelta(seconds=self.rng.uniform(*step_interval_seconds))
            event_index += 1

        return rows, transition_state, timestamp, event_index

    def _initialize_latent_state(self, plan: SegmentPlan, previous_state: dict | None) -> dict:
        if previous_state is None:
            load_level = plan.load_anchor
        else:
            load_level = 0.6 * previous_state["load_level"] + 0.4 * plan.load_anchor

        health = {component: self.rng.uniform(0.82, 0.96) for component in COMPONENTS}
        affected = plan.initial_component
        if plan.behavior_id == 0:
            risk = self.rng.uniform(0.32, 0.48)
            recovery = self.rng.uniform(0.12, 0.22)
            health[affected] = self.rng.uniform(0.45, 0.62)
        elif plan.behavior_id == 1:
            risk = self.rng.uniform(0.40, 0.58)
            recovery = self.rng.uniform(0.06, 0.14)
            health[affected] = self.rng.uniform(0.40, 0.58)
        elif plan.behavior_id == 2:
            risk = self.rng.uniform(0.42, 0.60)
            recovery = self.rng.uniform(0.08, 0.18)
            health[affected] = self.rng.uniform(0.38, 0.55)
        else:
            spread_path = list(plan.variant_params["spread_path"])
            affected = str(spread_path[0])
            risk = self.rng.uniform(0.50, 0.68)
            recovery = self.rng.uniform(0.02, 0.10)
            for component in spread_path[:2]:
                health[str(component)] = self.rng.uniform(0.38, 0.55)

        latent_state_name = self._derive_latent_state_name(risk, recovery, health)
        severity_name = self._severity_from_risk(risk)
        delay_min, delay_max = self.runtime_cfg.get("state_delay_steps_range", (2, 5))
        state_delay_steps = self.rng.randint(int(delay_min), int(delay_max))
        severity_delay_steps = max(1, state_delay_steps - 1 + self.rng.choice((0, 1)))

        return {
            "risk_level": self._clip(risk),
            "load_level": self._clip(load_level),
            "recovery_progress": self._clip(recovery),
            "component_health": health,
            "primary_component": affected,
            "spread_index": 0,
            "observed_state": latent_state_name,
            "observed_severity": severity_name,
            "latent_state_history": [latent_state_name],
            "risk_history": [self._clip(risk)],
            "state_delay_steps": state_delay_steps,
            "severity_delay_steps": severity_delay_steps,
        }

    def _update_latent_state(
        self,
        state: dict,
        behavior_id: int,
        params: Dict[str, float | int | Sequence[str]],
        load_anchor: float,
        step_index: int,
        segment_length: int,
    ) -> dict:
        current = self._clone_state(state)
        load_noise = self.rng.uniform(-0.03, 0.03)
        current["load_level"] = self._clip(current["load_level"] + 0.08 * (load_anchor - current["load_level"]) + load_noise)
        load_effect = 0.05 * (current["load_level"] - 0.5)
        phase = step_index / max(segment_length - 1, 1)
        risk = current["risk_level"]
        recovery = current["recovery_progress"]
        health = current["component_health"]
        primary_component = str(current["primary_component"])
        noise = self.rng.uniform(-0.02, 0.02)

        if behavior_id == 0:
            shock = float(params["shock"]) * max(0.0, 0.30 - phase)
            recovery_gain = float(params["recovery_rate"]) * (0.65 + phase)
            risk = risk + shock + load_effect - recovery_gain + noise
            recovery = recovery + float(params["recovery_rate"]) - 0.02 * current["load_level"] + self.rng.uniform(-0.02, 0.02)
            health[primary_component] = self._clip(health[primary_component] + float(params["health_recovery"]) + self.rng.uniform(-0.02, 0.02))

        elif behavior_id == 1:
            shock = float(params["shock"]) * max(0.0, 0.35 - phase)
            pause_penalty = 0.05 if self.rng.random() < float(params["pause_prob"]) else 0.0
            recovery_gain = float(params["recovery_rate"]) * (0.40 + 0.60 * phase)
            risk = risk + shock + load_effect - 0.60 * recovery_gain + noise + 0.02 * pause_penalty
            recovery = recovery + recovery_gain - pause_penalty - 0.03 * current["load_level"] + self.rng.uniform(-0.02, 0.02)
            health[primary_component] = self._clip(health[primary_component] + float(params["health_recovery"]) - 0.02 * pause_penalty + self.rng.uniform(-0.02, 0.02))

        elif behavior_id == 2:
            amplitude = float(params["oscillation_amplitude"])
            period = max(2, int(params["oscillation_period"]))
            oscillation_phase = 1.0 if ((step_index // period) % 2 == 0) else -1.0
            oscillation_force = oscillation_phase * amplitude
            temporary_recovery = float(params["recovery_rate"]) * (0.7 if oscillation_phase < 0 else 0.3)
            regression = amplitude * (0.5 if oscillation_phase > 0 else 0.2)
            risk = risk + oscillation_force + load_effect - temporary_recovery + noise
            recovery = recovery + temporary_recovery - regression + self.rng.uniform(-0.03, 0.03)
            health[primary_component] = self._clip(health[primary_component] + temporary_recovery - 0.8 * regression + self.rng.uniform(-0.03, 0.03))

        else:
            spread_path = [str(component) for component in params["spread_path"]]
            spread_speed = max(1, int(params["spread_speed"]))
            damage_rate = float(params["damage_rate"])
            spread_index = min(len(spread_path) - 1, step_index // spread_speed)
            current["spread_index"] = spread_index
            primary_component = spread_path[spread_index]
            current["primary_component"] = primary_component
            cascade_damage = damage_rate + max(0.0, load_effect)
            risk = risk + cascade_damage + 0.02 + noise
            recovery = recovery + 0.03 - 0.06 - 0.04 * current["load_level"] + self.rng.uniform(-0.02, 0.02)
            for index, component in enumerate(spread_path):
                if index <= spread_index:
                    health[component] = self._clip(health[component] - damage_rate + self.rng.uniform(-0.02, 0.02))
                else:
                    health[component] = self._clip(health[component] + self.rng.uniform(-0.01, 0.01))

        for component in COMPONENTS:
            if component != primary_component and behavior_id != 3:
                health[component] = self._clip(health[component] + self.rng.uniform(-0.01, 0.01))

        current["risk_level"] = self._clip(risk)
        current["recovery_progress"] = self._clip(recovery)
        current["primary_component"] = primary_component
        current["component_health"] = {component: self._clip(value) for component, value in health.items()}
        return current

    def _emit_row(
        self,
        timestamp: datetime,
        trajectory_id: int,
        segment_id: int,
        event_index: int,
        behavior_id: int,
        behavior_name: str,
        variant_id: int,
        variant_name: str,
        state: dict,
        local_phase: float,
        is_transition: bool,
        source_behavior_id: int,
        target_behavior_id: int,
        transition_progress: float,
        irrelevant_noise_prob: float,
        severity_noise_prob: float,
        numeric_noise_std: float,
        component_noise_prob: float,
    ) -> dict:
        risk_level = float(state["risk_level"])
        load_level = float(state["load_level"])
        recovery_progress = float(state["recovery_progress"])
        component_health = dict(state["component_health"])
        primary_component = str(state["primary_component"])

        state_name = self._observe_state(state)
        event_type = self._sample_event_type(
            risk_level=risk_level,
            recovery_progress=recovery_progress,
            state_name=state_name,
            is_transition=is_transition,
            component_health=component_health,
            local_phase=local_phase,
        )
        if self.rng.random() < irrelevant_noise_prob:
            event_type = self.rng.choice(("STATUS_CHECK", "VALUE_READ", "HEALTH_OK"))

        component = self._sample_component(
            event_type=event_type,
            primary_component=primary_component,
            component_health=component_health,
            behavior_id=behavior_id,
            is_transition=is_transition,
            component_noise_prob=component_noise_prob,
        )
        severity = self._observe_severity(state)
        sensor_value = self._sensor_value(
            risk_level=risk_level,
            load_level=load_level,
            component=component,
            numeric_noise_std=numeric_noise_std,
        )
        control_value = self._control_value(
            event_type=event_type,
            risk_level=risk_level,
            load_level=load_level,
            recovery_progress=recovery_progress,
            numeric_noise_std=numeric_noise_std,
        )

        return {
            "timestamp": timestamp.isoformat(),
            "trajectory_id": trajectory_id,
            "segment_id": segment_id,
            "event_index": event_index,
            "event_type": event_type,
            "component": component,
            "severity": severity,
            "state": state_name,
            "message": MESSAGE_TEMPLATES[event_type].format(component=component, state=state_name),
            "sensor_value": sensor_value,
            "control_value": control_value,
            "behavior_id": behavior_id,
            "behavior_name": behavior_name,
            "variant_id": variant_id,
            "variant_name": variant_name,
            "is_transition": is_transition,
            "source_behavior_id": source_behavior_id,
            "target_behavior_id": target_behavior_id,
            "transition_progress": round(float(transition_progress), 4),
            "risk_level": round(risk_level, 6),
            "load_level": round(load_level, 6),
            "recovery_progress": round(recovery_progress, 6),
            "component_health_sensor": round(component_health["SensorUnit"], 6),
            "component_health_control": round(component_health["ControlUnit"], 6),
            "component_health_network": round(component_health["NetworkUnit"], 6),
            "component_health_processing": round(component_health["ProcessingUnit"], 6),
            "primary_component": primary_component,
        }

    def _sample_event_type(
        self,
        risk_level: float,
        recovery_progress: float,
        state_name: str,
        is_transition: bool,
        component_health: Dict[str, float],
        local_phase: float,
    ) -> str:
        weights = dict(BASE_EVENT_PROBS)
        low_health_count = sum(1 for value in component_health.values() if value < 0.60)
        if risk_level < 0.25:
            weights["STATUS_CHECK"] += 0.10
            weights["VALUE_READ"] += 0.10
            weights["HEALTH_OK"] += 0.12
        elif risk_level < 0.6:
            weights["VALUE_READ"] += 0.08
            weights["ERROR_DETECTED"] += 0.08
            weights["WARNING_RAISED"] += 0.08
        else:
            weights["ERROR_DETECTED"] += 0.10
            weights["WARNING_RAISED"] += 0.08
            weights["ESCALATION"] += 0.10

        if recovery_progress > 0.25:
            weights["DIAGNOSIS_START"] += 0.06
            weights["RECOVERY_ACTION"] += 0.10
            weights["RETRY_OPERATION"] += 0.07
        if state_name == "RECOVERY":
            weights["RECOVERY_ACTION"] += 0.08
            weights["HEALTH_OK"] += 0.05
        if state_name == "DEGRADED":
            weights["ESCALATION"] += 0.08
            weights["COMM_DELAY"] += 0.06
        if low_health_count >= 2:
            weights["ESCALATION"] += 0.06
            weights["COMM_DELAY"] += 0.05
            weights["ERROR_DETECTED"] += 0.04
        if local_phase < 0.20:
            weights["VALUE_READ"] += 0.05
            weights["ERROR_DETECTED"] += 0.04
        elif local_phase > 0.75:
            weights["HEALTH_OK"] += 0.05
            weights["STATUS_CHECK"] += 0.04
        if is_transition:
            weights["STATUS_CHECK"] += 0.05
            weights["DIAGNOSIS_START"] += 0.05
            weights["WARNING_RAISED"] += 0.04

        events = list(weights)
        probs = self._normalize_probability_list([weights[event] for event in events])
        return self.rng.choices(events, weights=probs, k=1)[0]

    def _sample_component(
        self,
        event_type: str,
        primary_component: str,
        component_health: Dict[str, float],
        behavior_id: int,
        is_transition: bool,
        component_noise_prob: float,
    ) -> str:
        weights = dict(EVENT_COMPONENT_WEIGHTS[event_type])
        lowest_health_component = min(component_health, key=component_health.get)
        weights[primary_component] = weights.get(primary_component, 0.1) + 0.7
        weights[lowest_health_component] = weights.get(lowest_health_component, 0.1) + 0.5
        if behavior_id == 3 and event_type in {"ESCALATION", "ERROR_DETECTED", "COMM_DELAY"}:
            weights[lowest_health_component] += 0.6
        if is_transition:
            for component in COMPONENTS:
                weights[component] += 0.1
        if self.rng.random() < component_noise_prob:
            return self.rng.choice(COMPONENTS)
        components = list(weights)
        probs = self._normalize_probability_list([weights[component] for component in components])
        return self.rng.choices(components, weights=probs, k=1)[0]

    def _derive_latent_state_name(self, risk_level: float, recovery_progress: float, component_health: Dict[str, float]) -> str:
        avg_health = sum(component_health.values()) / len(component_health)
        if recovery_progress >= 0.45 and risk_level < 0.65:
            return "RECOVERY"
        if risk_level < 0.25 and avg_health > 0.75:
            return "RUNNING"
        if risk_level > 0.72 or avg_health < 0.48:
            return "DEGRADED"
        return "WARNING"

    def _severity_from_risk(self, risk_level: float) -> str:
        if risk_level < 0.25:
            return "INFO"
        if risk_level < 0.55:
            return "WARNING"
        if risk_level < 0.82:
            return "ERROR"
        return "CRITICAL"

    def _observe_state(self, state: dict) -> str:
        latent_state = self._derive_latent_state_name(
            risk_level=float(state["risk_level"]),
            recovery_progress=float(state["recovery_progress"]),
            component_health=state["component_health"],
        )
        history = state.setdefault("latent_state_history", [])
        history.append(latent_state)
        if len(history) > 16:
            del history[0]

        delay = int(state.get("state_delay_steps", 2))
        candidate = history[max(0, len(history) - 1 - delay)]
        if candidate == "RECOVERY" and self.rng.random() < 0.25:
            candidate = "WARNING"
        elif candidate == "DEGRADED" and self.rng.random() < 0.20:
            candidate = self.rng.choice(["WARNING", "DEGRADED"])

        if self.rng.random() < float(self.runtime_cfg.get("state_noise_prob", 0.12)):
            candidate = self.rng.choice([name for name in STATE_VOCAB if name != candidate])

        previous = str(state.get("observed_state", candidate))
        if candidate != previous and self.rng.random() > float(self.runtime_cfg.get("state_update_prob", 0.5)):
            observed = previous
        else:
            observed = candidate
        state["observed_state"] = observed
        return observed

    def _observe_severity(self, state: dict) -> str:
        risk_history = state.setdefault("risk_history", [])
        risk_history.append(float(state["risk_level"]))
        if len(risk_history) > 16:
            del risk_history[0]

        delay = int(state.get("severity_delay_steps", max(1, int(state.get("state_delay_steps", 2)))))
        delayed_risk = float(risk_history[max(0, len(risk_history) - 1 - delay)])
        thresholds = [
            0.25 + self.rng.uniform(-0.05, 0.05),
            0.55 + self.rng.uniform(-0.06, 0.06),
            0.82 + self.rng.uniform(-0.05, 0.03),
        ]
        if delayed_risk < thresholds[0]:
            index = 0
        elif delayed_risk < thresholds[1]:
            index = 1
        elif delayed_risk < thresholds[2]:
            index = 2
        else:
            index = 3

        if delayed_risk < thresholds[0] and self.rng.random() < float(self.runtime_cfg.get("false_positive_prob", 0.08)):
            index = min(1, index + 1)
        if delayed_risk >= thresholds[1] and self.rng.random() < float(self.runtime_cfg.get("false_negative_prob", 0.10)):
            index = max(1, index - 1)
        if index == 3 and self.rng.random() > float(self.runtime_cfg.get("critical_prob_cap", 0.25)):
            index = 2
        if self.rng.random() < float(self.runtime_cfg.get("severity_noise_prob", 0.04)):
            index = max(0, min(len(SEVERITY_VOCAB) - 1, index + self.rng.choice((-1, 1))))

        observed = SEVERITY_VOCAB[index]
        state["observed_severity"] = observed
        return observed

    def _sensor_value(self, risk_level: float, load_level: float, component: str, numeric_noise_std: float) -> float:
        value = 0.15 + 0.70 * risk_level + 0.20 * load_level + COMPONENT_BIAS[component] + self.rng.gauss(0.0, numeric_noise_std)
        return round(self._clip(value), 6)

    def _control_value(
        self,
        event_type: str,
        risk_level: float,
        load_level: float,
        recovery_progress: float,
        numeric_noise_std: float,
    ) -> float:
        if event_type not in {"RECOVERY_ACTION", "RETRY_OPERATION", "DIAGNOSIS_START"}:
            base = max(0.0, 0.04 * recovery_progress + self.rng.gauss(0.0, numeric_noise_std * 0.5))
            return round(self._clip(base), 6)
        value = 0.30 + 0.35 * risk_level + 0.20 * load_level + 0.15 * recovery_progress + self.rng.gauss(0.0, numeric_noise_std)
        return round(self._clip(value), 6)

    def _clone_state(self, state: dict) -> dict:
        return {
            "risk_level": float(state["risk_level"]),
            "load_level": float(state["load_level"]),
            "recovery_progress": float(state["recovery_progress"]),
            "component_health": {component: float(value) for component, value in state["component_health"].items()},
            "primary_component": str(state["primary_component"]),
            "spread_index": int(state.get("spread_index", 0)),
            "observed_state": str(state.get("observed_state", "WARNING")),
            "observed_severity": str(state.get("observed_severity", "WARNING")),
            "latent_state_history": list(state.get("latent_state_history", [])),
            "risk_history": list(state.get("risk_history", [])),
            "state_delay_steps": int(state.get("state_delay_steps", 2)),
            "severity_delay_steps": int(state.get("severity_delay_steps", 1)),
        }

    def _blend_states(self, source: dict, target: dict, progress: float) -> dict:
        progress = self._clip(progress)
        blended_health = {
            component: self._clip((1.0 - progress) * source["component_health"][component] + progress * target["component_health"][component] + self.rng.uniform(-0.02, 0.02))
            for component in COMPONENTS
        }
        primary_component = source["primary_component"] if progress < 0.5 else target["primary_component"]
        return {
            "risk_level": self._clip((1.0 - progress) * source["risk_level"] + progress * target["risk_level"] + self.rng.uniform(-0.03, 0.03)),
            "load_level": self._clip((1.0 - progress) * source["load_level"] + progress * target["load_level"]),
            "recovery_progress": self._clip((1.0 - progress) * source["recovery_progress"] + progress * target["recovery_progress"] + self.rng.uniform(-0.03, 0.03)),
            "component_health": blended_health,
            "primary_component": primary_component,
            "spread_index": 0,
        }

    def _validate_behavior_ids(self, behavior_ids: Iterable[int]) -> None:
        unknown = sorted(set(behavior_ids) - set(self.behaviors))
        if unknown:
            raise ValueError(f"Unknown behavior_ids: {unknown}")

    def _normalize_behavior_probs(self, behavior_ids: Sequence[int], behavior_probs: Sequence[float] | None) -> list[float]:
        if behavior_probs is None:
            return [1.0 / len(behavior_ids)] * len(behavior_ids)
        if len(behavior_probs) != len(behavior_ids):
            raise ValueError("behavior_probs must match behavior_ids length.")
        return self._normalize_probability_list(list(behavior_probs))

    @staticmethod
    def _normalize_probability_list(values: Sequence[float]) -> list[float]:
        total = float(sum(values))
        if total <= 0:
            raise ValueError("Probability weights must sum to a positive value.")
        return [float(value) / total for value in values]

    @staticmethod
    def _clip(value: float) -> float:
        return max(0.0, min(1.0, float(value)))
