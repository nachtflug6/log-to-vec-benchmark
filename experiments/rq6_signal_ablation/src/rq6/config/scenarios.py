from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Literal

SignalType = Literal["sawtooth", "damped", "chirp", "ratio"]
MissingType = Literal["none", "random_point", "contiguous_block", "channel_dropout"]


@dataclass
class ScenarioConfig:
    scenario_id: str
    signal_type: SignalType
    num_modes: int
    num_channels: int
    noise_std: float
    missing_type: MissingType
    missing_fraction: float
    step_channels: int = 0    # for signal_type="ratio": how many channels carry step signal
    seed: int = 42
    num_trajectories: int = 20
    window_length: int = 48
    stride: int = 12
    axis: str = "signal"


SCENARIOS: List[ScenarioConfig] = [
    # ── AXIS A: Signal type (3 modes, 4ch, noise=0.05, no missing) ────────────
    # ratio_0step (pure sine) serves as the common anchor for both axes
    ScenarioConfig("ratio_0step",  "ratio",    3, 4, 0.05, "none", 0.00, step_channels=0, axis="ratio"),
    ScenarioConfig("sig_sawtooth", "sawtooth", 3, 4, 0.05, "none", 0.00, axis="signal"),
    ScenarioConfig("sig_damped",   "damped",   3, 4, 0.05, "none", 0.00, axis="signal"),
    ScenarioConfig("sig_chirp",    "chirp",    3, 4, 0.05, "none", 0.00, axis="signal"),

    # ── AXIS B: Channel-ratio sweep (3 modes, 4ch, noise=0.05, no missing) ────
    # ratio_0step listed above; ratio_4step = pure step
    ScenarioConfig("ratio_1step",  "ratio",    3, 4, 0.05, "none", 0.00, step_channels=1, axis="ratio"),
    ScenarioConfig("ratio_2step",  "ratio",    3, 4, 0.05, "none", 0.00, step_channels=2, axis="ratio"),
    ScenarioConfig("ratio_3step",  "ratio",    3, 4, 0.05, "none", 0.00, step_channels=3, axis="ratio"),
    ScenarioConfig("ratio_4step",  "ratio",    3, 4, 0.05, "none", 0.00, step_channels=4, axis="ratio"),
]

SCENARIO_IDS: List[str] = [s.scenario_id for s in SCENARIOS]


def get_scenario(scenario_id: str) -> ScenarioConfig:
    for s in SCENARIOS:
        if s.scenario_id == scenario_id:
            return s
    raise KeyError(f"Unknown scenario_id: {scenario_id!r}")
