from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Literal

SignalType = Literal["sine", "step", "event", "mixed"]
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
    seed: int = 42
    num_trajectories: int = 20
    window_length: int = 48
    stride: int = 12
    axis: str = "baseline"


SCENARIOS: List[ScenarioConfig] = [
    # ── BASELINE (anchor for all axes) ───────────────────────────────────────
    ScenarioConfig("baseline",     "sine",  3, 4, 0.05, "none",             0.00, axis="baseline"),

    # ── SIGNAL TYPE (3 modes, 4ch, noise=0.05, no missing) ───────────────────
    ScenarioConfig("sig_step",     "step",  3, 4, 0.05, "none",             0.00, axis="signal"),
    ScenarioConfig("sig_event",    "event", 3, 4, 0.05, "none",             0.00, axis="signal"),
    ScenarioConfig("sig_mixed",    "mixed", 3, 4, 0.05, "none",             0.00, axis="signal"),

    # ── MODE COUNT (sine, 4ch, noise=0.05, no missing) ────────────────────────
    ScenarioConfig("modes_2",      "sine",  2, 4, 0.05, "none",             0.00, axis="modes"),
    ScenarioConfig("modes_5",      "sine",  5, 4, 0.05, "none",             0.00, axis="modes"),
    ScenarioConfig("modes_8",      "sine",  8, 4, 0.05, "none",             0.00, axis="modes"),

    # ── NOISE ROBUSTNESS (sine, 3 modes, 4ch, no missing) ────────────────────
    ScenarioConfig("noise_000",    "sine",  3, 4, 0.00, "none",             0.00, axis="noise"),
    ScenarioConfig("noise_020",    "sine",  3, 4, 0.20, "none",             0.00, axis="noise"),
    ScenarioConfig("noise_040",    "sine",  3, 4, 0.40, "none",             0.00, axis="noise"),

    # ── MISSING VALUES (sine, 3 modes, 4ch, noise=0.05) ──────────────────────
    ScenarioConfig("miss_rand10",  "sine",  3, 4, 0.05, "random_point",     0.10, axis="missing"),
    ScenarioConfig("miss_rand30",  "sine",  3, 4, 0.05, "random_point",     0.30, axis="missing"),
    ScenarioConfig("miss_block30", "sine",  3, 4, 0.05, "contiguous_block", 0.30, axis="missing"),
    ScenarioConfig("miss_channel", "sine",  3, 4, 0.05, "channel_dropout",  0.25, axis="missing"),
]

SCENARIO_IDS: List[str] = [s.scenario_id for s in SCENARIOS]


def get_scenario(scenario_id: str) -> ScenarioConfig:
    for s in SCENARIOS:
        if s.scenario_id == scenario_id:
            return s
    raise KeyError(f"Unknown scenario_id: {scenario_id!r}")
