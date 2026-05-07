"""Scenario definitions for RQ7 real-world dataset evaluation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass
class Scenario:
    id: str
    dataset: str
    description: str
    loader_fn: str          # function name in the loader module
    n_channels: int
    n_modes: int
    pseudo_runs: bool       # True if run_id is synthetic time-chunks
    label_key: str          # key in windows.npz for mode labels


SCENARIO_REGISTRY: dict[str, Scenario] = {
    "swat_binary": Scenario(
        id          = "swat_binary",
        dataset     = "SWaT",
        description = "SWaT water-plant sensor data — normal vs attack (binary)",
        loader_fn   = "load_swat_binary",
        n_channels  = 51,
        n_modes     = 2,
        pseudo_runs = True,
        label_key   = "mode_id",
    ),
    "msl_binary": Scenario(
        id          = "msl_binary",
        dataset     = "MSL",
        description = "MSL spacecraft telemetry — normal vs anomaly (binary)",
        loader_fn   = "load_msl_binary",
        n_channels  = 55,
        n_modes     = 2,
        pseudo_runs = True,
        label_key   = "mode_id",
    ),
    "tep_modes": Scenario(
        id          = "tep_modes",
        dataset     = "TEP",
        description = "Tennessee Eastman Process — 6 production modes, fault-free",
        loader_fn   = "load_tep_modes",
        n_channels  = 52,
        n_modes     = 6,
        pseudo_runs = False,   # genuine independent simulation runs
        label_key   = "mode_id",
    ),
    "tep_faults": Scenario(
        id          = "tep_faults",
        dataset     = "TEP",
        description = "Tennessee Eastman Process — IDV0-IDV8 fault types (9 classes)",
        loader_fn   = "load_tep_faults",
        n_channels  = 52,
        n_modes     = 9,
        pseudo_runs = False,
        label_key   = "mode_id",
    ),
}

# Ordered list used as default when no --scenarios flag is given
SCENARIO_IDS = ["swat_binary", "msl_binary", "tep_modes", "tep_faults"]


def get_scenario(sid: str) -> Scenario:
    if sid not in SCENARIO_REGISTRY:
        raise KeyError(f"Unknown scenario: {sid!r}. Valid: {list(SCENARIO_REGISTRY)}")
    return SCENARIO_REGISTRY[sid]
