"""RQ4 scenario definitions.

Five scenarios that tell a progressive story about CNC machine embedding:

  cnc_part_simple   — sanity check: 1 program, 6 channels, clean
  cnc_part_full     — realistic: 1 program, all 10 channels, full op-state cycle
  cnc_two_parts     — cross-part: 2 different programs alternating, 10 channels
  cnc_tool_wear     — drift detection: 1 program, vibration grows 4%/run
  cnc_noisy         — robustness: same as full, σ=0.25

Primary mode label:   program_step_id  (which G-code block)
Secondary mode label: op_state_id      (operating state)
"""
from __future__ import annotations

from typing import List

from rq4.generation.cnc_generator import CNCScenarioConfig

# Channel indices for the "simple" 6-channel subset:
#   0=axis_x, 1=axis_y, 2=axis_z, 3=spindle_speed, 5=vibration_x, 8=tool_id
_SIMPLE_CHANNELS = [0, 1, 2, 3, 5, 8]

SCENARIOS: List[CNCScenarioConfig] = [
    # ── 1. Sanity check ──────────────────────────────────────────────────────
    CNCScenarioConfig(
        scenario_id    = "cnc_part_simple",
        program_ids    = [0],           # square contour only
        num_runs       = 20,
        noise_std      = 0.05,
        fault_prob     = 0.0,
        wear_rate      = 0.0,
        seed           = 42,
        channel_subset = _SIMPLE_CHANNELS,
    ),

    # ── 2. Full realistic scenario ────────────────────────────────────────────
    CNCScenarioConfig(
        scenario_id = "cnc_part_full",
        program_ids = [0],              # square contour, all 10 channels
        num_runs    = 20,
        noise_std   = 0.05,
        fault_prob  = 0.0,
        wear_rate   = 0.0,
        seed        = 42,
    ),

    # ── 3. Two different parts alternating ───────────────────────────────────
    CNCScenarioConfig(
        scenario_id = "cnc_two_parts",
        program_ids = [0, 1],           # square + triangle, 10 each
        num_runs    = 20,
        noise_std   = 0.05,
        fault_prob  = 0.0,
        wear_rate   = 0.0,
        seed        = 42,
    ),

    # ── 4. Tool wear (vibration drift across runs) ────────────────────────────
    CNCScenarioConfig(
        scenario_id = "cnc_tool_wear",
        program_ids = [0],
        num_runs    = 20,
        noise_std   = 0.05,
        fault_prob  = 0.0,
        wear_rate   = 0.04,             # +4% vib amplitude per run → +76% at run 19
        seed        = 42,
    ),

    # ── 5. High noise (robustness test) ──────────────────────────────────────
    CNCScenarioConfig(
        scenario_id = "cnc_noisy",
        program_ids = [0],
        num_runs    = 20,
        noise_std   = 0.25,
        fault_prob  = 0.0,
        wear_rate   = 0.0,
        seed        = 42,
    ),

    # ── 6. Pocket milling (Z-varying, more program steps) ────────────────────
    CNCScenarioConfig(
        scenario_id = "cnc_pocket",
        program_ids = [2],
        num_runs    = 20,
        noise_std   = 0.05,
        fault_prob  = 0.0,
        wear_rate   = 0.0,
        seed        = 42,
    ),

    # ── 7. Fault injection (alarm events) ────────────────────────────────────
    CNCScenarioConfig(
        scenario_id = "cnc_faults",
        program_ids = [0],
        num_runs    = 20,
        noise_std   = 0.05,
        fault_prob  = 0.08,             # ~8% chance of fault per active step
        wear_rate   = 0.0,
        seed        = 42,
    ),
]

SCENARIO_IDS: List[str] = [s.scenario_id for s in SCENARIOS]


def get_scenario(scenario_id: str) -> CNCScenarioConfig:
    for s in SCENARIOS:
        if s.scenario_id == scenario_id:
            return s
    raise KeyError(f"Unknown scenario_id: {scenario_id!r}")
