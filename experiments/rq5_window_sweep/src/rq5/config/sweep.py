"""RQ5 window-size sweep configuration.

Base signal: cnc_part_full (square contour, 10 channels, 20 runs, clean).
Sweep:       window_length ∈ {32, 64, 128, 256, 512}.
             stride = window_length // 8  (constant 87.5 % overlap).

Key hypothesis: window size affects
  - FFT frequency resolution  (larger → more spectral detail)
  - MOMENT temporal context   (smaller → pads more aggressively to 512)
  - Transition sharpness       (smaller → boundary localisation improves)
  - Loop consistency (DTW)     (larger → each segment is a longer trajectory)
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import List

_HERE    = Path(__file__).resolve()
_RQ4_SRC = _HERE.parents[4] / "rq4_cnc_benchmark" / "src"
if str(_RQ4_SRC) not in sys.path:
    sys.path.insert(0, str(_RQ4_SRC))

from rq4.generation.cnc_generator import CNCScenarioConfig

WINDOW_SIZES: List[int] = [32, 64, 128, 256, 512]
MODELS: List[str]       = ["fft", "moment"]


def make_sweep_scenario(window_length: int) -> CNCScenarioConfig:
    stride = max(1, window_length // 8)
    return CNCScenarioConfig(
        scenario_id   = f"cnc_ws_{window_length:03d}",
        program_ids   = [0],       # square contour — stable, repeatable
        num_runs      = 20,
        noise_std     = 0.05,
        fault_prob    = 0.0,
        wear_rate     = 0.0,
        seed          = 42,
        window_length = window_length,
        stride        = stride,
    )


SWEEP_SCENARIOS: List[CNCScenarioConfig] = [
    make_sweep_scenario(ws) for ws in WINDOW_SIZES
]
SCENARIO_IDS: List[str] = [s.scenario_id for s in SWEEP_SCENARIOS]


def get_scenario(sid: str) -> CNCScenarioConfig:
    for s in SWEEP_SCENARIOS:
        if s.scenario_id == sid:
            return s
    raise KeyError(f"Unknown sweep scenario: {sid!r}")
