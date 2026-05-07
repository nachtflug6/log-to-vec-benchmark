"""CNC trajectory generator for RQ4.

Two-level label structure:
  - program_step_id : which G-code block is executing  (fine, the industrial selling point)
  - op_state_id     : what the machine is doing        (coarse, matches real PLC codes)

10 output channels (physical units, un-normalised — embedding scripts z-score per channel):
  0  axis_x        mm          absolute X position
  1  axis_y        mm          absolute Y position
  2  axis_z        mm          absolute Z position
  3  spindle_speed RPM         continuous
  4  feed_rate     mm/min      continuous
  5  vibration_x   m/s²        periodic + noise, cutting-freq depends on spindle
  6  vibration_y   m/s²        same, phase-shifted by π/3
  7  temperature   °C          slow thermal lag of activity
  8  tool_id       int (0–5)   discrete — passed as float
  9  alarm_code    int (0–99)  discrete — 0 = normal; non-zero = active fault

Operating state codes (op_state_id):
  IDLE=0  HOMING=1  RAPID=2  CUTTING=3  DWELL=4  FAULT=5
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

# Re-use _extract_windows from RQ2
_HERE = Path(__file__).resolve()
_RQ2_SRC = _HERE.parents[4] / "rq2_trace_comparison" / "src"
if str(_RQ2_SRC) not in sys.path:
    sys.path.insert(0, str(_RQ2_SRC))

from rq2.generation.periodic_mode_generator import _extract_windows  # noqa: E402

# ── operating state constants ─────────────────────────────────────────────────
IDLE    = 0
HOMING  = 1
RAPID   = 2
CUTTING = 3
DWELL   = 4
FAULT   = 5

OP_STATE_NAMES = {IDLE: "IDLE", HOMING: "HOMING", RAPID: "RAPID",
                  CUTTING: "CUTTING", DWELL: "DWELL", FAULT: "FAULT"}

# Thermal activity factor: how much heat each state generates
_HEAT = {IDLE: 0.0, HOMING: 0.2, RAPID: 0.3, CUTTING: 1.0, DWELL: 0.4, FAULT: 0.1}
_T_BASE = 20.0   # ambient °C
_T_RISE = 40.0   # max temperature rise above ambient
_T_TAU  = 0.997  # thermal time constant (slow lag, ~333 steps ≈ 3.3 s at 100 Hz)


@dataclass
class ProgramStep:
    """One G-code-like command: defines geometry, operation, and signal parameters."""
    step_id:      int
    name:         str
    op_state:     int     # IDLE / HOMING / RAPID / CUTTING / DWELL

    # Target position reached by end of this step
    target_x:     float   # mm
    target_y:     float   # mm
    target_z:     float   # mm

    # Nominal duration in timesteps (varied ±10% per run)
    duration:     int

    # Spindle and motion
    spindle_rpm:  float   # 0 when not spinning
    feed_rate:    float   # mm/min; 0 during DWELL/IDLE

    # Vibration model: A·sin(2π·f·t) + A·0.3·sin(2π·2f·t) + noise
    vib_freq:     float   # normalised frequency (fraction of Nyquist)
    vib_amplitude: float  # m/s²; modulated by tool wear and op_state

    # Discrete channels
    tool_id:      int     # 0 = no tool in spindle


@dataclass
class PartProgram:
    """Ordered sequence of program steps constituting one part operation."""
    program_id:   int
    name:         str
    steps:        List[ProgramStep]
    home_x:       float = 0.0
    home_y:       float = 0.0
    home_z:       float = 5.0


@dataclass
class CNCScenarioConfig:
    scenario_id:    str
    program_ids:    List[int]         # which part programs to execute (in order, then repeat)
    num_runs:       int               # total part runs across all programs
    noise_std:      float
    fault_prob:     float             # per-step probability of a fault interruption
    wear_rate:      float             # fractional vibration amplitude increase per run
    seed:           int = 42
    window_length:  int = 96
    stride:         int = 12
    channel_subset: Optional[List[int]] = None   # None = all 10 channels


# ── part programs ─────────────────────────────────────────────────────────────

def _make_square_program() -> PartProgram:
    """Program 0: square contour 50×50 mm."""
    return PartProgram(
        program_id=0, name="square_contour",
        home_x=0.0, home_y=0.0, home_z=5.0,
        steps=[
            ProgramStep(0, "HOMING",     HOMING,  0.0,  0.0, 5.0, 60, 0,    2000, 0.03, 0.30, 0),
            ProgramStep(1, "APPROACH",   RAPID,   0.0,  0.0, 0.0, 35, 3000, 4000, 0.04, 0.20, 1),
            ProgramStep(2, "CUT_X_POS",  CUTTING, 50.0, 0.0, 0.0, 160,3000,  800, 0.13, 1.80, 1),
            ProgramStep(3, "CUT_Y_POS",  CUTTING, 50.0,50.0, 0.0, 160,3200,  850, 0.15, 2.00, 1),
            ProgramStep(4, "CUT_X_NEG",  CUTTING,  0.0,50.0, 0.0, 160,3000,  800, 0.13, 1.80, 1),
            ProgramStep(5, "CUT_Y_NEG",  CUTTING,  0.0, 0.0, 0.0, 160,3200,  850, 0.15, 2.00, 1),
            ProgramStep(6, "RETRACT",    RAPID,   0.0,  0.0, 5.0, 35, 0,    4000, 0.04, 0.15, 0),
            ProgramStep(7, "DWELL",      DWELL,   0.0,  0.0, 5.0, 50, 0,       0, 0.01, 0.05, 0),
        ],
    )


def _make_triangle_program() -> PartProgram:
    """Program 1: triangular profile — different coordinate space from square."""
    return PartProgram(
        program_id=1, name="triangle_profile",
        home_x=0.0, home_y=0.0, home_z=5.0,
        steps=[
            ProgramStep(0, "HOMING",     HOMING,  0.0,  0.0, 5.0, 60, 0,    2000, 0.03, 0.30, 0),
            ProgramStep(1, "POSITION",   RAPID,   5.0,  5.0, 0.0, 30, 2800, 5000, 0.04, 0.18, 2),
            ProgramStep(2, "CUT_DIAG",   CUTTING,45.0, 45.0, 0.0, 200,2800,  700, 0.11, 1.60, 2),
            ProgramStep(3, "CUT_HORIZ",  CUTTING,45.0,  5.0, 0.0, 150,2800,  750, 0.11, 1.55, 2),
            ProgramStep(4, "CUT_CLOSE",  CUTTING, 5.0,  5.0, 0.0, 170,2800,  700, 0.11, 1.60, 2),
            ProgramStep(5, "RETRACT",    RAPID,   0.0,  0.0, 5.0, 30, 0,    5000, 0.03, 0.15, 0),
            ProgramStep(6, "DWELL",      DWELL,   0.0,  0.0, 5.0, 50, 0,       0, 0.01, 0.05, 0),
        ],
    )


def _make_pocket_program() -> PartProgram:
    """Program 2: pocket milling — multiple depth passes, Z-varying."""
    return PartProgram(
        program_id=2, name="pocket_milling",
        home_x=0.0, home_y=0.0, home_z=5.0,
        steps=[
            ProgramStep(0,  "HOMING",     HOMING,  0.0,  0.0,  5.0, 60,  0,    2000, 0.03, 0.30, 0),
            ProgramStep(1,  "PLUNGE_1",   RAPID,  10.0, 10.0, -1.0, 40, 2500, 1000, 0.05, 0.50, 3),
            ProgramStep(2,  "SLOT_X1",    CUTTING,40.0, 10.0, -1.0, 170,2500,  600, 0.10, 1.40, 3),
            ProgramStep(3,  "SLOT_Y",     CUTTING,40.0, 40.0, -1.0, 170,2500,  600, 0.10, 1.45, 3),
            ProgramStep(4,  "SLOT_X2",    CUTTING,10.0, 40.0, -1.0, 170,2500,  600, 0.10, 1.40, 3),
            ProgramStep(5,  "PLUNGE_2",   RAPID,  10.0, 10.0, -2.0, 30, 2500, 1000, 0.05, 0.50, 3),
            ProgramStep(6,  "FINISH_X1",  CUTTING,40.0, 10.0, -2.0, 160,2800,  500, 0.12, 1.70, 3),
            ProgramStep(7,  "FINISH_Y",   CUTTING,40.0, 40.0, -2.0, 160,2800,  500, 0.12, 1.75, 3),
            ProgramStep(8,  "FINISH_X2",  CUTTING,10.0, 40.0, -2.0, 160,2800,  500, 0.12, 1.70, 3),
            ProgramStep(9,  "RETRACT",    RAPID,   0.0,  0.0,  5.0, 35,  0,   5000, 0.03, 0.15, 0),
            ProgramStep(10, "DWELL",      DWELL,   0.0,  0.0,  5.0, 50,  0,      0, 0.01, 0.05, 0),
        ],
    )


PROGRAMS = {p.program_id: p for p in [
    _make_square_program(),
    _make_triangle_program(),
    _make_pocket_program(),
]}


# ── signal rendering ──────────────────────────────────────────────────────────

def _render_step(
    step: ProgramStep,
    start_pos: Tuple[float, float, float],
    actual_duration: int,
    wear_factor: float,
    noise_std: float,
    rng: np.random.Generator,
    temp_init: float,
) -> Tuple[np.ndarray, float]:
    """Render one program step as a [actual_duration, 10] signal array.

    Returns (signal, temp_final) so thermal state threads across steps.
    """
    T = actual_duration
    t = np.arange(T, dtype=np.float64)
    sig = np.zeros((T, 10), dtype=np.float64)

    # ── Channels 0-2: axis positions (linear interpolation start→target) ──
    x0, y0, z0 = start_pos
    frac = np.linspace(0.0, 1.0, T)
    sig[:, 0] = x0 + (step.target_x - x0) * frac
    sig[:, 1] = y0 + (step.target_y - y0) * frac
    sig[:, 2] = z0 + (step.target_z - z0) * frac
    # small positioning noise (encoder quantisation / vibration)
    sig[:, 0] += rng.normal(0, noise_std * 0.1, T)
    sig[:, 1] += rng.normal(0, noise_std * 0.1, T)
    sig[:, 2] += rng.normal(0, noise_std * 0.05, T)

    # ── Channel 3: spindle speed ──
    spindle_noise = rng.normal(0, max(1.0, step.spindle_rpm * 0.01), T)
    sig[:, 3] = np.maximum(0.0, step.spindle_rpm + spindle_noise)

    # ── Channel 4: feed rate ──
    feed_noise = rng.normal(0, max(1.0, step.feed_rate * 0.02), T)
    sig[:, 4] = np.maximum(0.0, step.feed_rate + feed_noise)

    # ── Channels 5-6: vibration (fundamental + 2nd harmonic, phase-shifted Y) ──
    f = step.vib_freq
    amp = step.vib_amplitude * wear_factor
    phase_x = rng.uniform(0, 2 * np.pi)
    phase_y = phase_x + np.pi / 3
    vib_x = (amp * np.sin(2 * np.pi * f * t + phase_x)
              + amp * 0.3 * np.sin(2 * np.pi * 2 * f * t + phase_x))
    vib_y = (amp * np.sin(2 * np.pi * f * t + phase_y)
              + amp * 0.3 * np.sin(2 * np.pi * 2 * f * t + phase_y))
    sig[:, 5] = vib_x + rng.normal(0, noise_std * amp, T)
    sig[:, 6] = vib_y + rng.normal(0, noise_std * amp, T)

    # ── Channel 7: temperature (thermal lag) ──
    T_target = _T_BASE + _T_RISE * _HEAT[step.op_state]
    temp = temp_init
    temps = np.empty(T)
    for i in range(T):
        temp = _T_TAU * temp + (1 - _T_TAU) * T_target
        temps[i] = temp
    sig[:, 7] = temps + rng.normal(0, noise_std * 0.5, T)

    # ── Channel 8: tool_id (discrete, constant over step) ──
    sig[:, 8] = float(step.tool_id)

    # ── Channel 9: alarm_code (0 = normal) ──
    sig[:, 9] = 0.0

    return sig.astype(np.float32), float(temp)


def _render_fault_burst(
    duration: int,
    last_pos: Tuple[float, float, float],
    alarm_id: int,
    noise_std: float,
    rng: np.random.Generator,
    temp_init: float,
) -> Tuple[np.ndarray, float]:
    """Inject a FAULT burst: spindle ramps to 0, broadband vibration spike."""
    T = duration
    sig = np.zeros((T, 10), dtype=np.float64)
    sig[:, 0] = last_pos[0]
    sig[:, 1] = last_pos[1]
    sig[:, 2] = last_pos[2]
    # spindle ramps down
    sig[:, 3] = np.linspace(500, 0, T)
    sig[:, 4] = 0.0
    # broadband vibration burst (decaying)
    decay = np.exp(-np.arange(T) / (T * 0.4))
    sig[:, 5] = decay * rng.normal(0, 3.0, T)
    sig[:, 6] = decay * rng.normal(0, 3.0, T)
    # temperature
    T_target = _T_BASE + _T_RISE * _HEAT[FAULT]
    temp = temp_init
    temps = np.empty(T)
    for i in range(T):
        temp = _T_TAU * temp + (1 - _T_TAU) * T_target
        temps[i] = temp
    sig[:, 7] = temps
    sig[:, 8] = 0.0            # tool removed in fault
    sig[:, 9] = float(alarm_id)
    return sig.astype(np.float32), float(temp)


# ── trajectory assembler ──────────────────────────────────────────────────────

def generate_cnc_trajectory(
    program: PartProgram,
    run_id: int,
    cfg: CNCScenarioConfig,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[int]]:
    """Generate one part run as a raw timestep-level signal.

    Returns:
        signal            [T, 10] float32
        program_step_ids  [T]     int32
        op_state_ids      [T]     int32
        change_points     List[int]   timestep indices where either label changes
    """
    wear_factor = 1.0 + cfg.wear_rate * run_id

    segments_sig:  List[np.ndarray] = []
    segments_step: List[np.ndarray] = []
    segments_ops:  List[np.ndarray] = []
    change_points: List[int] = []
    cursor = 0

    pos: Tuple[float, float, float] = (program.home_x, program.home_y, program.home_z)
    temp = _T_BASE

    for step in program.steps:
        # ±10% duration variation per run
        dur_jitter = int(rng.integers(-step.duration // 10, step.duration // 10 + 1))
        actual_dur = max(20, step.duration + dur_jitter)

        # Maybe inject a fault before this step
        if step.op_state in (CUTTING, RAPID) and rng.uniform() < cfg.fault_prob:
            alarm_id = int(rng.integers(1, 10))
            fault_dur = int(rng.integers(25, 55))
            fsig, temp = _render_fault_burst(fault_dur, pos, alarm_id, cfg.noise_std, rng, temp)
            change_points.append(cursor)
            segments_sig.append(fsig)
            segments_step.append(np.full(fault_dur, step.step_id, dtype=np.int32))
            segments_ops.append(np.full(fault_dur, FAULT, dtype=np.int32))
            cursor += fault_dur

        change_points.append(cursor)
        ssig, temp = _render_step(step, pos, actual_dur, wear_factor, cfg.noise_std, rng, temp)
        segments_sig.append(ssig)
        segments_step.append(np.full(actual_dur, step.step_id, dtype=np.int32))
        segments_ops.append(np.full(actual_dur, step.op_state, dtype=np.int32))

        pos = (step.target_x, step.target_y, step.target_z)
        cursor += actual_dur

    signal         = np.concatenate(segments_sig,  axis=0)
    program_steps  = np.concatenate(segments_step, axis=0).astype(np.int32)
    op_states      = np.concatenate(segments_ops,  axis=0).astype(np.int32)

    # Deduplicate change_points (fault→step can produce consecutive entries)
    change_points = sorted(set(change_points))

    return signal, program_steps, op_states, change_points


# ── dataset generator ─────────────────────────────────────────────────────────

def generate_cnc_dataset(cfg: CNCScenarioConfig, output_dir: Path) -> Path:
    """Generate a full RQ4 dataset for one CNCScenarioConfig.

    Saves:
      windows.npz      — X [N,L,C], program_step_id, op_state_id, run_id,
                         trajectory_id, window_start, is_transition_window,
                         distance_to_boundary
      trajectories/    — per-run raw signal + labels
      metadata.json
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    traj_dir = output_dir / "trajectories"
    traj_dir.mkdir(exist_ok=True)

    rng = np.random.default_rng(cfg.seed)

    programs = [PROGRAMS[pid] for pid in cfg.program_ids]
    n_progs  = len(programs)

    all_windows: List[dict] = []
    total_runs = 0

    for run_id in range(cfg.num_runs):
        program = programs[run_id % n_progs]
        part_type_id = run_id % n_progs

        signal, step_ids, state_ids, change_points = generate_cnc_trajectory(
            program, run_id, cfg, rng
        )

        # Channel subset (for the "simple" scenario with fewer channels)
        if cfg.channel_subset is not None:
            signal = signal[:, cfg.channel_subset]

        np.savez_compressed(
            traj_dir / f"traj_{run_id:03d}.npz",
            signal=signal,
            program_step_id=step_ids,
            op_state_id=state_ids,
            run_id=np.int32(run_id),
            part_type_id=np.int32(part_type_id),
            change_points=np.array(change_points, dtype=np.int64),
        )

        # Extract windows using the step_id as primary mode label
        wins = _extract_windows(
            signal, list(step_ids), change_points,
            run_id, cfg.window_length, cfg.stride,
        )

        # Attach secondary labels: op_state majority per window
        for w in wins:
            ws, we = w["window_start"], w["window_start"] + cfg.window_length
            we = min(we, len(state_ids))
            state_slice = state_ids[ws:we]
            unique, counts = np.unique(state_slice, return_counts=True)
            w["op_state_id"] = int(unique[counts.argmax()])
            w["run_id"] = run_id
            w["part_type_id"] = part_type_id

        all_windows.extend(wins)
        total_runs += 1

    # Stack arrays
    n_ch = signal.shape[1]  # after optional subset
    X             = np.stack([w["x"]              for w in all_windows], axis=0)
    prog_step_ids = np.array([w["mode_id"]         for w in all_windows], dtype=np.int32)
    op_state_ids  = np.array([w["op_state_id"]     for w in all_windows], dtype=np.int32)
    run_ids       = np.array([w["run_id"]           for w in all_windows], dtype=np.int32)
    part_type_ids = np.array([w["part_type_id"]    for w in all_windows], dtype=np.int32)
    traj_ids      = np.array([w["trajectory_id"]   for w in all_windows], dtype=np.int64)
    win_starts    = np.array([w["window_start"]    for w in all_windows], dtype=np.int64)
    is_trans      = np.array([w["is_transition_window"] for w in all_windows], dtype=bool)
    dist_bound    = np.array([w["distance_to_boundary"] for w in all_windows], dtype=np.int64)

    np.savez_compressed(
        output_dir / "windows.npz",
        X=X,
        program_step_id=prog_step_ids,
        op_state_id=op_state_ids,
        run_id=run_ids,
        part_type_id=part_type_ids,
        trajectory_id=traj_ids,
        window_start=win_starts,
        is_transition_window=is_trans,
        distance_to_boundary=dist_bound,
    )

    prog_info = [{"program_id": p.program_id, "name": p.name,
                  "num_steps": len(p.steps)} for p in programs]
    step_names = {str(s.step_id): s.name for s in programs[0].steps}

    meta = {
        "scenario_id":   cfg.scenario_id,
        "programs":      prog_info,
        "step_names":    step_names,
        "num_runs":      total_runs,
        "noise_std":     cfg.noise_std,
        "fault_prob":    cfg.fault_prob,
        "wear_rate":     cfg.wear_rate,
        "num_channels":  n_ch,
        "window_length": cfg.window_length,
        "stride":        cfg.stride,
        "seed":          cfg.seed,
        "num_windows":   len(all_windows),
        "channel_names": (
            ["axis_x","axis_y","axis_z","spindle_speed","feed_rate",
             "vibration_x","vibration_y","temperature","tool_id","alarm_code"]
            if cfg.channel_subset is None
            else [["axis_x","axis_y","axis_z","spindle_speed","feed_rate",
                   "vibration_x","vibration_y","temperature","tool_id","alarm_code"][i]
                  for i in cfg.channel_subset]
        ),
        "discrete_channels": [8, 9] if cfg.channel_subset is None else [],
    }
    (output_dir / "metadata.json").write_text(json.dumps(meta, indent=2))

    return output_dir
