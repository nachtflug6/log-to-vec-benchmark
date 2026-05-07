"""RQ6 dataset generator.

Produces sawtooth / damped-sine / chirp / ratio-mixed signal datasets.
Reuses _extract_windows from RQ2 and the missing-value infrastructure from RQ3.

New signal types vs RQ3:
  sawtooth  — periodic with rich harmonic content (vs FFT-friendly pure sine)
  damped    — amplitude-decaying sine (non-stationary; τ ≈ half the segment length)
  chirp     — linearly frequency-modulated sine (time-varying spectral content)
  ratio     — configurable fraction of step vs sine channels (step_channels field)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np

_HERE = Path(__file__).resolve()
# rq6_generator.py: <repo>/experiments/rq6_signal_ablation/src/rq6/generation/
_EXPERIMENTS = _HERE.parent.parent.parent.parent.parent   # <repo>/experiments/
_RQ2_SRC = _EXPERIMENTS / "rq2_trace_comparison" / "src"
_RQ6_SRC = _HERE.parent.parent.parent                     # <repo>/experiments/rq6_signal_ablation/src/

for _p in [str(_RQ2_SRC), str(_RQ6_SRC)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from rq2.generation.periodic_mode_generator import _extract_windows  # noqa: E402
from rq6.config.scenarios import ScenarioConfig  # noqa: E402


# ---------------------------------------------------------------------------
# Segment length helpers (shared with RQ3)
# ---------------------------------------------------------------------------

_MIN_SEG_STEPS = 36
_PERIODS_PER_SEG = 5


def _seg_length_freq(mode_id: int, num_modes: int, rng: np.random.Generator) -> int:
    """Variable segment length based on mode frequency (5 periods of base freq)."""
    base_freq = 0.04 + mode_id * (0.08 / max(num_modes - 1, 1))
    period = max(1, round(1.0 / base_freq))
    return max(_MIN_SEG_STEPS, period * _PERIODS_PER_SEG)


def _seg_length_fixed(rng: np.random.Generator) -> int:
    return rng.integers(60, 120)


# ---------------------------------------------------------------------------
# Shared mode-parameter factories
# ---------------------------------------------------------------------------

def _make_sine_mode_components(
    num_modes: int, num_channels: int
) -> List[List[List[Tuple[float, float]]]]:
    """mode_components[mode_id][channel] = list of (freq, amp) pairs."""
    components: List[List[List[Tuple[float, float]]]] = []
    for m in range(num_modes):
        f_base = 0.04 + m * (0.10 / max(num_modes - 1, 1))
        ch_comps: List[List[Tuple[float, float]]] = []
        for c in range(num_channels):
            f = f_base + c * 0.005
            ch_comps.append([(f, 1.0)])
        components.append(ch_comps)
    return components


def _make_step_amplitudes(
    num_modes: int, num_channels: int, rng: np.random.Generator
) -> np.ndarray:
    for _ in range(100):
        amps = rng.uniform(0.5, 3.0, size=(num_modes, num_channels))
        ok = all(
            np.max(np.abs(amps[i] - amps[j])) >= 0.3
            for i in range(num_modes)
            for j in range(i + 1, num_modes)
        )
        if ok:
            return amps
    amps = np.zeros((num_modes, num_channels), dtype=np.float32)
    for m in range(num_modes):
        amps[m] = 0.5 + m * (2.0 / max(num_modes - 1, 1))
    return amps


def _make_chirp_ranges(
    num_modes: int, num_channels: int
) -> List[List[Tuple[float, float]]]:
    """mode_ranges[mode_id][channel] = (f_low, f_high) for linear chirp."""
    ranges = []
    for m in range(num_modes):
        f_center = 0.04 + m * (0.08 / max(num_modes - 1, 1))
        ch_ranges = []
        for c in range(num_channels):
            f_low  = max(0.01, f_center - 0.02 + c * 0.003)
            f_high = min(0.45, f_center + 0.04 + c * 0.003)
            ch_ranges.append((f_low, f_high))
        ranges.append(ch_ranges)
    return ranges


# ---------------------------------------------------------------------------
# Segment renderers — new signal types
# ---------------------------------------------------------------------------

def _render_sawtooth_segment(
    mode_id: int,
    length: int,
    num_channels: int,
    mode_components: List[List[List[Tuple[float, float]]]],
    noise_std: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sawtooth: (t/T mod 1) × 2 − 1.  Same frequency allocation as sine."""
    t = np.arange(length, dtype=np.float64)
    signal = np.zeros((length, num_channels), dtype=np.float64)
    for ch, comps in enumerate(mode_components[mode_id]):
        for freq, amp in comps:
            period = 1.0 / max(freq, 1e-6)
            signal[:, ch] += amp * 2.0 * ((t / period) % 1.0 - 0.5)
    signal += rng.normal(0.0, noise_std, size=signal.shape)
    return signal.astype(np.float32)


def _render_damped_segment(
    mode_id: int,
    length: int,
    num_channels: int,
    mode_components: List[List[List[Tuple[float, float]]]],
    noise_std: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Damped sine: A·exp(−t/τ)·sin(2πft).  τ = half the segment length."""
    t = np.arange(length, dtype=np.float64)
    tau = max(5.0, length * 0.5)
    envelope = np.exp(-t / tau)
    signal = np.zeros((length, num_channels), dtype=np.float64)
    phase_offsets = rng.uniform(0, 2 * np.pi, size=num_channels)
    for ch, comps in enumerate(mode_components[mode_id]):
        for freq, amp in comps:
            signal[:, ch] += amp * envelope * np.sin(2 * np.pi * freq * t + phase_offsets[ch])
    signal += rng.normal(0.0, noise_std, size=signal.shape)
    return signal.astype(np.float32)


def _render_chirp_segment(
    mode_id: int,
    length: int,
    num_channels: int,
    mode_ranges: List[List[Tuple[float, float]]],
    noise_std: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Linear chirp: frequency sweeps from f_low to f_high over the segment."""
    t = np.arange(length, dtype=np.float64)
    signal = np.zeros((length, num_channels), dtype=np.float64)
    phase_offsets = rng.uniform(0, 2 * np.pi, size=num_channels)
    for ch, (f_low, f_high) in enumerate(mode_ranges[mode_id]):
        # Instantaneous phase = integral of 2π·f(t) dt
        # f(t) = f_low + (f_high - f_low) · t / length
        inst_phase = 2 * np.pi * (
            f_low * t + 0.5 * (f_high - f_low) * t ** 2 / max(length, 1)
        )
        signal[:, ch] = np.sin(inst_phase + phase_offsets[ch])
    signal += rng.normal(0.0, noise_std, size=signal.shape)
    return signal.astype(np.float32)


def _render_ratio_segment(
    mode_id: int,
    length: int,
    num_channels: int,
    step_channels: int,
    mode_amplitudes: np.ndarray,
    mode_components: List[List[List[Tuple[float, float]]]],
    noise_std: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Mixed rendering: first `step_channels` channels carry step, rest carry sine."""
    seg = np.zeros((length, num_channels), dtype=np.float64)
    if step_channels > 0:
        step = np.tile(mode_amplitudes[mode_id, :step_channels], (length, 1)).astype(np.float64)
        step += rng.normal(0.0, noise_std, size=step.shape)
        seg[:, :step_channels] = step
    if step_channels < num_channels:
        t = np.arange(length, dtype=np.float64)
        n_sine = num_channels - step_channels
        phase_offsets = rng.uniform(0, 2 * np.pi, size=n_sine)
        for i, c in enumerate(range(step_channels, num_channels)):
            for freq, amp in mode_components[mode_id][c]:
                seg[:, c] += amp * np.sin(2 * np.pi * freq * t + phase_offsets[i])
        seg[:, step_channels:] += rng.normal(0.0, noise_std, size=(length, n_sine))
    return seg.astype(np.float32)


# ---------------------------------------------------------------------------
# Missing-value mask (same as RQ3)
# ---------------------------------------------------------------------------

def _apply_missing_mask(
    X: np.ndarray,
    missing_type: str,
    missing_fraction: float,
    rng: np.random.Generator,
):
    N, L, C = X.shape
    obs_mask = np.ones((N, L, C), dtype=bool)
    if missing_type == "none" or missing_fraction == 0.0:
        return X.copy(), obs_mask
    X_out = X.copy()
    if missing_type == "random_point":
        obs_mask[rng.uniform(size=(N, L, C)) < missing_fraction] = False
    elif missing_type == "contiguous_block":
        block_len = max(1, round(missing_fraction * L))
        for i in range(N):
            start = rng.integers(0, max(1, L - block_len + 1))
            obs_mask[i, start:start + block_len, :] = False
    elif missing_type == "channel_dropout":
        for i in range(N):
            for c in range(C):
                if rng.uniform() < missing_fraction:
                    obs_mask[i, :, c] = False
    X_out[~obs_mask] = np.nan
    return X_out, obs_mask


# ---------------------------------------------------------------------------
# Mode sequence
# ---------------------------------------------------------------------------

def _make_mode_sequence(num_modes: int, num_segments: int, rng: np.random.Generator) -> List[int]:
    if num_segments <= num_modes:
        return list(rng.permutation(num_modes))[:num_segments]
    seq: List[int] = list(rng.permutation(num_modes))
    while len(seq) < num_segments:
        last = seq[-1]
        candidates = [m for m in range(num_modes) if m != last]
        seq.append(int(rng.choice(candidates)))
    return seq


# ---------------------------------------------------------------------------
# Top-level dataset generator
# ---------------------------------------------------------------------------

def generate_rq6_dataset(cfg: ScenarioConfig, output_dir: Path) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    traj_dir = output_dir / "trajectories"
    traj_dir.mkdir(exist_ok=True)

    rng = np.random.default_rng(cfg.seed)

    # Pre-compute mode-level parameters
    mode_components = _make_sine_mode_components(cfg.num_modes, cfg.num_channels)
    mode_amplitudes = _make_step_amplitudes(cfg.num_modes, cfg.num_channels, rng)
    mode_chirp_ranges = _make_chirp_ranges(cfg.num_modes, cfg.num_channels)

    all_windows: List[dict] = []
    num_segments = max(7, cfg.num_modes + 2)

    for traj_id in range(cfg.num_trajectories):
        mode_seq = _make_mode_sequence(cfg.num_modes, num_segments, rng)
        segments = []
        change_points = [0]

        for mode_id in mode_seq:
            if cfg.signal_type == "sawtooth":
                seg_len = _seg_length_freq(mode_id, cfg.num_modes, rng)
                seg = _render_sawtooth_segment(
                    mode_id, seg_len, cfg.num_channels, mode_components,
                    cfg.noise_std, rng,
                )
            elif cfg.signal_type == "damped":
                seg_len = _seg_length_freq(mode_id, cfg.num_modes, rng)
                seg = _render_damped_segment(
                    mode_id, seg_len, cfg.num_channels, mode_components,
                    cfg.noise_std, rng,
                )
            elif cfg.signal_type == "chirp":
                seg_len = _seg_length_fixed(rng)
                seg = _render_chirp_segment(
                    mode_id, seg_len, cfg.num_channels, mode_chirp_ranges,
                    cfg.noise_std, rng,
                )
            elif cfg.signal_type == "ratio":
                seg_len = _seg_length_fixed(rng)
                seg = _render_ratio_segment(
                    mode_id, seg_len, cfg.num_channels, cfg.step_channels,
                    mode_amplitudes, mode_components, cfg.noise_std, rng,
                )
            else:
                raise ValueError(f"Unknown signal_type: {cfg.signal_type!r}")

            segments.append((seg, mode_id))
            change_points.append(change_points[-1] + len(seg))

        change_points = change_points[:-1]

        signal = np.concatenate([s for s, _ in segments], axis=0)
        mode_labels = []
        for s, m in segments:
            mode_labels.extend([m] * len(s))

        np.savez_compressed(
            traj_dir / f"traj_{traj_id:03d}.npz",
            signal=signal,
            mode_labels=np.array(mode_labels, dtype=np.int32),
            change_points=np.array(change_points, dtype=np.int64),
        )

        windows = _extract_windows(
            signal, mode_labels, change_points,
            traj_id, cfg.window_length, cfg.stride,
        )
        all_windows.extend(windows)

    X = np.stack([w["x"] for w in all_windows], axis=0)
    mode_ids = np.array([w["mode_id"] for w in all_windows], dtype=np.int32)
    traj_ids = np.array([w["trajectory_id"] for w in all_windows], dtype=np.int64)
    window_starts = np.array([w["window_start"] for w in all_windows], dtype=np.int64)
    is_trans = np.array([w["is_transition_window"] for w in all_windows], dtype=bool)
    dist_bound = np.array([w["distance_to_boundary"] for w in all_windows], dtype=np.int64)

    rng_mask = np.random.default_rng(cfg.seed + 9999)
    X_masked, obs_mask = _apply_missing_mask(
        X, cfg.missing_type, cfg.missing_fraction, rng_mask,
    )

    np.savez_compressed(
        output_dir / "windows.npz",
        X=X_masked,
        mask=obs_mask,
        mode_id=mode_ids,
        trajectory_id=traj_ids,
        window_start=window_starts,
        is_transition_window=is_trans,
        distance_to_boundary=dist_bound,
    )

    meta = {
        "scenario_id":       cfg.scenario_id,
        "signal_type":       cfg.signal_type,
        "step_channels":     cfg.step_channels,
        "num_modes":         cfg.num_modes,
        "num_channels":      cfg.num_channels,
        "noise_std":         cfg.noise_std,
        "missing_type":      cfg.missing_type,
        "missing_fraction":  cfg.missing_fraction,
        "num_trajectories":  cfg.num_trajectories,
        "window_length":     cfg.window_length,
        "stride":            cfg.stride,
        "seed":              cfg.seed,
        "num_windows":       int(len(all_windows)),
        "observed_fraction": float(obs_mask.mean()),
    }
    (output_dir / "metadata.json").write_text(json.dumps(meta, indent=2))

    return output_dir
