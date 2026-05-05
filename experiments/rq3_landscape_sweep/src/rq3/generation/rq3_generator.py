"""RQ3 dataset generator.

Produces sine / step / event / mixed signal datasets with optional missing-value
masks.  Reuses _extract_windows from the RQ2 periodic_mode_generator without
modification; only the segment *rendering* is new here.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Import _extract_windows from RQ2 without modifying that module
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve()
# rq3_generator.py lives at: <repo>/experiments/rq3_landscape_sweep/src/rq3/generation/
_EXPERIMENTS = _HERE.parent.parent.parent.parent.parent   # <repo>/experiments/
_RQ2_SRC = _EXPERIMENTS / "rq2_trace_comparison" / "src"
_RQ3_SRC = _HERE.parent.parent.parent                     # <repo>/experiments/rq3_landscape_sweep/src/

for _p in [str(_RQ2_SRC), str(_RQ3_SRC)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from rq2.generation.periodic_mode_generator import _extract_windows  # noqa: E402
from rq3.config.scenarios import ScenarioConfig  # noqa: E402


# ---------------------------------------------------------------------------
# Segment length helpers
# ---------------------------------------------------------------------------

_MIN_SEG_STEPS = 36   # minimum timesteps per segment (≥ window_length)
_PERIODS_PER_SEG = 5  # for sine: how many full periods per segment


def _seg_length_sine(mode_id: int, num_modes: int, rng: np.random.Generator) -> int:
    """Variable segment length based on mode frequency (lower freq → longer seg)."""
    base_freq = 0.04 + mode_id * (0.08 / max(num_modes - 1, 1))
    period = max(1, round(1.0 / base_freq))
    return max(_MIN_SEG_STEPS, period * _PERIODS_PER_SEG)


def _seg_length_fixed(rng: np.random.Generator) -> int:
    """Fixed-length segment for step/event (no natural period)."""
    return rng.integers(60, 120)


# ---------------------------------------------------------------------------
# Sine segment (thin wrapper — modes defined by frequency allocation)
# ---------------------------------------------------------------------------

def _make_sine_mode_components(
    num_modes: int, num_channels: int
) -> List[List[List[Tuple[float, float]]]]:
    """Return mode_components[mode_id][channel] = list of (freq, amp) pairs."""
    components: List[List[List[Tuple[float, float]]]] = []
    for m in range(num_modes):
        # Each mode gets a distinct base frequency spread across channels
        f_base = 0.04 + m * (0.10 / max(num_modes - 1, 1))
        ch_comps: List[List[Tuple[float, float]]] = []
        for c in range(num_channels):
            f = f_base + c * 0.005  # slight per-channel detuning
            ch_comps.append([(f, 1.0)])
        components.append(ch_comps)
    return components


def _render_sine_segment(
    mode_id: int,
    length: int,
    num_channels: int,
    mode_components: List[List[List[Tuple[float, float]]]],
    noise_std: float,
    rng: np.random.Generator,
) -> np.ndarray:
    t = np.arange(length, dtype=np.float64)
    signal = np.zeros((length, num_channels), dtype=np.float64)
    phase_offsets = rng.uniform(0, 2 * np.pi, size=num_channels)
    for ch, comps in enumerate(mode_components[mode_id]):
        for freq, amp in comps:
            signal[:, ch] += amp * np.sin(2 * np.pi * freq * t + phase_offsets[ch])
    signal += rng.normal(0.0, noise_std, size=signal.shape)
    return signal.astype(np.float32)


# ---------------------------------------------------------------------------
# Step segment
# ---------------------------------------------------------------------------

def _make_step_amplitudes(
    num_modes: int, num_channels: int, rng: np.random.Generator
) -> np.ndarray:
    """Generate [num_modes, num_channels] amplitude table with uniqueness constraint."""
    for _ in range(100):
        amps = rng.uniform(0.5, 3.0, size=(num_modes, num_channels))
        # Each mode pair must differ by >= 0.3 on at least one channel
        ok = True
        for i in range(num_modes):
            for j in range(i + 1, num_modes):
                if np.max(np.abs(amps[i] - amps[j])) < 0.3:
                    ok = False
                    break
            if not ok:
                break
        if ok:
            return amps
    # Fallback: evenly spaced
    amps = np.zeros((num_modes, num_channels), dtype=np.float32)
    for m in range(num_modes):
        amps[m] = 0.5 + m * (2.0 / max(num_modes - 1, 1))
    return amps


def _render_step_segment(
    mode_id: int,
    length: int,
    num_channels: int,
    mode_amplitudes: np.ndarray,
    noise_std: float,
    rng: np.random.Generator,
) -> np.ndarray:
    signal = np.tile(mode_amplitudes[mode_id], (length, 1)).astype(np.float64)
    signal += rng.normal(0.0, noise_std, size=signal.shape)
    return signal.astype(np.float32)


# ---------------------------------------------------------------------------
# Event (Poisson spike) segment
# ---------------------------------------------------------------------------

def _make_event_rates(
    num_modes: int, num_channels: int
) -> np.ndarray:
    """[num_modes, num_channels] Poisson rates (events per timestep).
    Modes are spread across low/medium/high rate levels."""
    # Rates chosen so mode pairs differ meaningfully
    levels = [0.05, 0.20, 0.50, 0.80, 0.95]
    rates = np.zeros((num_modes, num_channels), dtype=np.float32)
    for m in range(num_modes):
        base = levels[m % len(levels)]
        for c in range(num_channels):
            # stagger channels so cross-channel pattern also varies
            rates[m, c] = np.clip(base + c * 0.04 * (1 if m % 2 == 0 else -1), 0.02, 0.98)
    return rates


def _render_event_segment(
    mode_id: int,
    length: int,
    num_channels: int,
    mode_rates: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    signal = np.zeros((length, num_channels), dtype=np.float32)
    for c in range(num_channels):
        lam = float(mode_rates[mode_id, c])
        signal[:, c] = (rng.uniform(size=length) < lam).astype(np.float32)
    return signal


# ---------------------------------------------------------------------------
# Missing-value mask application
# ---------------------------------------------------------------------------

def _apply_missing_mask(
    X: np.ndarray,            # [N, L, C]
    missing_type: str,
    missing_fraction: float,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (X_with_nan, obs_mask).  obs_mask: True = observed."""
    N, L, C = X.shape
    obs_mask = np.ones((N, L, C), dtype=bool)

    if missing_type == "none" or missing_fraction == 0.0:
        return X.copy(), obs_mask

    X_out = X.copy()

    if missing_type == "random_point":
        drop = rng.uniform(size=(N, L, C)) < missing_fraction
        obs_mask[drop] = False

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
# Mode sequence generator (works for any num_modes)
# ---------------------------------------------------------------------------

def _make_mode_sequence(num_modes: int, num_segments: int, rng: np.random.Generator) -> List[int]:
    """Generate a mode sequence that visits every mode at least once and avoids
    consecutive repeats."""
    if num_segments <= num_modes:
        seq = list(rng.permutation(num_modes))[:num_segments]
        return seq
    seq: List[int] = list(rng.permutation(num_modes))  # visit all modes first
    while len(seq) < num_segments:
        last = seq[-1]
        candidates = [m for m in range(num_modes) if m != last]
        seq.append(int(rng.choice(candidates)))
    return seq


# ---------------------------------------------------------------------------
# Top-level dataset generator
# ---------------------------------------------------------------------------

def generate_rq3_dataset(cfg: ScenarioConfig, output_dir: Path) -> Path:
    """Generate a full RQ3 dataset for one ScenarioConfig.

    Saves:
      windows.npz     — X [N,L,C], mask [N,L,C], mode_id, trajectory_id,
                        window_start, is_transition_window, distance_to_boundary
      trajectories/   — per-trajectory signal + mode timeline
      metadata.json   — config dump + statistics
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    traj_dir = output_dir / "trajectories"
    traj_dir.mkdir(exist_ok=True)

    rng = np.random.default_rng(cfg.seed)

    # Pre-compute mode-level parameters (same across all trajectories)
    if cfg.signal_type in ("sine", "mixed"):
        mode_components = _make_sine_mode_components(cfg.num_modes, cfg.num_channels)
    if cfg.signal_type in ("step", "mixed"):
        mode_amplitudes = _make_step_amplitudes(cfg.num_modes, cfg.num_channels, rng)
    if cfg.signal_type == "event":
        mode_rates = _make_event_rates(cfg.num_modes, cfg.num_channels)

    all_windows: List[dict] = []
    num_segments = max(7, cfg.num_modes + 2)  # enough to visit all modes multiple times

    for traj_id in range(cfg.num_trajectories):
        mode_seq = _make_mode_sequence(cfg.num_modes, num_segments, rng)
        segments = []
        change_points = [0]

        for mode_id in mode_seq:
            if cfg.signal_type == "sine":
                seg_len = _seg_length_sine(mode_id, cfg.num_modes, rng)
                seg = _render_sine_segment(
                    mode_id, seg_len, cfg.num_channels, mode_components,
                    cfg.noise_std, rng
                )
            elif cfg.signal_type == "step":
                seg_len = _seg_length_fixed(rng)
                seg = _render_step_segment(
                    mode_id, seg_len, cfg.num_channels, mode_amplitudes,
                    cfg.noise_std, rng
                )
            elif cfg.signal_type == "event":
                seg_len = _seg_length_fixed(rng)
                seg = _render_event_segment(
                    mode_id, seg_len, cfg.num_channels, mode_rates, rng
                )
            elif cfg.signal_type == "mixed":
                # Channels 0–1: step, channels 2–3: sine
                seg_len = _seg_length_fixed(rng)
                step_part = _render_step_segment(
                    mode_id, seg_len, cfg.num_channels, mode_amplitudes,
                    cfg.noise_std, rng
                )
                sine_part = _render_sine_segment(
                    mode_id, seg_len, cfg.num_channels, mode_components,
                    cfg.noise_std, rng
                )
                half = cfg.num_channels // 2
                seg = step_part.copy()
                seg[:, half:] = sine_part[:, half:]
            else:
                raise ValueError(f"Unknown signal_type: {cfg.signal_type!r}")

            segments.append((seg, mode_id))
            change_points.append(change_points[-1] + len(seg))

        change_points = change_points[:-1]

        signal = np.concatenate([s for s, _ in segments], axis=0)
        mode_labels = []
        for s, m in segments:
            mode_labels.extend([m] * len(s))

        # Save raw trajectory
        np.savez_compressed(
            traj_dir / f"traj_{traj_id:03d}.npz",
            signal=signal,
            mode_labels=np.array(mode_labels, dtype=np.int32),
            change_points=np.array(change_points, dtype=np.int64),
        )

        windows = _extract_windows(
            signal, mode_labels, change_points,
            traj_id, cfg.window_length, cfg.stride
        )
        all_windows.extend(windows)

    # Stack into arrays
    X = np.stack([w["x"] for w in all_windows], axis=0)           # [N, L, C]
    mode_ids = np.array([w["mode_id"] for w in all_windows], dtype=np.int32)
    traj_ids = np.array([w["trajectory_id"] for w in all_windows], dtype=np.int64)
    window_starts = np.array([w["window_start"] for w in all_windows], dtype=np.int64)
    is_trans = np.array([w["is_transition_window"] for w in all_windows], dtype=bool)
    dist_bound = np.array([w["distance_to_boundary"] for w in all_windows], dtype=np.int64)

    # Apply missing mask
    rng_mask = np.random.default_rng(cfg.seed + 9999)
    X_masked, obs_mask = _apply_missing_mask(
        X, cfg.missing_type, cfg.missing_fraction, rng_mask
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
        "scenario_id": cfg.scenario_id,
        "signal_type": cfg.signal_type,
        "num_modes": cfg.num_modes,
        "num_channels": cfg.num_channels,
        "noise_std": cfg.noise_std,
        "missing_type": cfg.missing_type,
        "missing_fraction": cfg.missing_fraction,
        "num_trajectories": cfg.num_trajectories,
        "window_length": cfg.window_length,
        "stride": cfg.stride,
        "seed": cfg.seed,
        "num_windows": int(len(all_windows)),
        "observed_fraction": float(obs_mask.mean()),
    }
    (output_dir / "metadata.json").write_text(json.dumps(meta, indent=2))

    return output_dir
