"""CNC-like periodic mode generator for RQ2 trace comparison experiments.

Generates three synthetic datasets of increasing difficulty:
  P1 — Simple 1D, clearly separated single-frequency modes
  P2 — Multi-channel (4ch), cross-channel frequency mixing
  P3 — Hard multi-channel, similar frequencies + high noise
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Mode definitions
# ---------------------------------------------------------------------------

# Each mode is a list of (frequency, amplitude) pairs per channel.
# Shape: num_modes × num_channels × num_components

PROBLEM_CONFIGS: Dict[str, dict] = {
    "p1_simple_1d": {
        "num_channels": 1,
        "noise_std": 0.05,
        "modes": {
            0: [[(0.04, 1.0)]],           # channel 0: single freq 0.04
            1: [[(0.10, 1.0)]],           # channel 0: single freq 0.10
            2: [[(0.04, 1.0), (0.18, 0.5)]],  # channel 0: composite
        },
    },
    "p2_multichannel": {
        "num_channels": 4,
        "noise_std": 0.08,
        "modes": {
            # mode_id: list of component lists, one per channel
            0: [
                [(0.04, 1.0)],                    # ch0
                [(0.07, 1.0)],                    # ch1
                [(0.10, 1.0)],                    # ch2
                [(0.04, 0.8), (0.10, 0.6)],       # ch3
            ],
            1: [
                [(0.10, 1.0)],
                [(0.04, 1.0)],
                [(0.07, 1.0)],
                [(0.04, 0.7), (0.07, 0.5)],
            ],
            2: [
                [(0.04, 0.9), (0.18, 0.5)],
                [(0.10, 0.8), (0.07, 0.4)],
                [(0.04, 1.0)],
                [(0.07, 0.6), (0.10, 0.6)],
            ],
        },
    },
    "p3_hard_noisy": {
        "num_channels": 4,
        "noise_std": 0.20,
        # Same channel structure as P2 but frequencies compressed closer together
        "modes": {
            0: [
                [(0.06, 1.0)],
                [(0.08, 1.0)],
                [(0.10, 1.0)],
                [(0.06, 0.8), (0.10, 0.6)],
            ],
            1: [
                [(0.10, 1.0)],
                [(0.06, 1.0)],
                [(0.08, 1.0)],
                [(0.06, 0.7), (0.08, 0.5)],
            ],
            2: [
                [(0.06, 0.9), (0.12, 0.5)],
                [(0.10, 0.8), (0.08, 0.4)],
                [(0.06, 1.0)],
                [(0.08, 0.6), (0.10, 0.6)],
            ],
        },
    },
}

# Production sequence: mode_id sequence for each trajectory.
# 7 segments, repeating pattern (like CNC printing A→B→A→C→B→A→C).
PRODUCTION_SEQUENCE = [0, 1, 0, 2, 1, 0, 2]


@dataclass
class PeriodicModeConfig:
    problem: str = "p1_simple_1d"
    seed: int = 42

    num_trajectories: int = 20
    segments_per_trajectory: int = 7
    periods_per_segment: int = 5     # each segment = this many full periods of dominant freq
    min_period_steps: int = 25       # minimum period length in timesteps

    window_length: int = 48
    stride: int = 12

    # Derived — filled in __post_init__
    num_channels: int = field(init=False)
    noise_std: float = field(init=False)

    def __post_init__(self) -> None:
        cfg = PROBLEM_CONFIGS[self.problem]
        self.num_channels = cfg["num_channels"]
        self.noise_std = cfg["noise_std"]


# ---------------------------------------------------------------------------
# Core generation
# ---------------------------------------------------------------------------

def _dominant_period(mode_components: List[List[Tuple[float, float]]]) -> int:
    """Return the period in steps of the lowest frequency component across all channels."""
    freqs = [f for ch in mode_components for (f, _) in ch]
    lowest = min(freqs)
    return max(1, round(1.0 / lowest))


def _render_segment(
    mode_id: int,
    length: int,
    num_channels: int,
    mode_components: List[List[Tuple[float, float]]],
    noise_std: float,
    rng: np.random.Generator,
    phase_offsets: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Render `length` timesteps of mode `mode_id` as [length, num_channels]."""
    t = np.arange(length, dtype=np.float64)
    signal = np.zeros((length, num_channels), dtype=np.float64)

    for ch_idx, components in enumerate(mode_components):
        for freq, amp in components:
            phase = 0.0 if phase_offsets is None else phase_offsets[ch_idx]
            signal[:, ch_idx] += amp * np.sin(2 * np.pi * freq * t + phase)

    noise = rng.normal(0.0, noise_std, size=signal.shape)
    return (signal + noise).astype(np.float32)


def _generate_trajectory(
    problem_cfg: dict,
    mode_sequence: List[int],
    periods_per_segment: int,
    min_period_steps: int,
    noise_std: float,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, List[int], List[int]]:
    """Generate one full trajectory and return (signal, mode_id_per_t, change_points)."""
    modes = problem_cfg["modes"]
    num_channels = problem_cfg["num_channels"]

    segments = []
    change_points = [0]  # first segment starts at t=0

    for mode_id in mode_sequence:
        components = modes[mode_id]
        dom_period = _dominant_period(components)
        seg_len = max(min_period_steps, dom_period) * periods_per_segment
        # Random per-channel phase offsets so repeated mode segments aren't identical
        phase_offsets = rng.uniform(0, 2 * np.pi, size=num_channels)
        seg = _render_segment(
            mode_id, seg_len, num_channels, components, noise_std, rng, phase_offsets
        )
        segments.append((seg, mode_id))
        change_points.append(change_points[-1] + seg_len)

    change_points = change_points[:-1]  # remove trailing sentinel

    signal_parts = [s for s, _ in segments]
    mode_labels = []
    for (s, m) in segments:
        mode_labels.extend([m] * len(s))

    signal = np.concatenate(signal_parts, axis=0)  # [T, C]
    return signal, mode_labels, change_points


def _extract_windows(
    signal: np.ndarray,
    mode_labels: List[int],
    change_points: List[int],
    trajectory_id: int,
    window_length: int,
    stride: int,
) -> List[dict]:
    T = len(signal)
    windows = []
    cp_set = set(change_points)

    for start in range(0, T - window_length + 1, stride):
        end = start + window_length
        x = signal[start:end]  # [L, C]
        labels_in_window = mode_labels[start:end]

        # majority mode label
        unique, counts = np.unique(labels_in_window, return_counts=True)
        mode_id = int(unique[counts.argmax()])

        # transition detection
        boundaries_inside = [cp for cp in change_points if start < cp < end]
        is_transition = len(boundaries_inside) > 0
        if boundaries_inside:
            dist = min(abs(cp - start) for cp in boundaries_inside)
            dist = min(dist, min(abs(end - cp) for cp in boundaries_inside))
        else:
            dist = min(
                (abs(cp - start) for cp in change_points), default=window_length
            )
            dist = int(min(dist, window_length))

        windows.append({
            "x": x,
            "mode_id": mode_id,
            "window_start": start,
            "trajectory_id": trajectory_id,
            "is_transition_window": is_transition,
            "distance_to_boundary": int(dist),
        })

    return windows


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_dataset(
    config: PeriodicModeConfig,
    output_dir: Path,
    mode_sequence: Optional[List[int]] = None,
) -> Path:
    """Generate a full periodic-mode dataset and save to output_dir.

    Outputs:
      windows.npz        — window arrays + per-window metadata
      trajectories/      — per-trajectory raw signal + mode timeline
      metadata.json      — dataset config summary
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    traj_dir = output_dir / "trajectories"
    traj_dir.mkdir(exist_ok=True)

    rng = np.random.default_rng(config.seed)
    problem_cfg = PROBLEM_CONFIGS[config.problem]

    if mode_sequence is None:
        mode_sequence = PRODUCTION_SEQUENCE[: config.segments_per_trajectory]

    all_windows: List[dict] = []

    for traj_id in range(config.num_trajectories):
        signal, mode_labels, change_points = _generate_trajectory(
            problem_cfg=problem_cfg,
            mode_sequence=mode_sequence,
            periods_per_segment=config.periods_per_segment,
            min_period_steps=config.min_period_steps,
            noise_std=config.noise_std,
            rng=rng,
        )

        # Save full trajectory
        np.save(traj_dir / f"traj_{traj_id:03d}_signal.npy", signal)
        np.save(
            traj_dir / f"traj_{traj_id:03d}_mode_labels.npy",
            np.array(mode_labels, dtype=np.int32),
        )
        with open(traj_dir / f"traj_{traj_id:03d}_timeline.json", "w") as f:
            json.dump(
                {"trajectory_id": traj_id, "change_points": change_points, "mode_sequence": mode_sequence},
                f,
            )

        windows = _extract_windows(
            signal=signal,
            mode_labels=mode_labels,
            change_points=change_points,
            trajectory_id=traj_id,
            window_length=config.window_length,
            stride=config.stride,
        )
        all_windows.extend(windows)

    # Pack into arrays
    X = np.stack([w["x"] for w in all_windows], axis=0)  # [N, L, C]
    mode_id_arr = np.array([w["mode_id"] for w in all_windows], dtype=np.int32)
    window_start_arr = np.array([w["window_start"] for w in all_windows], dtype=np.int64)
    traj_id_arr = np.array([w["trajectory_id"] for w in all_windows], dtype=np.int64)
    is_trans_arr = np.array([w["is_transition_window"] for w in all_windows], dtype=bool)
    dist_arr = np.array([w["distance_to_boundary"] for w in all_windows], dtype=np.int64)

    np.savez_compressed(
        output_dir / "windows.npz",
        X=X,
        mode_id=mode_id_arr,
        window_start=window_start_arr,
        trajectory_id=traj_id_arr,
        is_transition_window=is_trans_arr,
        distance_to_boundary=dist_arr,
    )

    metadata = {
        "config": asdict(config),
        "problem": config.problem,
        "mode_sequence": mode_sequence,
        "num_trajectories": config.num_trajectories,
        "num_windows": len(all_windows),
        "window_shape": list(X.shape[1:]),
        "unique_mode_ids": sorted(np.unique(mode_id_arr).tolist()),
        "transition_window_fraction": float(is_trans_arr.mean()),
        "num_channels": config.num_channels,
    }
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(
        f"[generate] {config.problem}: {config.num_trajectories} trajectories, "
        f"{len(all_windows)} windows, shape {X.shape}"
    )
    return output_dir


def create_splits(
    data_dir: Path,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> Dict[str, dict]:
    """Split windows by trajectory_id (leakage-aware). Returns dict of split→arrays."""
    data = np.load(data_dir / "windows.npz")
    traj_ids = data["trajectory_id"]
    unique_trajs = np.unique(traj_ids)

    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(unique_trajs)

    n = len(shuffled)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    splits = {
        "train": shuffled[:n_train],
        "val": shuffled[n_train : n_train + n_val],
        "test": shuffled[n_train + n_val :],
    }

    result = {}
    for name, traj_subset in splits.items():
        mask = np.isin(traj_ids, traj_subset)
        result[name] = {k: data[k][mask] for k in data.files}

    return result
