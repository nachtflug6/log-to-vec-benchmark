from __future__ import annotations

"""
Factorized Switched State-Space Synthetic Generator

This generator supports multiple curriculum variants:
- easy_clean
- easy_clean_with_noise
- factorized_clean
- factorized_noisy
- full
"""

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class GeneratorConfig:
    # -------------------------------------------------------------------------
    # General
    # -------------------------------------------------------------------------
    seed: int = 42
    variant: str = "full"

    num_trajectories: int = 120
    trajectory_length: int = 320
    latent_dim: int = 6
    num_channels: int = 4

    # -------------------------------------------------------------------------
    # Regime schedule
    # -------------------------------------------------------------------------
    min_segments: int = 3
    max_segments: int = 6
    min_dwell: int = 56
    max_dwell: int = 112
    switching_enabled: bool = True

    # -------------------------------------------------------------------------
    # Noise
    # -------------------------------------------------------------------------
    process_noise_std: float = 0.03
    observation_noise_std: float = 0.02

    # -------------------------------------------------------------------------
    # Continuous nuisance factor
    # -------------------------------------------------------------------------
    load_min: float = 0.8
    load_max: float = 1.2
    load_curve_enabled: bool = True

    # -------------------------------------------------------------------------
    # Device-specific nuisance corruption
    # -------------------------------------------------------------------------
    num_devices: int = 6
    device_gain_jitter: float = 0.08
    device_bias_jitter: float = 0.10
    cross_channel_mix: float = 0.03
    device_effects_enabled: bool = True

    # -------------------------------------------------------------------------
    # Mild corruption
    # -------------------------------------------------------------------------
    channel_dropout_prob: float = 0.01
    offset_shift_prob: float = 0.10
    offset_shift_std: float = 0.05
    corruption_enabled: bool = True

    # -------------------------------------------------------------------------
    # Window extraction
    # -------------------------------------------------------------------------
    window_length: int = 48
    stride: int = 12
    transition_margin: int = 4
    keep_transition_windows: bool = True

    # -------------------------------------------------------------------------
    # Transition sharpness
    # -------------------------------------------------------------------------
    transition_blend: int = 6

    # -------------------------------------------------------------------------
    # Optional factor controls
    # -------------------------------------------------------------------------
    fixed_coupling_name: Optional[str] = None
    fixed_spectral_name: Optional[str] = None

    def __post_init__(self) -> None:
        self._apply_variant_defaults()

    def _apply_variant_defaults(self) -> None:
        """
        Apply curriculum-style defaults.
        The user can still override values from CLI after construction if desired.
        """
        if self.variant == "easy_clean":
            # Pure structure, almost no nuisance, no switching.
            self.switching_enabled = False
            self.min_segments = 1
            self.max_segments = 1
            self.min_dwell = self.trajectory_length
            self.max_dwell = self.trajectory_length

            self.fixed_coupling_name = "medium"
            self.fixed_spectral_name = None

            self.process_noise_std = 0.005
            self.observation_noise_std = 0.005

            self.load_min = 1.0
            self.load_max = 1.0
            self.load_curve_enabled = False

            self.num_devices = 1
            self.device_gain_jitter = 0.0
            self.device_bias_jitter = 0.0
            self.cross_channel_mix = 0.0
            self.device_effects_enabled = False

            self.channel_dropout_prob = 0.0
            self.offset_shift_prob = 0.0
            self.offset_shift_std = 0.0
            self.corruption_enabled = False

            self.transition_blend = 0
            self.keep_transition_windows = True

        elif self.variant == "easy_clean_with_noise":
            # Same basic structure as easy_clean, but adds mild nuisance/noise.
            self.switching_enabled = False
            self.min_segments = 1
            self.max_segments = 1
            self.min_dwell = self.trajectory_length
            self.max_dwell = self.trajectory_length

            self.fixed_coupling_name = "medium"
            self.fixed_spectral_name = None

            self.process_noise_std = 0.02
            self.observation_noise_std = 0.02

            self.load_min = 1.0
            self.load_max = 1.0
            self.load_curve_enabled = False

            self.num_devices = 3
            self.device_gain_jitter = 0.03
            self.device_bias_jitter = 0.03
            self.cross_channel_mix = 0.01
            self.device_effects_enabled = True

            self.channel_dropout_prob = 0.0
            self.offset_shift_prob = 0.05
            self.offset_shift_std = 0.02
            self.corruption_enabled = True

            self.transition_blend = 0
            self.keep_transition_windows = True

        elif self.variant == "factorized_clean":
            # Spectral + coupling + switching, but no nuisance.
            self.switching_enabled = True
            self.min_segments = 3
            self.max_segments = 5
            self.min_dwell = 56
            self.max_dwell = 112

            self.fixed_coupling_name = None
            self.fixed_spectral_name = None

            self.process_noise_std = 0.01
            self.observation_noise_std = 0.01

            self.load_min = 1.0
            self.load_max = 1.0
            self.load_curve_enabled = False

            self.num_devices = 1
            self.device_gain_jitter = 0.0
            self.device_bias_jitter = 0.0
            self.cross_channel_mix = 0.0
            self.device_effects_enabled = False

            self.channel_dropout_prob = 0.0
            self.offset_shift_prob = 0.0
            self.offset_shift_std = 0.0
            self.corruption_enabled = False

            self.transition_blend = 0
            self.keep_transition_windows = True

        elif self.variant == "factorized_noisy":
            # Spectral + coupling + switching + moderate nuisance.
            self.switching_enabled = True
            self.min_segments = 3
            self.max_segments = 6
            self.min_dwell = 56
            self.max_dwell = 112

            self.fixed_coupling_name = None
            self.fixed_spectral_name = None

            self.process_noise_std = 0.025
            self.observation_noise_std = 0.02

            self.load_min = 0.9
            self.load_max = 1.1
            self.load_curve_enabled = True

            self.num_devices = 4
            self.device_gain_jitter = 0.04
            self.device_bias_jitter = 0.04
            self.cross_channel_mix = 0.015
            self.device_effects_enabled = True

            self.channel_dropout_prob = 0.005
            self.offset_shift_prob = 0.05
            self.offset_shift_std = 0.02
            self.corruption_enabled = True

            self.transition_blend = 3
            self.keep_transition_windows = True

        elif self.variant == "full":
            # Original behavior.
            pass

        else:
            raise ValueError(
                f"Unknown variant '{self.variant}'. "
                f"Choose from: easy_clean, easy_clean_with_noise, "
                f"factorized_clean, factorized_noisy, full"
            )


# -----------------------------------------------------------------------------
# Main generator
# -----------------------------------------------------------------------------


class FactorizedSwitchingGenerator:
    """Generate a factorized switched state-space benchmark."""

    SPECTRAL_FAMILIES = [
        "clean_oscillatory",
        "damped_oscillatory",
        "multi_component",
        "quasi_aperiodic",
    ]
    COUPLING_LEVELS = ["low", "medium", "high"]

    def __init__(self, config: GeneratorConfig):
        self.config = config
        self.rng = np.random.default_rng(config.seed)

        if config.latent_dim % 2 != 0:
            raise ValueError("latent_dim must be even so latent states can be built from 2x2 blocks.")

        self._spectral_ids = {name: i for i, name in enumerate(self.SPECTRAL_FAMILIES)}
        self._coupling_ids = {name: i for i, name in enumerate(self.COUPLING_LEVELS)}
        self.device_params = self._build_devices()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_dataset(self) -> Dict[str, object]:
        trajectories = []
        step_rows = []

        for trajectory_id in range(self.config.num_trajectories):
            traj = self._generate_single_trajectory(trajectory_id)
            trajectories.append(traj)
            step_rows.extend(self._trajectory_to_rows(traj))

        windows = self.extract_windows(trajectories)
        df = pd.DataFrame(step_rows)

        return {
            "trajectories": trajectories,
            "trajectory_df": df,
            "windows": windows,
            "metadata": self._build_dataset_metadata(trajectories, windows),
        }

    def extract_windows(self, trajectories: List[Dict[str, object]]) -> Dict[str, np.ndarray]:
        cfg = self.config

        X_list = []
        trajectory_ids = []
        device_ids = []
        start_indices = []
        full_mode_ids = []
        spectral_ids = []
        coupling_ids = []
        mean_loads = []
        is_transition_window = []
        distance_to_boundary = []
        left_mode_ids = []
        right_mode_ids = []

        for traj in trajectories:
            y = traj["observed"]
            mode_id = traj["mode_id_per_t"]
            spectral_id = traj["spectral_id_per_t"]
            coupling_id = traj["coupling_id_per_t"]
            load = traj["load_per_t"]
            boundaries = np.array(traj["boundary_indices"], dtype=np.int64)

            T = y.shape[0]
            for start in range(0, T - cfg.window_length + 1, cfg.stride):
                end = start + cfg.window_length
                xw = y[start:end]

                mode_majority = self._majority_label(mode_id[start:end])
                spectral_majority = self._majority_label(spectral_id[start:end])
                coupling_majority = self._majority_label(coupling_id[start:end])
                mean_load = float(np.mean(load[start:end]))

                inside = boundaries[(boundaries >= start) & (boundaries < end)]
                transition = len(inside) > 0
                if transition:
                    b = int(inside[0])
                    dist = min(abs(b - start), abs(end - 1 - b))
                    left_mode = int(mode_id[max(start, b - 1)])
                    right_mode = int(mode_id[min(end - 1, b)])
                else:
                    dist = cfg.window_length
                    left_mode = -1
                    right_mode = -1

                if (not cfg.keep_transition_windows) and transition:
                    continue

                X_list.append(xw.astype(np.float32))
                trajectory_ids.append(int(traj["trajectory_id"]))
                device_ids.append(int(traj["device_id"]))
                start_indices.append(int(start))
                full_mode_ids.append(int(mode_majority))
                spectral_ids.append(int(spectral_majority))
                coupling_ids.append(int(coupling_majority))
                mean_loads.append(mean_load)
                is_transition_window.append(bool(transition))
                distance_to_boundary.append(int(dist))
                left_mode_ids.append(int(left_mode))
                right_mode_ids.append(int(right_mode))

        return {
            "X": np.stack(X_list, axis=0),
            "trajectory_id": np.array(trajectory_ids, dtype=np.int64),
            "device_id": np.array(device_ids, dtype=np.int64),
            "window_start": np.array(start_indices, dtype=np.int64),
            "mode_id": np.array(full_mode_ids, dtype=np.int64),
            "spectral_id": np.array(spectral_ids, dtype=np.int64),
            "coupling_id": np.array(coupling_ids, dtype=np.int64),
            "mean_load": np.array(mean_loads, dtype=np.float32),
            "is_transition_window": np.array(is_transition_window, dtype=np.bool_),
            "distance_to_boundary": np.array(distance_to_boundary, dtype=np.int64),
            "left_mode_id": np.array(left_mode_ids, dtype=np.int64),
            "right_mode_id": np.array(right_mode_ids, dtype=np.int64),
        }

    # ------------------------------------------------------------------
    # Device model
    # ------------------------------------------------------------------

    def _build_devices(self) -> List[Dict[str, np.ndarray]]:
        cfg = self.config
        devices = []

        if not cfg.device_effects_enabled:
            C = np.eye(cfg.num_channels, cfg.latent_dim, dtype=np.float32)
            bias = np.zeros(cfg.num_channels, dtype=np.float32)
            devices.append({
                "device_id": 0,
                "C": C,
                "bias": bias,
            })
            return devices

        for device_id in range(cfg.num_devices):
            diag = 1.0 + self.rng.uniform(
                -cfg.device_gain_jitter,
                cfg.device_gain_jitter,
                size=cfg.num_channels
            )

            C = np.eye(cfg.num_channels, cfg.latent_dim, dtype=np.float32)
            C[: cfg.num_channels, : cfg.num_channels] *= diag[:, None]

            for i in range(cfg.num_channels):
                for j in range(cfg.num_channels):
                    if i != j:
                        C[i, j] += self.rng.uniform(
                            -cfg.cross_channel_mix,
                            cfg.cross_channel_mix
                        )

            bias = self.rng.normal(
                0.0,
                cfg.device_bias_jitter,
                size=cfg.num_channels
            ).astype(np.float32)

            devices.append({
                "device_id": device_id,
                "C": C.astype(np.float32),
                "bias": bias,
            })

        return devices

    # ------------------------------------------------------------------
    # Trajectory generation
    # ------------------------------------------------------------------

    def _generate_single_trajectory(self, trajectory_id: int) -> Dict[str, object]:
        cfg = self.config
        T = cfg.trajectory_length
        H = cfg.latent_dim

        device = self.device_params[self.rng.integers(0, len(self.device_params))]
        device_id = int(device["device_id"])
        device_C = device["C"]
        device_bias = device["bias"]

        if cfg.load_curve_enabled:
            load_base = float(self.rng.uniform(cfg.load_min, cfg.load_max))
            load_per_t = self._generate_load_curve(T, load_base)
        else:
            load_per_t = np.full(T, fill_value=cfg.load_min, dtype=np.float32)

        segments = self._sample_segments(T)
        mode_id_per_t = np.zeros(T, dtype=np.int64)
        spectral_id_per_t = np.zeros(T, dtype=np.int64)
        coupling_id_per_t = np.zeros(T, dtype=np.int64)
        segment_id_per_t = np.zeros(T, dtype=np.int64)

        mode_vocab = []
        boundaries = []

        for seg_idx, seg in enumerate(segments):
            s, e = seg["start"], seg["end"]
            spectral_name = seg["spectral_family"]
            coupling_name = seg["coupling_strength"]
            spectral_id = self._spectral_ids[spectral_name]
            coupling_id = self._coupling_ids[coupling_name]
            mode_id = spectral_id * len(self.COUPLING_LEVELS) + coupling_id

            mode_id_per_t[s:e] = mode_id
            spectral_id_per_t[s:e] = spectral_id
            coupling_id_per_t[s:e] = coupling_id
            segment_id_per_t[s:e] = seg_idx

            mode_vocab.append({
                "segment_id": seg_idx,
                "start": s,
                "end": e,
                "spectral_family": spectral_name,
                "coupling_strength": coupling_name,
                "mode_id": mode_id,
            })

            if seg_idx > 0:
                boundaries.append(s)

        x = np.zeros((T, H), dtype=np.float32)
        y = np.zeros((T, cfg.num_channels), dtype=np.float32)
        transition_mask = np.zeros(T, dtype=np.int64)

        x_prev = self.rng.normal(0.0, 0.4, size=H)

        for t in range(T):
            spectral_name = self.SPECTRAL_FAMILIES[int(spectral_id_per_t[t])]
            coupling_name = self.COUPLING_LEVELS[int(coupling_id_per_t[t])]
            load_t = float(load_per_t[t])

            A_curr = self._build_transition_aware_matrix(
                t=t,
                boundaries=boundaries,
                spectral_name=spectral_name,
                coupling_name=coupling_name,
                load=load_t,
                spectral_id_per_t=spectral_id_per_t,
                coupling_id_per_t=coupling_id_per_t,
            )

            process_noise = self.rng.normal(0.0, cfg.process_noise_std, size=H)
            x_curr = A_curr @ x_prev + process_noise

            obs_noise = self.rng.normal(0.0, cfg.observation_noise_std, size=cfg.num_channels)
            y_curr = device_C @ x_curr + device_bias + obs_noise

            if cfg.corruption_enabled:
                if self.rng.random() < cfg.channel_dropout_prob:
                    drop_idx = int(self.rng.integers(0, cfg.num_channels))
                    y_curr[drop_idx] = 0.0

                if self.rng.random() < cfg.offset_shift_prob:
                    y_curr += self.rng.normal(0.0, cfg.offset_shift_std, size=cfg.num_channels)

            x[t] = x_curr.astype(np.float32)
            y[t] = y_curr.astype(np.float32)
            x_prev = x_curr

        for b in boundaries:
            left = max(0, b - cfg.transition_margin)
            right = min(T, b + cfg.transition_margin + 1)
            transition_mask[left:right] = 1

        return {
            "trajectory_id": trajectory_id,
            "device_id": device_id,
            "observed": y,
            "latent": x,
            "mode_id_per_t": mode_id_per_t,
            "spectral_id_per_t": spectral_id_per_t,
            "coupling_id_per_t": coupling_id_per_t,
            "load_per_t": load_per_t.astype(np.float32),
            "segment_id_per_t": segment_id_per_t,
            "transition_mask": transition_mask,
            "boundary_indices": boundaries,
            "segments": mode_vocab,
        }

    def _sample_segments(self, total_length: int) -> List[Dict[str, object]]:
        cfg = self.config

        if not cfg.switching_enabled:
            pair = self._sample_mode_pair(prev_pair=None)
            return [{
                "start": 0,
                "end": total_length,
                "spectral_family": pair[0],
                "coupling_strength": pair[1],
            }]

        n_segments = int(self.rng.integers(cfg.min_segments, cfg.max_segments + 1))
        raw = self.rng.integers(cfg.min_dwell, cfg.max_dwell + 1, size=n_segments)
        scaled = np.maximum(cfg.min_dwell, (raw / raw.sum() * total_length).astype(int))

        diff = total_length - int(scaled.sum())
        scaled[-1] += diff

        while scaled[-1] < cfg.min_dwell:
            donor = int(np.argmax(scaled[:-1]))
            if scaled[donor] <= cfg.min_dwell:
                break
            scaled[donor] -= 1
            scaled[-1] += 1

        segments = []
        cursor = 0
        prev_pair = None

        for dwell in scaled:
            pair = self._sample_mode_pair(prev_pair)
            start = cursor
            end = min(total_length, cursor + int(dwell))
            cursor = end
            segments.append({
                "start": start,
                "end": end,
                "spectral_family": pair[0],
                "coupling_strength": pair[1],
            })
            prev_pair = pair

        segments[-1]["end"] = total_length
        return segments

    def _sample_mode_pair(self, prev_pair: Tuple[str, str] | None) -> Tuple[str, str]:
        while True:
            spectral_name = (
                self.config.fixed_spectral_name
                if self.config.fixed_spectral_name is not None
                else str(self.rng.choice(self.SPECTRAL_FAMILIES))
            )
            coupling_name = (
                self.config.fixed_coupling_name
                if self.config.fixed_coupling_name is not None
                else str(self.rng.choice(self.COUPLING_LEVELS))
            )

            pair = (spectral_name, coupling_name)
            if prev_pair is None or pair != prev_pair:
                return pair

    def _generate_load_curve(self, T: int, base_load: float) -> np.ndarray:
        cfg = self.config
        drift = np.cumsum(self.rng.normal(0.0, 0.003, size=T))
        slow_wave = 0.05 * np.sin(np.linspace(0, 3 * np.pi, T))
        load = base_load + drift + slow_wave
        return np.clip(load, cfg.load_min, cfg.load_max)

    # ------------------------------------------------------------------
    # Dynamics
    # ------------------------------------------------------------------

    def _build_transition_aware_matrix(
        self,
        t: int,
        boundaries: List[int],
        spectral_name: str,
        coupling_name: str,
        load: float,
        spectral_id_per_t: np.ndarray,
        coupling_id_per_t: np.ndarray,
    ) -> np.ndarray:
        cfg = self.config
        curr = self._build_dynamics_matrix(spectral_name, coupling_name, load)

        if cfg.transition_blend <= 0 or len(boundaries) == 0:
            return curr

        nearest = min(boundaries, key=lambda b: abs(b - t))
        dt = abs(nearest - t)
        if dt > cfg.transition_blend:
            return curr

        left_t = max(0, nearest - 1)
        right_t = min(len(spectral_id_per_t) - 1, nearest)

        left_matrix = self._build_dynamics_matrix(
            self.SPECTRAL_FAMILIES[int(spectral_id_per_t[left_t])],
            self.COUPLING_LEVELS[int(coupling_id_per_t[left_t])],
            load,
        )
        right_matrix = self._build_dynamics_matrix(
            self.SPECTRAL_FAMILIES[int(spectral_id_per_t[right_t])],
            self.COUPLING_LEVELS[int(coupling_id_per_t[right_t])],
            load,
        )

        alpha = dt / max(1, cfg.transition_blend)
        return alpha * curr + (1.0 - alpha) * 0.5 * (left_matrix + right_matrix)

    def _build_dynamics_matrix(self, spectral_name: str, coupling_name: str, load: float) -> np.ndarray:
        H = self.config.latent_dim
        assert H % 2 == 0

        blocks = []
        if spectral_name == "clean_oscillatory":
            params = [(0.985, 0.30), (0.980, 0.18), (0.975, 0.10)]
        elif spectral_name == "damped_oscillatory":
            params = [(0.950, 0.26), (0.930, 0.16), (0.910, 0.08)]
        elif spectral_name == "multi_component":
            params = [(0.980, 0.38), (0.970, 0.21), (0.955, 0.12)]
        elif spectral_name == "quasi_aperiodic":
            params = [(0.970, 0.08), (0.950, -0.05), (0.930, 0.03)]
        else:
            raise ValueError(f"Unknown spectral family: {spectral_name}")

        n_blocks = H // 2
        params = params[:n_blocks]

        load_delta = load - 1.0

        for radius, theta in params:
            radius_adj = np.clip(radius - 0.06 * abs(load_delta), 0.82, 0.995)
            theta_adj = theta * (1.0 + 0.20 * load_delta)
            block = radius_adj * np.array(
                [
                    [np.cos(theta_adj), -np.sin(theta_adj)],
                    [np.sin(theta_adj),  np.cos(theta_adj)],
                ],
                dtype=np.float32,
            )
            blocks.append(block)

        A = np.zeros((H, H), dtype=np.float32)
        for i, block in enumerate(blocks):
            A[2 * i: 2 * i + 2, 2 * i: 2 * i + 2] = block

        coupling_strength = {
            "low": 0.01,
            "medium": 0.05,
            "high": 0.10,
        }[coupling_name]

        mix = np.zeros((H, H), dtype=np.float32)
        for i in range(H):
            for j in range(H):
                if i != j:
                    mix[i, j] = 1.0 / (H - 1)

        A += coupling_strength * mix

        eigvals = np.linalg.eigvals(A)
        spectral_radius = np.max(np.abs(eigvals))
        if spectral_radius >= 0.995:
            A *= (0.995 / float(spectral_radius))

        return A.astype(np.float32)

    # ------------------------------------------------------------------
    # Export helpers
    # ------------------------------------------------------------------

    def _trajectory_to_rows(self, traj: Dict[str, object]) -> List[Dict[str, object]]:
        observed = traj["observed"]
        latent = traj["latent"]
        mode_id = traj["mode_id_per_t"]
        spectral_id = traj["spectral_id_per_t"]
        coupling_id = traj["coupling_id_per_t"]
        load = traj["load_per_t"]
        segment_id = traj["segment_id_per_t"]
        transition_mask = traj["transition_mask"]

        rows = []
        for t in range(observed.shape[0]):
            row = {
                "trajectory_id": int(traj["trajectory_id"]),
                "device_id": int(traj["device_id"]),
                "t": int(t),
                "mode_id": int(mode_id[t]),
                "spectral_id": int(spectral_id[t]),
                "coupling_id": int(coupling_id[t]),
                "load": float(load[t]),
                "segment_id": int(segment_id[t]),
                "is_transition_timestep": int(transition_mask[t]),
            }
            for c in range(observed.shape[1]):
                row[f"y_{c}"] = float(observed[t, c])
            for h in range(latent.shape[1]):
                row[f"x_{h}"] = float(latent[t, h])
            rows.append(row)
        return rows

    def _build_dataset_metadata(
        self,
        trajectories: List[Dict[str, object]],
        windows: Dict[str, np.ndarray]
    ) -> Dict[str, object]:
        transition_fraction = float(np.mean(windows["is_transition_window"].astype(np.float32)))
        unique_modes = sorted(np.unique(windows["mode_id"]).astype(int).tolist())
        unique_spectral = sorted(np.unique(windows["spectral_id"]).astype(int).tolist())
        unique_coupling = sorted(np.unique(windows["coupling_id"]).astype(int).tolist())

        return {
            "config": asdict(self.config),
            "variant": self.config.variant,
            "spectral_families": self.SPECTRAL_FAMILIES,
            "coupling_levels": self.COUPLING_LEVELS,
            "num_trajectories": len(trajectories),
            "num_windows": int(windows["X"].shape[0]),
            "window_shape": list(windows["X"].shape[1:]),
            "transition_window_fraction": transition_fraction,
            "unique_mode_ids": unique_modes,
            "unique_spectral_ids": unique_spectral,
            "unique_coupling_ids": unique_coupling,
            "num_devices": int(self.config.num_devices if self.config.device_effects_enabled else 1),
        }

    @staticmethod
    def _majority_label(arr: np.ndarray) -> int:
        values, counts = np.unique(arr, return_counts=True)
        return int(values[np.argmax(counts)])


# -----------------------------------------------------------------------------
# Saving
# -----------------------------------------------------------------------------


def save_dataset(output_dir: Path, dataset: Dict[str, object]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    df: pd.DataFrame = dataset["trajectory_df"]
    windows: Dict[str, np.ndarray] = dataset["windows"]
    metadata: Dict[str, object] = dataset["metadata"]

    df.to_csv(output_dir / "trajectories.csv", index=False)
    np.savez_compressed(output_dir / "windows.npz", **windows)

    with open(output_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved trajectory table -> {output_dir / 'trajectories.csv'}")
    print(f"Saved windows         -> {output_dir / 'windows.npz'}")
    print(f"Saved metadata        -> {output_dir / 'metadata.json'}")
    print("\nDataset summary:")
    print(json.dumps(metadata, indent=2))


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate factorized switched state-space benchmark with curriculum variants.")
    parser.add_argument("--output-dir", type=str, default="data/fsss/")
    parser.add_argument("--variant", type=str, default="full",
                        choices=["easy_clean", "easy_clean_with_noise", "factorized_clean", "factorized_noisy", "full"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-trajectories", type=int, default=120)
    parser.add_argument("--trajectory-length", type=int, default=320)
    parser.add_argument("--latent-dim", type=int, default=6)
    parser.add_argument("--num-channels", type=int, default=4)
    parser.add_argument("--window-length", type=int, default=48)
    parser.add_argument("--stride", type=int, default=12)

    # Optional advanced override
    parser.add_argument("--keep-transition-windows", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    cfg = GeneratorConfig(
        seed=args.seed,
        variant=args.variant,
        num_trajectories=args.num_trajectories,
        trajectory_length=args.trajectory_length,
        latent_dim=args.latent_dim,
        num_channels=args.num_channels,
        window_length=args.window_length,
        stride=args.stride,
    )

    if args.keep_transition_windows:
        cfg.keep_transition_windows = True

    generator = FactorizedSwitchingGenerator(cfg)
    dataset = generator.generate_dataset()
    save_dataset(Path(args.output_dir), dataset)


if __name__ == "__main__":
    main()