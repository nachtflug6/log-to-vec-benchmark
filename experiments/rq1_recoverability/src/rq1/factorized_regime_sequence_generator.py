"""Formal FRS generator for RQ1 recoverability experiments."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Literal

import numpy as np
import pandas as pd


FRSProfile = Literal[
    "frs_clean",
    "frs_noisy",
    "frs_clean_v3",
    "frs_noisy_v3",
    "frs_clean_vnext",
    "frs_noisy_vnext",
]


@dataclass
class FRSConfig:
    profile: FRSProfile = "frs_clean"
    seed: int = 42

    num_trajectories: int = 160
    trajectory_length: int = 384
    latent_dim: int = 8
    num_channels: int = 6

    min_segments: int = 4
    max_segments: int = 7
    min_dwell: int = 48
    max_dwell: int = 120

    window_length: int = 48
    stride: int = 12
    transition_margin: int = 6
    transition_blend: int = 1

    process_noise_std: float = 0.008
    observation_noise_std: float = 0.008

    load_min: float = 0.95
    load_max: float = 1.05
    load_drift_std: float = 0.0015
    load_wave_amplitude: float = 0.025

    num_devices: int = 2
    device_gain_jitter: float = 0.015
    device_bias_jitter: float = 0.015
    cross_channel_mix: float = 0.006

    channel_dropout_prob: float = 0.0
    offset_shift_prob: float = 0.0
    offset_shift_std: float = 0.0

    load_amplitude_gain: float = 0.45
    load_frequency_gain: float = 0.35
    load_noise_gain: float = 0.30
    coupling_observation_gain: float = 0.18
    coupling_shared_drive_gain: float = 0.12
    load_offset_gain: float = 0.02
    coupling_lag_gain: float = 0.0
    footprint_version: int = 2
    load_envelope_gain: float = 0.0
    spectral_projection_mix: float = 0.75

    def __post_init__(self) -> None:
        if self.latent_dim % 2 != 0:
            raise ValueError("latent_dim must be even.")
        self._apply_profile_defaults()

    def _apply_profile_defaults(self) -> None:
        if self.profile == "frs_clean":
            self.min_segments = 4
            self.max_segments = 7
            self.min_dwell = 48
            self.max_dwell = 120
            self.transition_blend = 1

            self.process_noise_std = 0.008
            self.observation_noise_std = 0.008

            self.load_min = 0.95
            self.load_max = 1.05
            self.load_drift_std = 0.0015
            self.load_wave_amplitude = 0.025

            self.num_devices = 2
            self.device_gain_jitter = 0.015
            self.device_bias_jitter = 0.015
            self.cross_channel_mix = 0.006

            self.channel_dropout_prob = 0.0
            self.offset_shift_prob = 0.0
            self.offset_shift_std = 0.0
            self.load_amplitude_gain = 0.45
            self.load_frequency_gain = 0.35
            self.load_noise_gain = 0.30
            self.coupling_observation_gain = 0.18
            self.coupling_shared_drive_gain = 0.12
            self.load_offset_gain = 0.02
            self.coupling_lag_gain = 0.0
            self.footprint_version = 2
            self.load_envelope_gain = 0.0
            self.spectral_projection_mix = 0.75
            return

        if self.profile == "frs_noisy":
            self.min_segments = 4
            self.max_segments = 8
            self.min_dwell = 40
            self.max_dwell = 104
            self.transition_blend = 4

            self.process_noise_std = 0.022
            self.observation_noise_std = 0.02

            self.load_min = 0.82
            self.load_max = 1.18
            self.load_drift_std = 0.003
            self.load_wave_amplitude = 0.05

            self.num_devices = 6
            self.device_gain_jitter = 0.05
            self.device_bias_jitter = 0.04
            self.cross_channel_mix = 0.015

            self.channel_dropout_prob = 0.0075
            self.offset_shift_prob = 0.05
            self.offset_shift_std = 0.02
            self.load_amplitude_gain = 0.60
            self.load_frequency_gain = 0.50
            self.load_noise_gain = 0.45
            self.coupling_observation_gain = 0.28
            self.coupling_shared_drive_gain = 0.20
            self.load_offset_gain = 0.025
            self.coupling_lag_gain = 0.0
            self.footprint_version = 2
            self.load_envelope_gain = 0.0
            self.spectral_projection_mix = 0.75
            return

        if self.profile == "frs_clean_v3":
            self.min_segments = 4
            self.max_segments = 7
            self.min_dwell = 48
            self.max_dwell = 120
            self.transition_blend = 2

            self.process_noise_std = 0.010
            self.observation_noise_std = 0.010

            self.load_min = 0.90
            self.load_max = 1.10
            self.load_drift_std = 0.0018
            self.load_wave_amplitude = 0.030

            self.num_devices = 2
            self.device_gain_jitter = 0.015
            self.device_bias_jitter = 0.015
            self.cross_channel_mix = 0.005

            self.channel_dropout_prob = 0.0
            self.offset_shift_prob = 0.0
            self.offset_shift_std = 0.0

            self.load_amplitude_gain = 0.55
            self.load_frequency_gain = 0.12
            self.load_noise_gain = 0.22
            self.load_offset_gain = 0.05
            self.coupling_observation_gain = 0.16
            self.coupling_shared_drive_gain = 0.10
            self.coupling_lag_gain = 0.22
            self.footprint_version = 3
            self.load_envelope_gain = 0.0
            self.spectral_projection_mix = 0.75
            return

        if self.profile == "frs_noisy_v3":
            self.min_segments = 4
            self.max_segments = 8
            self.min_dwell = 40
            self.max_dwell = 104
            self.transition_blend = 4

            self.process_noise_std = 0.020
            self.observation_noise_std = 0.018

            self.load_min = 0.80
            self.load_max = 1.20
            self.load_drift_std = 0.003
            self.load_wave_amplitude = 0.050

            self.num_devices = 5
            self.device_gain_jitter = 0.035
            self.device_bias_jitter = 0.030
            self.cross_channel_mix = 0.010

            self.channel_dropout_prob = 0.005
            self.offset_shift_prob = 0.035
            self.offset_shift_std = 0.015

            self.load_amplitude_gain = 0.65
            self.load_frequency_gain = 0.16
            self.load_noise_gain = 0.32
            self.load_offset_gain = 0.07
            self.coupling_observation_gain = 0.20
            self.coupling_shared_drive_gain = 0.14
            self.coupling_lag_gain = 0.28
            self.footprint_version = 3
            self.load_envelope_gain = 0.0
            self.spectral_projection_mix = 0.75
            return

        if self.profile == "frs_clean_vnext":
            self.min_segments = 4
            self.max_segments = 7
            self.min_dwell = 48
            self.max_dwell = 120
            self.transition_blend = 2

            self.process_noise_std = 0.010
            self.observation_noise_std = 0.010

            self.load_min = 0.90
            self.load_max = 1.10
            self.load_drift_std = 0.0015
            self.load_wave_amplitude = 0.025

            self.num_devices = 2
            self.device_gain_jitter = 0.015
            self.device_bias_jitter = 0.012
            self.cross_channel_mix = 0.004

            self.channel_dropout_prob = 0.0
            self.offset_shift_prob = 0.0
            self.offset_shift_std = 0.0

            self.load_amplitude_gain = 0.20
            self.load_frequency_gain = 0.06
            self.load_noise_gain = 0.18
            self.load_offset_gain = 0.0
            self.load_envelope_gain = 0.30
            self.coupling_observation_gain = 0.14
            self.coupling_shared_drive_gain = 0.08
            self.coupling_lag_gain = 0.18
            self.spectral_projection_mix = 0.88
            self.footprint_version = 4
            return

        if self.profile == "frs_noisy_vnext":
            self.min_segments = 4
            self.max_segments = 8
            self.min_dwell = 40
            self.max_dwell = 104
            self.transition_blend = 4

            self.process_noise_std = 0.018
            self.observation_noise_std = 0.016

            self.load_min = 0.82
            self.load_max = 1.18
            self.load_drift_std = 0.0028
            self.load_wave_amplitude = 0.040

            self.num_devices = 5
            self.device_gain_jitter = 0.030
            self.device_bias_jitter = 0.022
            self.cross_channel_mix = 0.008

            self.channel_dropout_prob = 0.004
            self.offset_shift_prob = 0.025
            self.offset_shift_std = 0.012

            self.load_amplitude_gain = 0.26
            self.load_frequency_gain = 0.08
            self.load_noise_gain = 0.24
            self.load_offset_gain = 0.0
            self.load_envelope_gain = 0.40
            self.coupling_observation_gain = 0.18
            self.coupling_shared_drive_gain = 0.10
            self.coupling_lag_gain = 0.24
            self.spectral_projection_mix = 0.85
            self.footprint_version = 4
            return

        raise ValueError(f"Unknown FRS profile: {self.profile}")


class FactorizedRegimeSequenceGenerator:
    """Generate a multi-factor formal synthetic benchmark for RQ1."""

    SPECTRAL_FAMILIES = [
        "single_periodic",
        "damped_periodic",
        "multi_periodic",
        "quasi_aperiodic",
    ]
    COUPLING_LEVELS = ["low", "medium", "high"]

    def __init__(self, config: FRSConfig):
        self.config = config
        self.rng = np.random.default_rng(config.seed)
        self._spectral_ids = {name: i for i, name in enumerate(self.SPECTRAL_FAMILIES)}
        self._coupling_ids = {name: i for i, name in enumerate(self.COUPLING_LEVELS)}
        self.device_params = self._build_devices()

    def generate_dataset(self) -> Dict[str, object]:
        trajectories = []
        step_rows = []
        for trajectory_id in range(self.config.num_trajectories):
            trajectory = self._generate_trajectory(trajectory_id)
            trajectories.append(trajectory)
            step_rows.extend(self._trajectory_to_rows(trajectory))

        windows = self.extract_windows(trajectories)
        return {
            "trajectories": trajectories,
            "trajectory_df": pd.DataFrame(step_rows),
            "windows": windows,
            "metadata": self._build_metadata(trajectories, windows),
        }

    def extract_windows(self, trajectories: List[Dict[str, object]]) -> Dict[str, np.ndarray]:
        cfg = self.config
        X_list: list[np.ndarray] = []
        trajectory_ids = []
        device_ids = []
        window_starts = []
        regime_ids = []
        spectral_ids = []
        coupling_ids = []
        mean_loads = []
        transition_flags = []
        distance_to_boundary = []
        left_regimes = []
        right_regimes = []

        for trajectory in trajectories:
            observed = trajectory["observed"]
            regime_id = trajectory["regime_id_per_t"]
            spectral_id = trajectory["spectral_family_id_per_t"]
            coupling_id = trajectory["coupling_level_id_per_t"]
            load = trajectory["load_per_t"]
            boundaries = np.array(trajectory["boundary_indices"], dtype=np.int64)

            total_steps = observed.shape[0]
            for start in range(0, total_steps - cfg.window_length + 1, cfg.stride):
                end = start + cfg.window_length
                inside = boundaries[(boundaries >= start) & (boundaries < end)]
                is_transition = len(inside) > 0

                if is_transition:
                    boundary = int(inside[0])
                    dist = min(abs(boundary - start), abs(end - 1 - boundary))
                    left_regime = int(regime_id[max(start, boundary - 1)])
                    right_regime = int(regime_id[min(end - 1, boundary)])
                else:
                    dist = cfg.window_length
                    left_regime = -1
                    right_regime = -1

                X_list.append(observed[start:end].astype(np.float32))
                trajectory_ids.append(int(trajectory["trajectory_id"]))
                device_ids.append(int(trajectory["device_id"]))
                window_starts.append(int(start))
                regime_majority = self._majority(regime_id[start:end])
                spectral_majority = self._majority(spectral_id[start:end])
                coupling_majority = self._majority(coupling_id[start:end])
                regime_ids.append(regime_majority)
                spectral_ids.append(spectral_majority)
                coupling_ids.append(coupling_majority)
                mean_loads.append(float(np.mean(load[start:end])))
                transition_flags.append(bool(is_transition))
                distance_to_boundary.append(int(dist))
                left_regimes.append(left_regime)
                right_regimes.append(right_regime)

        regime_ids_np = np.array(regime_ids, dtype=np.int64)
        spectral_ids_np = np.array(spectral_ids, dtype=np.int64)
        coupling_ids_np = np.array(coupling_ids, dtype=np.int64)

        return {
            "X": np.stack(X_list, axis=0),
            "trajectory_id": np.array(trajectory_ids, dtype=np.int64),
            "device_id": np.array(device_ids, dtype=np.int64),
            "window_start": np.array(window_starts, dtype=np.int64),
            "regime_id": regime_ids_np.copy(),
            "spectral_family_id": spectral_ids_np.copy(),
            "coupling_level_id": coupling_ids_np.copy(),
            "mode_id": regime_ids_np.copy(),
            "spectral_id": spectral_ids_np.copy(),
            "coupling_id": coupling_ids_np.copy(),
            "mean_load": np.array(mean_loads, dtype=np.float32),
            "is_transition_window": np.array(transition_flags, dtype=np.bool_),
            "distance_to_boundary": np.array(distance_to_boundary, dtype=np.int64),
            "left_regime_id": np.array(left_regimes, dtype=np.int64),
            "right_regime_id": np.array(right_regimes, dtype=np.int64),
            "left_mode_id": np.array(left_regimes, dtype=np.int64),
            "right_mode_id": np.array(right_regimes, dtype=np.int64),
        }

    def _build_devices(self) -> List[Dict[str, np.ndarray]]:
        devices = []
        cfg = self.config
        for device_id in range(cfg.num_devices):
            diag = 1.0 + self.rng.uniform(-cfg.device_gain_jitter, cfg.device_gain_jitter, size=cfg.num_channels)
            C = np.eye(cfg.num_channels, cfg.latent_dim, dtype=np.float32)
            C[: cfg.num_channels, : cfg.num_channels] *= diag[:, None]

            for row in range(cfg.num_channels):
                for col in range(min(cfg.num_channels, cfg.latent_dim)):
                    if row != col:
                        C[row, col] += self.rng.uniform(-cfg.cross_channel_mix, cfg.cross_channel_mix)

            bias = self.rng.normal(0.0, cfg.device_bias_jitter, size=cfg.num_channels).astype(np.float32)
            devices.append({"device_id": device_id, "C": C.astype(np.float32), "bias": bias})
        return devices

    def _generate_trajectory(self, trajectory_id: int) -> Dict[str, object]:
        cfg = self.config
        device = self.device_params[int(self.rng.integers(0, len(self.device_params)))]
        total_steps = cfg.trajectory_length
        latent_dim = cfg.latent_dim

        load_base = float(self.rng.uniform(cfg.load_min, cfg.load_max))
        load_per_t = self._generate_load_curve(total_steps, load_base)

        segments = self._sample_segments(total_steps)
        regime_id_per_t = np.zeros(total_steps, dtype=np.int64)
        spectral_family_id_per_t = np.zeros(total_steps, dtype=np.int64)
        coupling_level_id_per_t = np.zeros(total_steps, dtype=np.int64)
        boundaries = []

        for index, segment in enumerate(segments):
            start, end = segment["start"], segment["end"]
            spectral_name = segment["spectral_family"]
            coupling_name = segment["coupling_level"]
            spectral_family_id = self._spectral_ids[spectral_name]
            coupling_level_id = self._coupling_ids[coupling_name]
            regime_id = spectral_family_id * len(self.COUPLING_LEVELS) + coupling_level_id

            regime_id_per_t[start:end] = regime_id
            spectral_family_id_per_t[start:end] = spectral_family_id
            coupling_level_id_per_t[start:end] = coupling_level_id
            if index > 0:
                boundaries.append(start)

        latent = np.zeros((total_steps, latent_dim), dtype=np.float32)
        observed = np.zeros((total_steps, cfg.num_channels), dtype=np.float32)
        transition_mask = np.zeros(total_steps, dtype=np.int64)
        previous_state = self.rng.normal(0.0, 0.35, size=latent_dim)
        shared_drive_phase = float(self.rng.uniform(0.0, 2.0 * np.pi))
        previous_observation = np.zeros(cfg.num_channels, dtype=np.float32)

        for step in range(total_steps):
            spectral_name = self.SPECTRAL_FAMILIES[int(spectral_family_id_per_t[step])]
            coupling_name = self.COUPLING_LEVELS[int(coupling_level_id_per_t[step])]
            load_t = float(load_per_t[step])
            dynamics = self._build_transition_aware_matrix(
                step=step,
                boundaries=boundaries,
                spectral_family_id_per_t=spectral_family_id_per_t,
                coupling_level_id_per_t=coupling_level_id_per_t,
                current_spectral=spectral_name,
                current_coupling=coupling_name,
                load=load_t,
            )

            load_centered = self._normalize_load(load_t)
            process_noise_scale = cfg.process_noise_std * (1.0 + cfg.load_noise_gain * abs(load_centered))
            process_noise = self.rng.normal(0.0, process_noise_scale, size=latent_dim)
            current_state = dynamics @ previous_state + process_noise

            coupling_mix_strength = self.config.coupling_observation_gain * {
                "low": 0.20,
                "medium": 0.95,
                "high": 1.75,
            }[coupling_name]
            coupling_matrix = self._build_coupling_observation_matrix(coupling_mix_strength)

            load_amplitude_scale = 1.0 + cfg.load_amplitude_gain * load_centered
            observation_noise_scale = cfg.observation_noise_std * (1.0 + cfg.load_noise_gain * abs(load_centered))
            observation_noise = self.rng.normal(0.0, observation_noise_scale, size=cfg.num_channels)

            if cfg.footprint_version >= 4:
                current_obs = self._compose_observation_vnext(
                    current_state=current_state,
                    previous_observation=previous_observation,
                    device=device,
                    spectral_name=spectral_name,
                    coupling_name=coupling_name,
                    coupling_matrix=coupling_matrix,
                    load_centered=load_centered,
                    load_amplitude_scale=load_amplitude_scale,
                    step=step,
                    total_steps=total_steps,
                    shared_drive_phase=shared_drive_phase,
                    observation_noise=observation_noise,
                )
            elif cfg.footprint_version >= 3:
                current_obs = self._compose_observation_v3(
                    current_state=current_state,
                    previous_observation=previous_observation,
                    device=device,
                    spectral_name=spectral_name,
                    coupling_name=coupling_name,
                    coupling_matrix=coupling_matrix,
                    load_centered=load_centered,
                    load_amplitude_scale=load_amplitude_scale,
                    step=step,
                    total_steps=total_steps,
                    shared_drive_phase=shared_drive_phase,
                    observation_noise=observation_noise,
                )
            else:
                base_obs = device["C"] @ current_state
                shared_drive = self._shared_drive_signal(
                    step=step,
                    total_steps=total_steps,
                    coupling_name=coupling_name,
                    phase=shared_drive_phase,
                    load_centered=load_centered,
                )
                shared_drive_vector = shared_drive * np.linspace(0.8, 1.2, cfg.num_channels, dtype=np.float32)

                current_obs = load_amplitude_scale * (coupling_matrix @ base_obs)
                current_obs = current_obs + shared_drive_vector + device["bias"] + observation_noise

            if self.rng.random() < cfg.channel_dropout_prob:
                drop_index = int(self.rng.integers(0, cfg.num_channels))
                current_obs[drop_index] = 0.0
            if self.rng.random() < cfg.offset_shift_prob:
                current_obs += self.rng.normal(0.0, cfg.offset_shift_std, size=cfg.num_channels)

            latent[step] = current_state.astype(np.float32)
            observed[step] = current_obs.astype(np.float32)
            previous_state = current_state
            previous_observation = current_obs.astype(np.float32)

        for boundary in boundaries:
            left = max(0, boundary - cfg.transition_margin)
            right = min(total_steps, boundary + cfg.transition_margin + 1)
            transition_mask[left:right] = 1

        return {
            "trajectory_id": trajectory_id,
            "device_id": int(device["device_id"]),
            "observed": observed,
            "latent": latent,
            "regime_id_per_t": regime_id_per_t,
            "spectral_family_id_per_t": spectral_family_id_per_t,
            "coupling_level_id_per_t": coupling_level_id_per_t,
            "load_per_t": load_per_t.astype(np.float32),
            "transition_mask": transition_mask,
            "boundary_indices": boundaries,
        }

    def _sample_segments(self, total_steps: int) -> List[Dict[str, object]]:
        cfg = self.config
        segment_count = int(self.rng.integers(cfg.min_segments, cfg.max_segments + 1))
        raw_lengths = self.rng.integers(cfg.min_dwell, cfg.max_dwell + 1, size=segment_count)
        scaled_lengths = np.maximum(cfg.min_dwell, (raw_lengths / raw_lengths.sum() * total_steps).astype(int))
        scaled_lengths[-1] += total_steps - int(scaled_lengths.sum())

        while scaled_lengths[-1] < cfg.min_dwell:
            donor = int(np.argmax(scaled_lengths[:-1]))
            if scaled_lengths[donor] <= cfg.min_dwell:
                break
            scaled_lengths[donor] -= 1
            scaled_lengths[-1] += 1

        segments = []
        cursor = 0
        previous_pair = None
        for dwell in scaled_lengths:
            pair = self._sample_regime(previous_pair)
            start = cursor
            end = min(total_steps, cursor + int(dwell))
            segments.append({"start": start, "end": end, "spectral_family": pair[0], "coupling_level": pair[1]})
            cursor = end
            previous_pair = pair
        segments[-1]["end"] = total_steps
        return segments

    def _sample_regime(self, previous_pair: tuple[str, str] | None) -> tuple[str, str]:
        while True:
            pair = (
                str(self.rng.choice(self.SPECTRAL_FAMILIES)),
                str(self.rng.choice(self.COUPLING_LEVELS)),
            )
            if previous_pair is None or pair != previous_pair:
                return pair

    def _generate_load_curve(self, total_steps: int, base_load: float) -> np.ndarray:
        cfg = self.config
        drift = np.cumsum(self.rng.normal(0.0, cfg.load_drift_std, size=total_steps))
        wave = cfg.load_wave_amplitude * np.sin(np.linspace(0.0, 3.0 * np.pi, total_steps))
        load = base_load + drift + wave
        return np.clip(load, cfg.load_min, cfg.load_max)

    def _normalize_load(self, load: float) -> float:
        midpoint = 0.5 * (self.config.load_min + self.config.load_max)
        half_range = max(1e-6, 0.5 * (self.config.load_max - self.config.load_min))
        return float((load - midpoint) / half_range)

    def _build_coupling_observation_matrix(self, mix_strength: float) -> np.ndarray:
        num_channels = self.config.num_channels
        identity = np.eye(num_channels, dtype=np.float32)
        shared = np.full((num_channels, num_channels), 1.0 / num_channels, dtype=np.float32)
        return (1.0 - mix_strength) * identity + mix_strength * shared

    def _spectral_projection_matrix(self, spectral_name: str) -> np.ndarray:
        num_channels = self.config.num_channels
        latent_dim = self.config.latent_dim
        matrix = np.zeros((num_channels, latent_dim), dtype=np.float32)

        if spectral_name == "single_periodic":
            templates = [
                [1.20, 0.25, 0.10, 0.00],
                [1.10, 0.35, 0.05, 0.00],
                [0.95, 0.15, 0.05, 0.00],
                [1.05, 0.20, 0.10, 0.00],
                [0.90, 0.25, 0.10, 0.00],
                [1.00, 0.30, 0.05, 0.00],
            ]
        elif spectral_name == "damped_periodic":
            templates = [
                [0.25, 1.10, 0.20, 0.05],
                [0.20, 1.20, 0.15, 0.05],
                [0.15, 1.00, 0.25, 0.05],
                [0.25, 0.90, 0.20, 0.10],
                [0.20, 1.05, 0.15, 0.10],
                [0.15, 1.15, 0.20, 0.05],
            ]
        elif spectral_name == "multi_periodic":
            templates = [
                [0.60, 0.55, 0.90, 0.10],
                [0.55, 0.50, 1.00, 0.10],
                [0.45, 0.55, 0.95, 0.15],
                [0.50, 0.45, 1.05, 0.10],
                [0.55, 0.40, 0.85, 0.20],
                [0.50, 0.50, 0.90, 0.15],
            ]
        elif spectral_name == "quasi_aperiodic":
            templates = [
                [0.25, 0.15, 0.20, 1.10],
                [0.20, 0.10, 0.15, 1.20],
                [0.15, 0.20, 0.10, 1.00],
                [0.25, 0.10, 0.15, 0.95],
                [0.15, 0.15, 0.20, 1.05],
                [0.20, 0.20, 0.10, 1.15],
            ]
        else:
            raise ValueError(f"Unknown spectral family: {spectral_name}")

        usable = min(num_channels, len(templates))
        num_blocks = latent_dim // 2
        for channel in range(num_channels):
            coeffs = templates[channel % usable]
            for block in range(num_blocks):
                weight = coeffs[min(block, len(coeffs) - 1)]
                matrix[channel, 2 * block : 2 * block + 2] = weight
        return matrix

    def _compose_observation_v3(
        self,
        current_state: np.ndarray,
        previous_observation: np.ndarray,
        device: Dict[str, np.ndarray],
        spectral_name: str,
        coupling_name: str,
        coupling_matrix: np.ndarray,
        load_centered: float,
        load_amplitude_scale: float,
        step: int,
        total_steps: int,
        shared_drive_phase: float,
        observation_noise: np.ndarray,
    ) -> np.ndarray:
        spectral_projection = self._spectral_projection_matrix(spectral_name)
        device_projection = device["C"][: self.config.num_channels, : self.config.latent_dim]
        base_projection = 0.75 * spectral_projection + 0.25 * device_projection
        base_obs = base_projection @ current_state

        coupled_obs = coupling_matrix @ base_obs

        lag_strength = self.config.coupling_lag_gain * {
            "low": 0.10,
            "medium": 0.55,
            "high": 1.00,
        }[coupling_name]
        lagged_neighbor = np.roll(previous_observation, 1)
        coupled_obs = (1.0 - lag_strength) * coupled_obs + lag_strength * lagged_neighbor

        shared_drive = self._shared_drive_signal(
            step=step,
            total_steps=total_steps,
            coupling_name=coupling_name,
            phase=shared_drive_phase,
            load_centered=load_centered,
        )
        channel_axis = np.linspace(-1.0, 1.0, self.config.num_channels, dtype=np.float32)
        load_offset = self.config.load_offset_gain * load_centered * channel_axis
        shared_drive_vector = shared_drive * (1.0 + 0.15 * channel_axis)

        current_obs = load_amplitude_scale * coupled_obs
        current_obs = current_obs + load_offset + shared_drive_vector + device["bias"] + observation_noise
        return current_obs.astype(np.float32)

    def _compose_observation_vnext(
        self,
        current_state: np.ndarray,
        previous_observation: np.ndarray,
        device: Dict[str, np.ndarray],
        spectral_name: str,
        coupling_name: str,
        coupling_matrix: np.ndarray,
        load_centered: float,
        load_amplitude_scale: float,
        step: int,
        total_steps: int,
        shared_drive_phase: float,
        observation_noise: np.ndarray,
    ) -> np.ndarray:
        spectral_projection = self._spectral_projection_matrix_vnext(spectral_name)
        device_projection = device["C"][: self.config.num_channels, : self.config.latent_dim]
        mix = self.config.spectral_projection_mix
        base_projection = mix * spectral_projection + (1.0 - mix) * device_projection
        base_obs = base_projection @ current_state

        coupled_obs = coupling_matrix @ base_obs
        lag_strength = self.config.coupling_lag_gain * {
            "low": 0.10,
            "medium": 0.55,
            "high": 1.00,
        }[coupling_name]
        lagged_neighbor = np.roll(previous_observation, 1)
        coupled_obs = (1.0 - lag_strength) * coupled_obs + lag_strength * lagged_neighbor

        shared_drive = self._shared_drive_signal(
            step=step,
            total_steps=total_steps,
            coupling_name=coupling_name,
            phase=shared_drive_phase,
            load_centered=0.0,
        )
        channel_axis = np.linspace(-1.0, 1.0, self.config.num_channels, dtype=np.float32)
        envelope = 1.0 + self.config.load_envelope_gain * load_centered * np.sin(
            2.0 * np.pi * step / max(12, total_steps / 5)
        )
        shared_drive_vector = shared_drive * (1.0 + 0.10 * channel_axis)

        current_obs = envelope * load_amplitude_scale * coupled_obs
        current_obs = current_obs + shared_drive_vector + device["bias"] + observation_noise
        return current_obs.astype(np.float32)

    def _shared_drive_signal(
        self,
        step: int,
        total_steps: int,
        coupling_name: str,
        phase: float,
        load_centered: float,
    ) -> float:
        coupling_scale = {"low": 0.15, "medium": 0.55, "high": 1.0}[coupling_name]
        base = np.sin(phase + 2.0 * np.pi * step / max(8, total_steps / 6))
        load_scale = 1.0 + 0.5 * load_centered
        return float(self.config.coupling_shared_drive_gain * coupling_scale * load_scale * base)

    def _build_transition_aware_matrix(
        self,
        step: int,
        boundaries: List[int],
        spectral_family_id_per_t: np.ndarray,
        coupling_level_id_per_t: np.ndarray,
        current_spectral: str,
        current_coupling: str,
        load: float,
    ) -> np.ndarray:
        current = self._build_dynamics_matrix(current_spectral, current_coupling, load)
        if self.config.transition_blend <= 0 or not boundaries:
            return current

        nearest = min(boundaries, key=lambda b: abs(b - step))
        dt = abs(nearest - step)
        if dt > self.config.transition_blend:
            return current

        left_index = max(0, nearest - 1)
        right_index = min(len(spectral_family_id_per_t) - 1, nearest)
        left_matrix = self._build_dynamics_matrix(
            self.SPECTRAL_FAMILIES[int(spectral_family_id_per_t[left_index])],
            self.COUPLING_LEVELS[int(coupling_level_id_per_t[left_index])],
            load,
        )
        right_matrix = self._build_dynamics_matrix(
            self.SPECTRAL_FAMILIES[int(spectral_family_id_per_t[right_index])],
            self.COUPLING_LEVELS[int(coupling_level_id_per_t[right_index])],
            load,
        )
        alpha = dt / max(1, self.config.transition_blend)
        return alpha * current + (1.0 - alpha) * 0.5 * (left_matrix + right_matrix)

    def _build_dynamics_matrix(self, spectral_family: str, coupling_level: str, load: float) -> np.ndarray:
        latent_dim = self.config.latent_dim
        if self.config.footprint_version >= 4:
            params_by_family = {
                "single_periodic": [(0.989, 0.24), (0.984, 0.12), (0.980, 0.05), (0.976, 0.02)],
                "damped_periodic": [(0.944, 0.19), (0.928, 0.10), (0.912, 0.04), (0.896, 0.01)],
                "multi_periodic": [(0.987, 0.34), (0.980, 0.22), (0.972, 0.12), (0.966, 0.06)],
                "quasi_aperiodic": [(0.969, 0.05), (0.956, -0.08), (0.944, 0.02), (0.931, -0.04)],
            }
        else:
            params_by_family = {
                "single_periodic": [(0.988, 0.28), (0.982, 0.15), (0.978, 0.08), (0.972, 0.04)],
                "damped_periodic": [(0.952, 0.24), (0.938, 0.14), (0.924, 0.07), (0.910, 0.03)],
                "multi_periodic": [(0.985, 0.36), (0.978, 0.23), (0.968, 0.14), (0.958, 0.06)],
                "quasi_aperiodic": [(0.972, 0.07), (0.955, -0.05), (0.944, 0.03), (0.930, -0.02)],
            }
        params = params_by_family[spectral_family][: latent_dim // 2]
        load_delta = load - 1.0
        blocks = []
        for radius, theta in params:
            radius_adj = np.clip(radius - 0.05 * abs(load_delta), 0.84, 0.995)
            theta_adj = theta * (1.0 + self.config.load_frequency_gain * load_delta)
            blocks.append(
                radius_adj
                * np.array(
                    [
                        [np.cos(theta_adj), -np.sin(theta_adj)],
                        [np.sin(theta_adj), np.cos(theta_adj)],
                    ],
                    dtype=np.float32,
                )
            )

        A = np.zeros((latent_dim, latent_dim), dtype=np.float32)
        for index, block in enumerate(blocks):
            A[2 * index : 2 * index + 2, 2 * index : 2 * index + 2] = block

        coupling_strength = {"low": 0.005, "medium": 0.055, "high": 0.13}[coupling_level]
        mix = np.zeros((latent_dim, latent_dim), dtype=np.float32)
        for row in range(latent_dim):
            for col in range(latent_dim):
                if row != col:
                    mix[row, col] = 1.0 / (latent_dim - 1)
        A += coupling_strength * mix

        eigvals = np.linalg.eigvals(A)
        spectral_radius = np.max(np.abs(eigvals))
        if spectral_radius >= 0.995:
            A *= 0.995 / float(spectral_radius)
        return A.astype(np.float32)

    def _spectral_projection_matrix_vnext(self, spectral_name: str) -> np.ndarray:
        num_channels = self.config.num_channels
        latent_dim = self.config.latent_dim
        matrix = np.zeros((num_channels, latent_dim), dtype=np.float32)
        templates = {
            "single_periodic": [
                [1.35, 0.18, 0.02, 0.00],
                [1.18, 0.16, 0.04, 0.00],
                [1.05, 0.12, 0.02, 0.00],
                [1.12, 0.18, 0.04, 0.00],
                [0.98, 0.14, 0.03, 0.00],
                [1.10, 0.16, 0.02, 0.00],
            ],
            "damped_periodic": [
                [0.10, 1.32, 0.08, 0.00],
                [0.12, 1.18, 0.12, 0.00],
                [0.08, 1.10, 0.10, 0.00],
                [0.14, 1.24, 0.06, 0.00],
                [0.10, 1.08, 0.10, 0.00],
                [0.12, 1.20, 0.08, 0.00],
            ],
            "multi_periodic": [
                [0.55, 0.42, 1.08, 0.06],
                [0.48, 0.36, 1.18, 0.08],
                [0.44, 0.40, 1.10, 0.10],
                [0.50, 0.32, 1.22, 0.05],
                [0.42, 0.38, 1.04, 0.12],
                [0.46, 0.34, 1.14, 0.08],
            ],
            "quasi_aperiodic": [
                [0.06, 0.10, 0.12, 1.30],
                [0.08, 0.06, 0.10, 1.16],
                [0.05, 0.12, 0.08, 1.08],
                [0.10, 0.08, 0.06, 1.24],
                [0.08, 0.10, 0.10, 1.10],
                [0.06, 0.08, 0.12, 1.18],
            ],
        }[spectral_name]

        usable = min(num_channels, len(templates))
        num_blocks = latent_dim // 2
        for channel in range(num_channels):
            coeffs = templates[channel % usable]
            for block in range(num_blocks):
                matrix[channel, 2 * block : 2 * block + 2] = coeffs[min(block, len(coeffs) - 1)]
        return matrix

    def _trajectory_to_rows(self, trajectory: Dict[str, object]) -> List[Dict[str, object]]:
        rows = []
        observed = trajectory["observed"]
        latent = trajectory["latent"]
        regime_id = trajectory["regime_id_per_t"]
        spectral_id = trajectory["spectral_family_id_per_t"]
        coupling_id = trajectory["coupling_level_id_per_t"]
        load = trajectory["load_per_t"]
        transition_mask = trajectory["transition_mask"]

        for step in range(observed.shape[0]):
            row = {
                "trajectory_id": int(trajectory["trajectory_id"]),
                "device_id": int(trajectory["device_id"]),
                "t": int(step),
                "regime_id": int(regime_id[step]),
                "spectral_family_id": int(spectral_id[step]),
                "coupling_level_id": int(coupling_id[step]),
                "mode_id": int(regime_id[step]),
                "spectral_id": int(spectral_id[step]),
                "coupling_id": int(coupling_id[step]),
                "load": float(load[step]),
                "is_transition_timestep": int(transition_mask[step]),
            }
            for channel in range(observed.shape[1]):
                row[f"y_{channel}"] = float(observed[step, channel])
            for latent_index in range(latent.shape[1]):
                row[f"x_{latent_index}"] = float(latent[step, latent_index])
            rows.append(row)
        return rows

    def _build_metadata(self, trajectories: List[Dict[str, object]], windows: Dict[str, np.ndarray]) -> Dict[str, object]:
        return {
            "dataset_name": self.config.profile,
            "family": "frs",
            "family_name": "Factorized Regime Sequence",
            "config": asdict(self.config),
            "num_trajectories": len(trajectories),
            "num_windows": int(windows["X"].shape[0]),
            "window_shape": list(windows["X"].shape[1:]),
            "transition_window_fraction": float(np.mean(windows["is_transition_window"].astype(np.float32))),
            "unique_regime_ids": sorted(np.unique(windows["regime_id"]).astype(int).tolist()),
            "unique_spectral_family_ids": sorted(np.unique(windows["spectral_family_id"]).astype(int).tolist()),
            "unique_coupling_level_ids": sorted(np.unique(windows["coupling_level_id"]).astype(int).tolist()),
            "spectral_families": self.SPECTRAL_FAMILIES,
            "coupling_levels": self.COUPLING_LEVELS,
            "primary_targets": [
                "mode_id",
                "spectral_id",
                "coupling_id",
                "is_transition_window",
                "mean_load",
            ],
            "notes": [
                "Formal RQ1 benchmark family with multi-factor latent structure.",
                "mode_id is an alias of regime_id for compatibility with the evaluation pipeline.",
            ],
        }

    @staticmethod
    def _majority(values: np.ndarray) -> int:
        unique_values, counts = np.unique(values, return_counts=True)
        return int(unique_values[np.argmax(counts)])


def save_dataset(output_dir: Path, dataset: Dict[str, object]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset["trajectory_df"].to_csv(output_dir / "trajectories.csv", index=False)
    np.savez_compressed(output_dir / "windows.npz", **dataset["windows"])
    with open(output_dir / "metadata.json", "w", encoding="utf-8") as handle:
        json.dump(dataset["metadata"], handle, indent=2)
