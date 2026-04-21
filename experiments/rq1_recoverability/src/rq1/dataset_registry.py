"""Dataset preparation and registration helpers for RQ1 experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import sys

from .io import read_json, write_json
from .factorized_regime_sequence_generator import (
    FRSConfig,
    FactorizedRegimeSequenceGenerator,
    save_dataset as save_frs_dataset,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "src"))

import numpy as np
import pandas as pd


def _build_moment_freq_dataset(
    artifact_dir: Path,
    seed: int,
    n_samples: int,
    seq_len: int,
    noise_std: float,
    num_bins: int,
) -> None:
    from moment.data.synthetic_data import SyntheticDataset

    dataset = SyntheticDataset(
        n_samples=n_samples,
        seq_len=seq_len,
        random_seed=seed,
        noise_std=noise_std,
    )
    y, c = dataset.gen_sinusoids_with_varying_freq()

    X = y.transpose(1, 2).detach().cpu().numpy().astype(np.float32)
    control_value = c[:, 0].detach().cpu().numpy().astype(np.float32)
    edges = np.linspace(control_value.min(), control_value.max(), num_bins + 1)
    bins = np.digitize(control_value, edges[1:-1], right=False).astype(np.int64)

    n = X.shape[0]
    windows = {
        "X": X,
        "trajectory_id": np.arange(n, dtype=np.int64),
        "device_id": np.zeros(n, dtype=np.int64),
        "window_start": np.zeros(n, dtype=np.int64),
        "mode_id": bins.copy(),
        "spectral_id": bins.copy(),
        "coupling_id": np.zeros(n, dtype=np.int64),
        "mean_load": np.zeros(n, dtype=np.float32),
        "is_transition_window": np.zeros(n, dtype=np.bool_),
        "distance_to_boundary": np.full(n, seq_len, dtype=np.int64),
        "left_mode_id": np.full(n, -1, dtype=np.int64),
        "right_mode_id": np.full(n, -1, dtype=np.int64),
        "control_value": control_value,
    }

    trajectory_df = pd.DataFrame(
        {
            "trajectory_id": np.arange(n, dtype=np.int64),
            "device_id": np.zeros(n, dtype=np.int64),
            "window_start": np.zeros(n, dtype=np.int64),
            "seq_len": np.full(n, seq_len, dtype=np.int64),
            "frequency_value": control_value,
            "mode_id": bins,
            "spectral_id": bins,
        }
    )
    metadata = {
        "dataset_name": "moment_freq",
        "family": "moment",
        "variant": "varying_freq",
        "seed": seed,
        "n_samples": n_samples,
        "seq_len": seq_len,
        "noise_std": noise_std,
        "num_bins": num_bins,
        "primary_targets": ["mode_id", "spectral_id"],
        "notes": [
            "Sanity-check dataset for the RQ1 pipeline.",
            "Only mode_id and spectral_id are meaningful recoverability targets here.",
        ],
    }

    artifact_dir.mkdir(parents=True, exist_ok=True)
    trajectory_df.to_csv(artifact_dir / "trajectories.csv", index=False)
    np.savez_compressed(artifact_dir / "windows.npz", **windows)
    write_json(artifact_dir / "metadata.json", metadata)


def _build_frs_dataset(
    artifact_dir: Path,
    dataset_name: str,
    seed: int,
    variant: str,
    num_trajectories: int,
    trajectory_length: int,
    latent_dim: int,
    num_channels: int,
    window_length: int,
    stride: int,
) -> None:
    cfg = FRSConfig(
        profile=dataset_name,
        seed=seed,
        num_trajectories=num_trajectories,
        trajectory_length=trajectory_length,
        latent_dim=latent_dim,
        num_channels=num_channels,
        window_length=window_length,
        stride=stride,
    )
    generator = FactorizedRegimeSequenceGenerator(cfg)
    dataset = generator.generate_dataset()

    metadata = dict(dataset["metadata"])
    metadata["dataset_name"] = dataset_name
    metadata["family"] = "frs"
    metadata["family_name"] = "Factorized Regime Sequence"
    metadata["variant"] = variant
    metadata["primary_targets"] = [
        "mode_id",
        "spectral_id",
        "coupling_id",
        "is_transition_window",
        "mean_load",
    ]
    metadata["notes"] = [
        "Formal synthetic benchmark family for RQ1 recoverability.",
        "FRS = Factorized Regime Sequence.",
    ]
    dataset["metadata"] = metadata

    save_frs_dataset(artifact_dir, dataset)


def build_dataset(
    dataset_name: str,
    artifact_root: Path,
    seed: int = 42,
    n_samples: int = 1024,
    seq_len: int = 512,
    noise_std: float = 0.1,
    num_bins: int = 8,
    num_trajectories: int = 120,
    trajectory_length: int = 320,
    latent_dim: int = 6,
    num_channels: int = 4,
    window_length: int = 48,
    stride: int = 12,
) -> Path:
    artifact_dir = artifact_root / dataset_name / "raw"
    if dataset_name == "moment_freq":
        _build_moment_freq_dataset(
            artifact_dir=artifact_dir,
            seed=seed,
            n_samples=n_samples,
            seq_len=seq_len,
            noise_std=noise_std,
            num_bins=num_bins,
        )
        return artifact_dir
    if dataset_name in {
        "frs_clean",
        "frs_noisy",
        "frs_clean_v3",
        "frs_noisy_v3",
        "frs_clean_vnext",
        "frs_noisy_vnext",
        "frs_clean_vnext_long",
        "frs_noisy_vnext_long",
    }:
        variant = {
            "frs_clean": "factorized_clean",
            "frs_noisy": "factorized_noisy",
            "frs_clean_v3": "frs_clean_v3",
            "frs_noisy_v3": "frs_noisy_v3",
            "frs_clean_vnext": "frs_clean_vnext",
            "frs_noisy_vnext": "frs_noisy_vnext",
            "frs_clean_vnext_long": "frs_clean_vnext_long",
            "frs_noisy_vnext_long": "frs_noisy_vnext_long",
        }[dataset_name]
        profile_name = {
            "frs_clean_vnext_long": "frs_clean_vnext",
            "frs_noisy_vnext_long": "frs_noisy_vnext",
        }.get(dataset_name, dataset_name)
        effective_window_length = window_length
        effective_stride = stride
        effective_trajectory_length = trajectory_length
        if dataset_name.endswith("_long"):
            effective_window_length = 96 if window_length == 48 else window_length
            effective_stride = 24 if stride == 12 else stride
            # Fair long-context evaluation needs longer stable regimes as well,
            # otherwise most windows become transition-dominated and the task changes.
            effective_trajectory_length = 640 if trajectory_length == 320 else trajectory_length
        _build_frs_dataset(
            artifact_dir=artifact_dir,
            dataset_name=profile_name,
            seed=seed,
            variant=variant,
            num_trajectories=num_trajectories,
            trajectory_length=effective_trajectory_length,
            latent_dim=latent_dim,
            num_channels=num_channels,
            window_length=effective_window_length,
            stride=effective_stride,
        )
        metadata_path = artifact_dir / "metadata.json"
        metadata = read_json(metadata_path)
        metadata["dataset_name"] = dataset_name
        metadata["variant"] = variant
        metadata.setdefault("long_context_protocol", {})
        metadata["long_context_protocol"] = {
            "window_length": effective_window_length,
            "stride": effective_stride,
            "trajectory_length": effective_trajectory_length,
            "design_note": "Long-context variants increase both window size and regime-supporting trajectory length.",
        }
        metadata["notes"] = list(metadata.get("notes", [])) + [
            "Long-window variant for MOMENT context ablation.",
            "Trajectory length is increased to preserve non-transition windows under longer context.",
        ]
        write_json(metadata_path, metadata)
        return artifact_dir
    raise ValueError(f"Unsupported dataset for now: {dataset_name}")


def register_dataset_artifact(manifests_dir: Path, dataset_name: str, artifact_dir: Path) -> None:
    manifests_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifests_dir / "datasets.json"
    payload: list[dict[str, Any]]
    if manifest_path.exists():
        payload = read_json(manifest_path)
    else:
        payload = []

    entry = {
        "dataset_name": dataset_name,
        "artifact_dir": str(artifact_dir),
    }
    payload = [item for item in payload if item.get("dataset_name") != dataset_name]
    payload.append(entry)
    write_json(manifest_path, payload)
