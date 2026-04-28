from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
from moment.data.synthetic_data import SyntheticDataset


@dataclass
class MomentSyntheticConfig:
    seed: int = 42
    variant: str = "varying_freq"

    n_samples: int = 1024
    seq_len: int = 512

    freq_range: Tuple[int, int] = (1, 32)
    amplitude_range: Tuple[int, int] = (1, 32)
    trend_range: Tuple[int, int] = (1, 32)
    baseline_range: Tuple[int, int] = (1, 32)

    noise_mean: float = 0.0
    noise_std: float = 0.1

    num_bins: int = 8


class MomentSyntheticGenerator:
    """
    Convert MOMENT SyntheticDataset output into a version2/FSSS-compatible bundle:
      - trajectories.csv
      - windows.npz
      - metadata.json

    First baseline only supports one window per sample:
      each synthetic sequence is treated as one full window.
    """

    def __init__(self, config: MomentSyntheticConfig):
        self.config = config

    def generate_dataset(self) -> Dict[str, object]:
        cfg = self.config

        ds = SyntheticDataset(
            n_samples=cfg.n_samples,
            seq_len=cfg.seq_len,
            freq_range=cfg.freq_range,
            amplitude_range=cfg.amplitude_range,
            trend_range=cfg.trend_range,
            baseline_range=cfg.baseline_range,
            noise_mean=cfg.noise_mean,
            noise_std=cfg.noise_std,
            random_seed=cfg.seed,
        )

        if cfg.variant == "varying_freq":
            y, c = ds.gen_sinusoids_with_varying_freq()
            control_name = "frequency"
        elif cfg.variant == "varying_amplitude":
            y, c = ds.gen_sinusoids_with_varying_amplitude()
            control_name = "amplitude"
        elif cfg.variant == "varying_trend":
            y, c = ds.gen_sinusoids_with_varying_trend()
            control_name = "trend"
        elif cfg.variant == "varying_baseline":
            y, c = ds.gen_sinusoids_with_varying_baseline()
            control_name = "baseline"
        else:
            raise ValueError(f"Unsupported variant: {cfg.variant}")

        # y: [N, 1, T] in MOMENT code
        # Convert to [N, T, C] for your pipeline
        X = y.transpose(1, 2).detach().cpu().numpy().astype(np.float32)  # [N, T, 1]

        # c: [N, T], same control value repeated along time for each sample
        control_value = c[:, 0].detach().cpu().numpy().astype(np.float32)

        bins = self._bin_values(control_value, cfg.num_bins)

        windows = self._build_windows(
            X=X,
            control_value=control_value,
            bins=bins,
        )
        trajectory_df = self._build_trajectory_df(
            X=X,
            control_value=control_value,
            bins=bins,
            control_name=control_name,
        )
        metadata = self._build_metadata(
            X=X,
            control_value=control_value,
            bins=bins,
            control_name=control_name,
        )

        return {
            "trajectory_df": trajectory_df,
            "windows": windows,
            "metadata": metadata,
        }

    def _bin_values(self, values: np.ndarray, num_bins: int) -> np.ndarray:
        edges = np.linspace(values.min(), values.max(), num_bins + 1)
        # right=False => [edge_i, edge_{i+1})
        bins = np.digitize(values, edges[1:-1], right=False)
        return bins.astype(np.int64)

    def _build_windows(
        self,
        X: np.ndarray,
        control_value: np.ndarray,
        bins: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        n, seq_len, _ = X.shape

        trajectory_id = np.arange(n, dtype=np.int64)
        device_id = np.zeros(n, dtype=np.int64)
        window_start = np.zeros(n, dtype=np.int64)

        # Main labels for evaluation
        mode_id = bins.copy()
        spectral_id = bins.copy()

        # Placeholder fields to stay compatible with version2/FSSS pipeline
        coupling_id = np.zeros(n, dtype=np.int64)
        mean_load = np.zeros(n, dtype=np.float32)
        is_transition_window = np.zeros(n, dtype=np.bool_)
        distance_to_boundary = np.full(n, seq_len, dtype=np.int64)
        left_mode_id = np.full(n, -1, dtype=np.int64)
        right_mode_id = np.full(n, -1, dtype=np.int64)

        return {
            "X": X.astype(np.float32),
            "trajectory_id": trajectory_id,
            "device_id": device_id,
            "window_start": window_start,
            "mode_id": mode_id,
            "spectral_id": spectral_id,
            "coupling_id": coupling_id,
            "mean_load": mean_load,
            "is_transition_window": is_transition_window,
            "distance_to_boundary": distance_to_boundary,
            "left_mode_id": left_mode_id,
            "right_mode_id": right_mode_id,
            # Extra metadata for this benchmark
            "control_value": control_value.astype(np.float32),
        }

    def _build_trajectory_df(
        self,
        X: np.ndarray,
        control_value: np.ndarray,
        bins: np.ndarray,
        control_name: str,
    ) -> pd.DataFrame:
        rows = []
        for i in range(X.shape[0]):
            rows.append(
                {
                    "trajectory_id": int(i),
                    "device_id": 0,
                    "window_start": 0,
                    "seq_len": int(X.shape[1]),
                    f"{control_name}_value": float(control_value[i]),
                    "mode_id": int(bins[i]),
                    "spectral_id": int(bins[i]),
                }
            )
        return pd.DataFrame(rows)

    def _build_metadata(
        self,
        X: np.ndarray,
        control_value: np.ndarray,
        bins: np.ndarray,
        control_name: str,
    ) -> Dict[str, object]:
        return {
            "dataset_name": "moment_synthetic_baseline",
            "source": "SyntheticDataset",
            "variant": self.config.variant,
            "control_name": control_name,
            "config": asdict(self.config),
            "num_trajectories": int(X.shape[0]),
            "num_windows": int(X.shape[0]),
            "window_shape": [int(X.shape[1]), int(X.shape[2])],
            "unique_mode_ids": sorted(np.unique(bins).astype(int).tolist()),
            "unique_spectral_ids": sorted(np.unique(bins).astype(int).tolist()),
            "unique_coupling_ids": [0],
            "num_devices": 1,
            "transition_window_fraction": 0.0,
            "control_min": float(control_value.min()),
            "control_max": float(control_value.max()),
        }


def save_dataset(output_dir: Path, dataset: Dict[str, object]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    trajectory_df: pd.DataFrame = dataset["trajectory_df"]
    windows: Dict[str, np.ndarray] = dataset["windows"]
    metadata: Dict[str, object] = dataset["metadata"]

    trajectory_df.to_csv(output_dir / "trajectories.csv", index=False)
    np.savez_compressed(output_dir / "windows.npz", **windows)

    with open(output_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved trajectory table -> {output_dir / 'trajectories.csv'}")
    print(f"Saved windows         -> {output_dir / 'windows.npz'}")
    print(f"Saved metadata        -> {output_dir / 'metadata.json'}")
    print(json.dumps(metadata, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate MOMENT synthetic baseline dataset.")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument(
        "--variant",
        type=str,
        default="varying_freq",
        choices=["varying_freq", "varying_amplitude", "varying_trend", "varying_baseline"],
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_samples", type=int, default=1024)
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--noise_std", type=float, default=0.1)
    parser.add_argument("--num_bins", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    cfg = MomentSyntheticConfig(
        seed=args.seed,
        variant=args.variant,
        n_samples=args.n_samples,
        seq_len=args.seq_len,
        noise_std=args.noise_std,
        num_bins=args.num_bins,
    )

    generator = MomentSyntheticGenerator(cfg)
    dataset = generator.generate_dataset()
    save_dataset(Path(args.output_dir), dataset)


if __name__ == "__main__":
    main()