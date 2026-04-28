
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REQUIRED_STEP_COLUMNS = {
    "trajectory_id",
    "device_id",
    "t",
    "mode_id",
    "spectral_id",
    "coupling_id",
    "load",
    "segment_id",
    "is_transition_timestep",
}

SPECTRAL_NAMES = {
    0: "clean_oscillatory",
    1: "damped_oscillatory",
    2: "multi_component",
    3: "quasi_aperiodic",
}

COUPLING_NAMES = {
    0: "low",
    1: "medium",
    2: "high",
}


class InspectError(Exception):
    pass


def load_data(dataset_dir: Path) -> Tuple[pd.DataFrame, Dict[str, np.ndarray], Dict[str, object]]:
    traj_path = dataset_dir / "trajectories.csv"
    windows_path = dataset_dir / "windows.npz"
    meta_path = dataset_dir / "metadata.json"

    if not traj_path.exists():
        raise InspectError(f"Missing file: {traj_path}")
    if not windows_path.exists():
        raise InspectError(f"Missing file: {windows_path}")
    if not meta_path.exists():
        raise InspectError(f"Missing file: {meta_path}")

    df = pd.read_csv(traj_path)
    with np.load(windows_path) as data:
        windows = {k: data[k] for k in data.files}
    with open(meta_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    return df, windows, metadata


def get_channel_cols(df: pd.DataFrame) -> List[str]:
    return sorted([c for c in df.columns if c.startswith("y_")], key=lambda x: int(x.split("_")[1]))


def get_latent_cols(df: pd.DataFrame) -> List[str]:
    return sorted([c for c in df.columns if c.startswith("x_")], key=lambda x: int(x.split("_")[1]))


def boundary_positions_from_segment(g: pd.DataFrame) -> np.ndarray:
    seg = g["segment_id"].to_numpy()
    return np.where(seg[1:] != seg[:-1])[0] + 1


def load_cfg(metadata: Dict[str, object]) -> Dict[str, object]:
    return metadata.get("config", {})


def check_schema(df: pd.DataFrame, windows: Dict[str, np.ndarray], metadata: Dict[str, object]) -> Dict[str, object]:
    missing = sorted(REQUIRED_STEP_COLUMNS - set(df.columns))
    if missing:
        raise InspectError(f"Missing required columns in trajectories.csv: {missing}")

    channel_cols = get_channel_cols(df)
    latent_cols = get_latent_cols(df)

    if not channel_cols:
        raise InspectError("No observed channel columns found (expected y_0, y_1, ...)")
    if not latent_cols:
        raise InspectError("No latent columns found (expected x_0, x_1, ...)")

    required_window_keys = {
        "X",
        "trajectory_id",
        "device_id",
        "window_start",
        "mode_id",
        "spectral_id",
        "coupling_id",
        "mean_load",
        "is_transition_window",
        "distance_to_boundary",
        "left_mode_id",
        "right_mode_id",
    }
    missing_window_keys = sorted(required_window_keys - set(windows.keys()))
    if missing_window_keys:
        raise InspectError(f"Missing required arrays in windows.npz: {missing_window_keys}")

    return {
        "num_channels": len(channel_cols),
        "latent_dim": len(latent_cols),
        "channel_columns": channel_cols,
        "latent_columns": latent_cols,
        "window_tensor_shape": list(windows["X"].shape),
    }


def compute_transition_timestep_iou(df: pd.DataFrame, transition_margin: int = 4) -> float:
    scores = []
    for _, g in df.groupby("trajectory_id"):
        g = g.sort_values("t")
        mode = g["mode_id"].to_numpy()
        observed = g["is_transition_timestep"].to_numpy().astype(bool)
        boundaries = np.where(mode[1:] != mode[:-1])[0] + 1
        expected = np.zeros_like(observed, dtype=bool)
        for b in boundaries:
            left = max(0, b - transition_margin)
            right = min(len(expected), b + transition_margin + 1)
            expected[left:right] = True

        union = np.logical_or(expected, observed).sum()
        if union == 0:
            scores.append(1.0)
        else:
            inter = np.logical_and(expected, observed).sum()
            scores.append(float(inter / union))
    return float(np.mean(scores)) if scores else 0.0


def check_step_level(df: pd.DataFrame, metadata: Dict[str, object]) -> Dict[str, object]:
    cfg = load_cfg(metadata)
    issues = []

    grouped = df.groupby("trajectory_id")
    lengths = grouped.size().sort_index()

    if len(lengths) == 0:
        raise InspectError("No trajectories found in trajectories.csv")

    expected_num_trajectories = cfg.get("num_trajectories")
    expected_T = cfg.get("trajectory_length")

    if expected_num_trajectories is not None and len(lengths) != expected_num_trajectories:
        issues.append(
            f"Expected {expected_num_trajectories} trajectories but found {len(lengths)}"
        )

    unique_lengths = sorted(np.unique(lengths.values).tolist())
    if len(unique_lengths) != 1:
        issues.append(f"Trajectory lengths are not uniform: {unique_lengths}")

    if expected_T is not None and not np.all(lengths.values == expected_T):
        issues.append(
            f"Expected trajectory_length={expected_T}, observed lengths={unique_lengths}"
        )

    reconstructed_mode = df["spectral_id"].to_numpy() * 3 + df["coupling_id"].to_numpy()
    mode_match_rate = float(np.mean(reconstructed_mode == df["mode_id"].to_numpy()))
    if mode_match_rate < 1.0:
        issues.append(f"mode_id consistency={mode_match_rate:.4f}, expected 1.0")

    seg_mode_counts = (
        df.groupby(["trajectory_id", "segment_id"])["mode_id"]
        .nunique()
        .reset_index(name="n_modes")
    )
    bad_segments = int((seg_mode_counts["n_modes"] > 1).sum())
    if bad_segments > 0:
        issues.append(f"Found {bad_segments} segments containing multiple mode_ids")

    transition_iou = compute_transition_timestep_iou(
        df, transition_margin=int(cfg.get("transition_margin", 4))
    )

    return {
        "num_trajectories_found": int(len(lengths)),
        "trajectory_length_found": int(lengths.iloc[0]),
        "mode_match_rate": mode_match_rate,
        "bad_segments": bad_segments,
        "transition_timestep_iou": transition_iou,
        "issues": issues,
    }


def check_window_level(df: pd.DataFrame, windows: Dict[str, np.ndarray], metadata: Dict[str, object]) -> Dict[str, object]:
    cfg = load_cfg(metadata)
    issues = []

    window_length = int(cfg.get("window_length", windows["X"].shape[1]))
    stride = int(cfg.get("stride", 1))
    X = windows["X"]

    if X.ndim != 3:
        issues.append(f"Window tensor X should be 3D, got shape={list(X.shape)}")
    if X.shape[1] != window_length:
        issues.append(f"Window length mismatch: metadata={window_length}, array={X.shape[1]}")

    n_windows = len(windows["trajectory_id"])
    sample_indices = np.linspace(0, n_windows - 1, num=min(120, n_windows), dtype=int)

    checks = {"mode": 0, "spectral": 0, "coupling": 0, "transition": 0}
    for idx in sample_indices:
        traj_id = int(windows["trajectory_id"][idx])
        start = int(windows["window_start"][idx])

        g = df[df["trajectory_id"] == traj_id].sort_values("t").reset_index(drop=True)
        block = g.iloc[start:start + window_length]

        mode_majority = int(block["mode_id"].mode().iloc[0])
        spectral_majority = int(block["spectral_id"].mode().iloc[0])
        coupling_majority = int(block["coupling_id"].mode().iloc[0])

        boundary_positions = boundary_positions_from_segment(g)
        is_transition = bool(
            ((boundary_positions >= start) & (boundary_positions < start + window_length)).any()
        )

        checks["mode"] += int(mode_majority == int(windows["mode_id"][idx]))
        checks["spectral"] += int(spectral_majority == int(windows["spectral_id"][idx]))
        checks["coupling"] += int(coupling_majority == int(windows["coupling_id"][idx]))
        checks["transition"] += int(is_transition == bool(windows["is_transition_window"][idx]))

    for key, passed in list(checks.items()):
        rate = passed / max(1, len(sample_indices))
        checks[key] = float(rate)
        if rate < 1.0:
            issues.append(f"Sampled window {key} consistency rate={rate:.3f}, expected 1.0")

    starts = np.sort(np.unique(windows["window_start"]))
    stride_observed = int(np.min(np.diff(starts))) if len(starts) > 1 else stride

    return {
        "num_windows": int(X.shape[0]),
        "window_shape": list(X.shape),
        "sampled_consistency": checks,
        "stride_observed": stride_observed,
        "issues": issues,
    }


def summarize_switching(df: pd.DataFrame) -> Dict[str, object]:
    segment_lengths = []
    switches = []
    for _, g in df.groupby("trajectory_id"):
        g = g.sort_values("t")
        values, counts = np.unique(g["segment_id"].to_numpy(), return_counts=True)
        segment_lengths.extend(counts.tolist())
        switches.append(len(values) - 1)

    return {
        "avg_segment_length": float(np.mean(segment_lengths)),
        "min_segment_length": int(np.min(segment_lengths)),
        "max_segment_length": int(np.max(segment_lengths)),
        "avg_num_switches_per_trajectory": float(np.mean(switches)),
    }


def mean_abs_offdiag_corr_from_block(block: pd.DataFrame, channel_cols: List[str]) -> float:
    corr = block[channel_cols].corr().to_numpy()
    off_diag = corr[~np.eye(corr.shape[0], dtype=bool)]
    return float(np.mean(np.abs(off_diag)))


def summarize_coupling_global(df: pd.DataFrame, channel_cols: List[str]) -> Dict[str, object]:
    values = {}
    for coupling_id, g in df.groupby("coupling_id"):
        values[int(coupling_id)] = mean_abs_offdiag_corr_from_block(g, channel_cols)

    ordered = [values[k] for k in sorted(values.keys())]
    monotonic = bool(np.all(np.diff(ordered) > 0)) if len(ordered) >= 2 else True

    return {
        "mean_abs_offdiag_corr_by_coupling": values,
        "coupling_monotonic_trend_ok": monotonic,
        "scope": "global_all_timesteps_mixed",
    }


def collect_clean_segment_blocks(
    df: pd.DataFrame,
    min_len: int,
) -> List[pd.DataFrame]:
    blocks = []
    for _, g in df.groupby("trajectory_id"):
        g = g.sort_values("t").reset_index(drop=True)
        for _, seg_block in g.groupby("segment_id"):
            if seg_block["is_transition_timestep"].sum() > 0:
                clean = seg_block[seg_block["is_transition_timestep"] == 0].copy()
            else:
                clean = seg_block.copy()
            if len(clean) >= min_len:
                blocks.append(clean.reset_index(drop=True))
    return blocks


def summarize_coupling_clean_blocks(
    df: pd.DataFrame,
    channel_cols: List[str],
    min_len: int = 48,
) -> Dict[str, object]:
    by_coupling: Dict[int, List[float]] = {}
    block_counts: Dict[int, int] = {}

    for block in collect_clean_segment_blocks(df, min_len=min_len):
        if block["coupling_id"].nunique() != 1:
            continue
        coupling_id = int(block["coupling_id"].iloc[0])
        by_coupling.setdefault(coupling_id, []).append(
            mean_abs_offdiag_corr_from_block(block, channel_cols)
        )
        block_counts[coupling_id] = block_counts.get(coupling_id, 0) + 1

    mean_values = {k: float(np.mean(v)) for k, v in by_coupling.items()}
    ordered_keys = sorted(mean_values.keys())
    ordered_vals = [mean_values[k] for k in ordered_keys]
    monotonic = bool(np.all(np.diff(ordered_vals) > 0)) if len(ordered_vals) >= 2 else True

    return {
        "mean_abs_offdiag_corr_by_coupling": mean_values,
        "num_clean_blocks_by_coupling": {int(k): int(v) for k, v in block_counts.items()},
        "coupling_monotonic_trend_ok": monotonic,
        "scope": "clean_segment_level_average",
    }


def summarize_spectral(df: pd.DataFrame, channel_cols: List[str]) -> Dict[str, object]:
    dominant_bins = {}
    dominant_energy_ratio = {}

    for spectral_id, g in df.groupby("spectral_id"):
        traj_stats = []
        energy_stats = []
        for _, tg in g.groupby("trajectory_id"):
            sig = tg.sort_values("t")[channel_cols[0]].to_numpy()
            sig = sig - sig.mean()
            fft = np.abs(np.fft.rfft(sig))
            if len(fft) <= 1:
                continue
            idx = int(np.argmax(fft[1:]) + 1)
            ratio = float(fft[idx] / (np.sum(fft[1:]) + 1e-8))
            traj_stats.append(idx)
            energy_stats.append(ratio)
        dominant_bins[int(spectral_id)] = float(np.mean(traj_stats)) if traj_stats else 0.0
        dominant_energy_ratio[int(spectral_id)] = float(np.mean(energy_stats)) if energy_stats else 0.0

    return {
        "mean_dominant_fft_bin_y0_by_spectral": dominant_bins,
        "mean_peak_energy_ratio_y0_by_spectral": dominant_energy_ratio,
    }


def pick_representative_trajectory(df: pd.DataFrame) -> int:
    scores = []
    for traj_id, g in df.groupby("trajectory_id"):
        g = g.sort_values("t")
        n_segments = int(g["segment_id"].nunique())
        n_transition = int(g["is_transition_timestep"].sum())
        score = n_segments * 10 - abs(n_transition - 20)
        scores.append((score, int(traj_id)))
    scores.sort(reverse=True)
    return scores[0][1]


def plot_representative_trajectory(df: pd.DataFrame, output_dir: Path, trajectory_id: int | None = None) -> Path:
    channel_cols = get_channel_cols(df)
    if trajectory_id is None:
        trajectory_id = pick_representative_trajectory(df)

    g = df[df["trajectory_id"] == trajectory_id].sort_values("t").reset_index(drop=True)
    boundaries = boundary_positions_from_segment(g)
    t = g["t"].to_numpy()

    fig, axes = plt.subplots(len(channel_cols), 1, figsize=(14, 9), sharex=True)
    if len(channel_cols) == 1:
        axes = [axes]

    for i, col in enumerate(channel_cols):
        axes[i].plot(t, g[col].to_numpy(), linewidth=1.2)
        axes[i].set_ylabel(col)
        for b in boundaries:
            axes[i].axvline(x=b, linestyle="--", linewidth=1.0)
        transition_idx = np.where(g["is_transition_timestep"].to_numpy() == 1)[0]
        if len(transition_idx) > 0:
            axes[i].scatter(
                transition_idx,
                g.iloc[transition_idx][col].to_numpy(),
                s=8,
                alpha=0.6,
            )

    axes[0].set_title(
        f"Trajectory {trajectory_id} | segments={g['segment_id'].nunique()} | device={g['device_id'].iloc[0]}"
    )
    axes[-1].set_xlabel("t")
    fig.tight_layout()

    out_path = output_dir / f"trajectory_{trajectory_id}_overview.png"
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out_path


def pick_clean_spectral_block(df: pd.DataFrame, spectral_id: int, min_len: int = 48) -> pd.DataFrame | None:
    candidates = []
    for _, g in df.groupby("trajectory_id"):
        g = g.sort_values("t").reset_index(drop=True)
        for _, block in g.groupby("segment_id"):
            if int(block["spectral_id"].iloc[0]) != spectral_id:
                continue
            if block["spectral_id"].nunique() != 1:
                continue
            clean = block[block["is_transition_timestep"] == 0].copy()
            if len(clean) >= min_len:
                candidates.append(clean.iloc[:min_len].copy())
    if not candidates:
        return None
    candidates.sort(key=lambda x: -len(x))
    return candidates[0]


def plot_spectral_family_comparison(df: pd.DataFrame, output_dir: Path, window_len: int = 48) -> Path:
    channel_cols = get_channel_cols(df)
    spectral_ids = sorted(df["spectral_id"].unique().tolist())

    fig, axes = plt.subplots(len(spectral_ids), 1, figsize=(14, 10), sharex=True)
    if len(spectral_ids) == 1:
        axes = [axes]

    for ax, spectral_id in zip(axes, spectral_ids):
        block = pick_clean_spectral_block(df, int(spectral_id), min_len=window_len)
        if block is None:
            ax.text(0.5, 0.5, f"No clean block for spectral_id={spectral_id}", ha="center", va="center")
            ax.set_axis_off()
            continue

        x = np.arange(len(block))
        for col in channel_cols:
            ax.plot(x, block[col].to_numpy(), linewidth=1.2, label=col)

        load_mean = float(block["load"].mean())
        coupling_mode = int(block["coupling_id"].mode().iloc[0])
        ax.set_title(
            f"spectral_id={spectral_id} ({SPECTRAL_NAMES.get(int(spectral_id), spectral_id)}) | "
            f"coupling={COUPLING_NAMES.get(coupling_mode, coupling_mode)} | "
            f"mean_load={load_mean:.3f}"
        )
        ax.set_ylabel("value")

    axes[-1].set_xlabel("relative timestep")
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper right")
    fig.tight_layout()

    out_path = output_dir / "spectral_family_comparison.png"
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out_path


def pick_clean_coupling_block(df: pd.DataFrame, coupling_id: int, min_len: int = 64) -> pd.DataFrame | None:
    candidates = []
    for _, g in df.groupby("trajectory_id"):
        g = g.sort_values("t").reset_index(drop=True)
        for _, block in g.groupby("segment_id"):
            if int(block["coupling_id"].iloc[0]) != coupling_id:
                continue
            if block["coupling_id"].nunique() != 1:
                continue
            clean = block[block["is_transition_timestep"] == 0].copy()
            if len(clean) >= min_len:
                candidates.append(clean.iloc[:min_len].copy())
    if not candidates:
        return None
    candidates.sort(key=lambda x: -len(x))
    return candidates[0]


def average_clean_block_correlation(df: pd.DataFrame, channel_cols: List[str], coupling_id: int, min_len: int = 48) -> np.ndarray | None:
    mats = []
    for block in collect_clean_segment_blocks(df, min_len=min_len):
        if block["coupling_id"].nunique() != 1:
            continue
        if int(block["coupling_id"].iloc[0]) != coupling_id:
            continue
        mats.append(block[channel_cols].corr().to_numpy())
    if not mats:
        return None
    return np.mean(np.stack(mats, axis=0), axis=0)


def plot_coupling_heatmaps_global(df: pd.DataFrame, output_dir: Path) -> Path:
    channel_cols = get_channel_cols(df)
    coupling_ids = sorted(df["coupling_id"].unique().tolist())

    fig, axes = plt.subplots(1, len(coupling_ids), figsize=(4.5 * len(coupling_ids), 4))
    if len(coupling_ids) == 1:
        axes = [axes]

    im = None
    for ax, coupling_id in zip(axes, coupling_ids):
        corr = df[df["coupling_id"] == coupling_id][channel_cols].corr().to_numpy()
        im = ax.imshow(corr, vmin=-1, vmax=1, cmap="RdBu_r")
        ax.set_title(f"global coupling={COUPLING_NAMES.get(int(coupling_id), coupling_id)}")
        ax.set_xticks(np.arange(len(channel_cols)))
        ax.set_yticks(np.arange(len(channel_cols)))
        ax.set_xticklabels(channel_cols, rotation=45, ha="right")
        ax.set_yticklabels(channel_cols)

        for i in range(corr.shape[0]):
            for j in range(corr.shape[1]):
                ax.text(
                    j, i,
                    f"{corr[i, j]:.2f}",
                    ha="center",
                    va="center",
                    color="white" if abs(corr[i, j]) > 0.5 else "black",
                    fontsize=8
                )

    if im is not None:
        cbar = fig.colorbar(im, ax=axes.ravel().tolist(), location='right')
        cbar.set_label("correlation")
    fig.tight_layout()

    out_path = output_dir / "coupling_heatmaps_global.png"
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_coupling_heatmaps_clean_average(df: pd.DataFrame, output_dir: Path, min_len: int = 48) -> Path:
    channel_cols = get_channel_cols(df)
    coupling_ids = sorted(df["coupling_id"].unique().tolist())

    fig, axes = plt.subplots(1, len(coupling_ids), figsize=(4.5 * len(coupling_ids), 4))
    if len(coupling_ids) == 1:
        axes = [axes]

    im = None
    for ax, coupling_id in zip(axes, coupling_ids):
        corr = average_clean_block_correlation(df, channel_cols, int(coupling_id), min_len=min_len)
        if corr is None:
            ax.text(0.5, 0.5, "No clean blocks", ha="center", va="center")
            ax.set_axis_off()
            continue
        im = ax.imshow(corr, vmin=-1, vmax=1, cmap="RdBu_r")
        ax.set_title(f"clean avg coupling={COUPLING_NAMES.get(int(coupling_id), coupling_id)}")
        ax.set_xticks(np.arange(len(channel_cols)))
        ax.set_yticks(np.arange(len(channel_cols)))
        ax.set_xticklabels(channel_cols, rotation=45, ha="right")
        ax.set_yticklabels(channel_cols)

        for i in range(corr.shape[0]):
            for j in range(corr.shape[1]):
                ax.text(
                    j, i,
                    f"{corr[i, j]:.2f}",
                    ha="center",
                    va="center",
                    color="white" if abs(corr[i, j]) > 0.5 else "black",
                    fontsize=8
                )

    if im is not None:
        cbar = fig.colorbar(im, ax=axes.ravel().tolist(), location='right')
        cbar.set_label("correlation")
    fig.tight_layout()

    out_path = output_dir / "coupling_heatmaps_clean_average.png"
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_coupling_heatmaps_representative(df: pd.DataFrame, output_dir: Path, min_len: int = 64) -> Path:
    channel_cols = get_channel_cols(df)
    coupling_ids = sorted(df["coupling_id"].unique().tolist())

    fig, axes = plt.subplots(1, len(coupling_ids), figsize=(4.5 * len(coupling_ids), 4))
    if len(coupling_ids) == 1:
        axes = [axes]

    im = None
    for ax, coupling_id in zip(axes, coupling_ids):
        block = pick_clean_coupling_block(df, int(coupling_id), min_len=min_len)
        if block is None:
            ax.text(0.5, 0.5, "No clean block", ha="center", va="center")
            ax.set_axis_off()
            continue
        corr = block[channel_cols].corr().to_numpy()
        im = ax.imshow(corr, vmin=-1, vmax=1, cmap="RdBu_r")
        ax.set_title(f"representative coupling={COUPLING_NAMES.get(int(coupling_id), coupling_id)}")
        ax.set_xticks(np.arange(len(channel_cols)))
        ax.set_yticks(np.arange(len(channel_cols)))
        ax.set_xticklabels(channel_cols, rotation=45, ha="right")
        ax.set_yticklabels(channel_cols)

        for i in range(corr.shape[0]):
            for j in range(corr.shape[1]):
                ax.text(
                    j, i,
                    f"{corr[i, j]:.2f}",
                    ha="center",
                    va="center",
                    color="white" if abs(corr[i, j]) > 0.5 else "black",
                    fontsize=8
                )

    if im is not None:
        cbar = fig.colorbar(im, ax=axes.ravel().tolist(), location='right')
        cbar.set_label("correlation")
    fig.tight_layout()

    out_path = output_dir / "coupling_heatmaps_representative.png"
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out_path


def find_good_transition(df: pd.DataFrame, context: int = 24) -> Tuple[int, int] | None:
    for traj_id, g in df.groupby("trajectory_id"):
        g = g.sort_values("t").reset_index(drop=True)
        boundaries = boundary_positions_from_segment(g)
        T = len(g)
        for b in boundaries:
            if b - context >= 0 and b + context < T:
                return int(traj_id), int(b)
    return None


def plot_transition_zoom(df: pd.DataFrame, output_dir: Path, context: int = 24) -> Path:
    channel_cols = get_channel_cols(df)
    found = find_good_transition(df, context=context)
    if found is None:
        raise InspectError("No valid boundary found for transition zoom plot.")

    trajectory_id, boundary = found
    g = df[df["trajectory_id"] == trajectory_id].sort_values("t").reset_index(drop=True)
    block = g.iloc[boundary - context : boundary + context].copy().reset_index(drop=True)
    x = np.arange(len(block)) - context

    fig, axes = plt.subplots(len(channel_cols), 1, figsize=(14, 8), sharex=True)
    if len(channel_cols) == 1:
        axes = [axes]

    for i, col in enumerate(channel_cols):
        axes[i].plot(x, block[col].to_numpy(), linewidth=1.2)
        axes[i].axvline(x=0, linestyle="--", linewidth=1.2)
        transition_idx = np.where(block["is_transition_timestep"].to_numpy() == 1)[0]
        if len(transition_idx) > 0:
            axes[i].scatter(
                x[transition_idx],
                block.iloc[transition_idx][col].to_numpy(),
                s=10,
                alpha=0.7,
            )
        axes[i].set_ylabel(col)

    left_mode = int(g["mode_id"].iloc[max(0, boundary - 1)])
    right_mode = int(g["mode_id"].iloc[min(len(g) - 1, boundary)])
    axes[0].set_title(
        f"Transition zoom | trajectory={trajectory_id} | boundary={boundary} | "
        f"left_mode={left_mode} | right_mode={right_mode}"
    )
    axes[-1].set_xlabel("relative timestep (0 = boundary)")
    fig.tight_layout()

    out_path = output_dir / f"transition_zoom_traj_{trajectory_id}_b_{boundary}.png"
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_spectral_summary(df: pd.DataFrame, output_dir: Path) -> Path:
    channel_cols = get_channel_cols(df)
    dom = summarize_spectral(df, channel_cols)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    x1 = list(dom["mean_dominant_fft_bin_y0_by_spectral"].keys())
    y1 = list(dom["mean_dominant_fft_bin_y0_by_spectral"].values())
    axes[0].bar(x1, y1)
    axes[0].set_title("Dominant FFT bin on y_0")
    axes[0].set_xlabel("spectral_id")
    axes[0].set_ylabel("mean bin")

    x2 = list(dom["mean_peak_energy_ratio_y0_by_spectral"].keys())
    y2 = list(dom["mean_peak_energy_ratio_y0_by_spectral"].values())
    axes[1].bar(x2, y2)
    axes[1].set_title("Peak energy ratio on y_0")
    axes[1].set_xlabel("spectral_id")
    axes[1].set_ylabel("mean ratio")

    fig.tight_layout()
    out_path = output_dir / "spectral_summary.png"
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out_path


def build_overall_status(
    step_level: Dict[str, object],
    window_level: Dict[str, object],
    coupling_clean: Dict[str, object],
) -> str:
    hard_fail = (
        len(step_level["issues"]) > 0
        or len(window_level["issues"]) > 0
        or step_level["mode_match_rate"] < 1.0
        or step_level["bad_segments"] > 0
        or window_level["sampled_consistency"]["mode"] < 1.0
        or window_level["sampled_consistency"]["spectral"] < 1.0
        or window_level["sampled_consistency"]["coupling"] < 1.0
    )
    if hard_fail:
        return "fail"

    if not coupling_clean["coupling_monotonic_trend_ok"]:
        return "warning"

    return "pass"


def save_text_summary(path: Path, report: Dict[str, object]) -> None:
    lines = []
    lines.append(f"Overall status: {report['overall_status']}")
    lines.append("")

    lines.append("[Schema]")
    for k, v in report["schema"].items():
        lines.append(f"- {k}: {v}")

    lines.append("")
    lines.append("[Step-level checks]")
    for k, v in report["step_level"].items():
        if k != "issues":
            lines.append(f"- {k}: {v}")
    if report["step_level"]["issues"]:
        lines.append("- issues:")
        for item in report["step_level"]["issues"]:
            lines.append(f"  * {item}")

    lines.append("")
    lines.append("[Window-level checks]")
    for k, v in report["window_level"].items():
        if k != "issues":
            lines.append(f"- {k}: {v}")
    if report["window_level"]["issues"]:
        lines.append("- issues:")
        for item in report["window_level"]["issues"]:
            lines.append(f"  * {item}")

    lines.append("")
    lines.append("[Switching summary]")
    for k, v in report["switching"].items():
        lines.append(f"- {k}: {v}")

    lines.append("")
    lines.append("[Coupling summary: global]")
    for k, v in report["coupling"]["global"].items():
        lines.append(f"- {k}: {v}")

    lines.append("")
    lines.append("[Coupling summary: clean blocks]")
    for k, v in report["coupling"]["clean_blocks"].items():
        lines.append(f"- {k}: {v}")

    lines.append("")
    lines.append("[Spectral summary]")
    for k, v in report["spectral"].items():
        lines.append(f"- {k}: {v}")

    lines.append("")
    lines.append("[How to read plots]")
    lines.append("- trajectory_overview: segment boundaries should align with behavior changes.")
    lines.append("- spectral_family_comparison: temporal style should differ across spectral families.")
    lines.append("- coupling_heatmaps_global: coarse mixed summary across all timesteps.")
    lines.append("- coupling_heatmaps_clean_average: cleaner factor-level summary across clean segments.")
    lines.append("- coupling_heatmaps_representative: one representative clean block per coupling level.")
    lines.append("- transition_zoom: boundary region should look like a short transition, not random noise.")

    path.write_text("\n".join(lines), encoding="utf-8")


def run_plots(df: pd.DataFrame, output_dir: Path, trajectory_id: int | None = None) -> List[Path]:
    files = []
    files.append(plot_representative_trajectory(df, output_dir, trajectory_id=trajectory_id))
    files.append(plot_spectral_family_comparison(df, output_dir))
    files.append(plot_coupling_heatmaps_global(df, output_dir))
    files.append(plot_coupling_heatmaps_clean_average(df, output_dir))
    files.append(plot_coupling_heatmaps_representative(df, output_dir))
    files.append(plot_transition_zoom(df, output_dir))
    files.append(plot_spectral_summary(df, output_dir))
    return files


def main() -> None:
    parser = argparse.ArgumentParser(description="Unified inspection script for generator_v2 datasets.")
    parser.add_argument("--dataset-dir", type=str, required=True, help="Directory containing trajectories.csv, windows.npz, metadata.json")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory (default: <dataset-dir>/inspection)")
    parser.add_argument("--trajectory-id", type=int, default=None, help="Optional trajectory id for the overview plot")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    output_dir = Path(args.output_dir) if args.output_dir else dataset_dir / "inspection"
    output_dir.mkdir(parents=True, exist_ok=True)

    df, windows, metadata = load_data(dataset_dir)

    schema = check_schema(df, windows, metadata)
    step_level = check_step_level(df, metadata)
    window_level = check_window_level(df, windows, metadata)
    switching = summarize_switching(df)
    coupling_global = summarize_coupling_global(df, schema["channel_columns"])
    coupling_clean = summarize_coupling_clean_blocks(
        df, schema["channel_columns"], min_len=int(load_cfg(metadata).get("window_length", 48))
    )
    spectral = summarize_spectral(df, schema["channel_columns"])

    report = {
        "overall_status": build_overall_status(step_level, window_level, coupling_clean),
        "schema": schema,
        "step_level": step_level,
        "window_level": window_level,
        "switching": switching,
        "coupling": {
            "global": coupling_global,
            "clean_blocks": coupling_clean,
        },
        "spectral": spectral,
    }

    plot_files = run_plots(df, output_dir, trajectory_id=args.trajectory_id)
    report["generated_files"] = [str(p.name) for p in plot_files]

    json_path = output_dir / "inspection_report.json"
    txt_path = output_dir / "inspection_summary.txt"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    save_text_summary(txt_path, report)

    print(f"Saved report: {json_path}")
    print(f"Saved summary: {txt_path}")
    print(f"Saved plots to: {output_dir}")
    print(f"Overall status: {report['overall_status']}")


if __name__ == "__main__":
    main()
