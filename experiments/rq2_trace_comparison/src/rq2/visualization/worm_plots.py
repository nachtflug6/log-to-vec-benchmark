"""Visualization helpers for RQ2 trace comparison.

All functions accept embeddings [N, D] and mode labels [N], project to 2D
via PCA fitted on the same data, then produce matplotlib figures.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection
from sklearn.decomposition import PCA


# Consistent color palette for up to 6 modes
_MODE_COLORS = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00", "#a65628"]


def _fit_pca_2d(embeddings: np.ndarray) -> Tuple[PCA, np.ndarray]:
    pca = PCA(n_components=2, random_state=0)
    emb_2d = pca.fit_transform(embeddings)
    return pca, emb_2d


def _mode_color(mode_id: int) -> str:
    return _MODE_COLORS[mode_id % len(_MODE_COLORS)]


def _draw_confidence_ellipse(ax, pts: np.ndarray, color: str, n_std: float = 1.0) -> None:
    """Draw a 1-sigma ellipse around pts using eigendecomposition of covariance."""
    if len(pts) < 3:
        return
    cov = np.cov(pts.T)
    if cov.ndim < 2:
        return
    try:
        vals, vecs = np.linalg.eigh(cov)
    except np.linalg.LinAlgError:
        return
    vals = np.maximum(vals, 0)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    angle = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    width, height = 2 * n_std * np.sqrt(vals)
    ell = mpatches.Ellipse(
        xy=pts.mean(axis=0),
        width=width,
        height=height,
        angle=angle,
        edgecolor=color,
        facecolor="none",
        linewidth=1.5,
        linestyle="--",
        alpha=0.7,
    )
    ax.add_patch(ell)


# ---------------------------------------------------------------------------
# Plot 1: PCA worm plot (full production run, colored by time)
# ---------------------------------------------------------------------------

def plot_worm(
    embeddings: np.ndarray,
    mode_ids: np.ndarray,
    title: str = "PCA Worm Plot",
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Worm plot: PC1 vs PC2, colored continuously by time index.

    Arrows show trajectory direction. Mode regions are lightly annotated.
    """
    _, emb_2d = _fit_pca_2d(embeddings)
    N = len(emb_2d)

    fig, ax = plt.subplots(figsize=(7, 6))

    # Draw colored line segments (time = color)
    points = emb_2d.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = LineCollection(segments, cmap="plasma", linewidth=1.2, alpha=0.8)
    lc.set_array(np.linspace(0, 1, N - 1))
    ax.add_collection(lc)

    # Mode-colored scatter (small dots)
    for m in np.unique(mode_ids):
        mask = mode_ids == m
        ax.scatter(
            emb_2d[mask, 0],
            emb_2d[mask, 1],
            color=_mode_color(m),
            s=12,
            alpha=0.5,
            label=f"Mode {m}",
            zorder=3,
        )

    # Start / end markers
    ax.scatter(*emb_2d[0], color="black", s=60, marker="o", zorder=5, label="Start")
    ax.scatter(*emb_2d[-1], color="black", s=60, marker="X", zorder=5, label="End")

    cbar = fig.colorbar(cm.ScalarMappable(cmap="plasma"), ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label("Time →")

    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.6)
    ax.set_aspect("equal", adjustable="datalim")
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
        plt.close(fig)

    return fig


# ---------------------------------------------------------------------------
# Plot 2: Mode loop overlay (all repetitions of each mode overlaid)
# ---------------------------------------------------------------------------

def plot_mode_loops(
    embeddings: np.ndarray,
    mode_ids: np.ndarray,
    change_point_windows: List[int],
    title: str = "Mode Loop Overlay",
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Overlay all contiguous segments of the same mode in PCA space.

    Each repetition of mode A is drawn in a semi-transparent line;
    the centroid is marked with a cross.
    """
    _, emb_2d = _fit_pca_2d(embeddings)

    fig, ax = plt.subplots(figsize=(7, 6))

    # Segment the trace into contiguous mode runs
    boundaries = sorted(set([0] + list(change_point_windows) + [len(embeddings)]))
    seen_modes = set()

    for start, end in zip(boundaries[:-1], boundaries[1:]):
        if end <= start:
            continue
        seg_labels = mode_ids[start:end]
        unique, counts = np.unique(seg_labels, return_counts=True)
        m = int(unique[counts.argmax()])
        color = _mode_color(m)

        seg_2d = emb_2d[start:end]
        label = f"Mode {m}" if m not in seen_modes else None
        ax.plot(seg_2d[:, 0], seg_2d[:, 1], color=color, alpha=0.4, linewidth=1.0, label=label)
        seen_modes.add(m)

    # Draw per-mode centroids and ellipses
    for m in np.unique(mode_ids):
        pts = emb_2d[mode_ids == m]
        centroid = pts.mean(axis=0)
        color = _mode_color(m)
        ax.scatter(*centroid, color=color, s=80, marker="+", linewidths=2, zorder=5)
        _draw_confidence_ellipse(ax, pts, color=color)

    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.6)
    ax.set_aspect("equal", adjustable="datalim")
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
        plt.close(fig)

    return fig


# ---------------------------------------------------------------------------
# Plot 3: Centroid + 1σ ellipse panel
# ---------------------------------------------------------------------------

def plot_centroids(
    embeddings: np.ndarray,
    mode_ids: np.ndarray,
    title: str = "Mode Centroids (PCA)",
    save_path: Optional[Path] = None,
) -> plt.Figure:
    _, emb_2d = _fit_pca_2d(embeddings)

    fig, ax = plt.subplots(figsize=(6, 5))

    for m in np.unique(mode_ids):
        pts = emb_2d[mode_ids == m]
        color = _mode_color(m)
        ax.scatter(pts[:, 0], pts[:, 1], color=color, s=8, alpha=0.25)
        centroid = pts.mean(axis=0)
        ax.scatter(*centroid, color=color, s=120, marker="*", zorder=5, label=f"Mode {m}")
        _draw_confidence_ellipse(ax, pts, color=color, n_std=1.0)

    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.6)
    ax.set_aspect("equal", adjustable="datalim")
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
        plt.close(fig)

    return fig


# ---------------------------------------------------------------------------
# Plot 4: Mode centroid distance over time
# ---------------------------------------------------------------------------

def plot_centroid_distance_over_time(
    embeddings: np.ndarray,
    mode_ids: np.ndarray,
    change_point_windows: List[int],
    title: str = "Distance to Mode Centroid Over Time",
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """For each window, plot its L2 distance to each mode centroid.

    Ground-truth mode change points are shown as vertical dashed lines.
    """
    unique_modes = np.unique(mode_ids)
    centroids = {m: embeddings[mode_ids == m].mean(axis=0) for m in unique_modes}

    N = len(embeddings)
    t = np.arange(N)

    fig, ax = plt.subplots(figsize=(10, 4))

    for m in unique_modes:
        dists = np.linalg.norm(embeddings - centroids[m], axis=1)
        ax.plot(t, dists, color=_mode_color(m), linewidth=1.0, alpha=0.85, label=f"→ Mode {m}")

    for cp in change_point_windows:
        ax.axvline(x=cp, color="black", linewidth=0.8, linestyle="--", alpha=0.5)

    ax.set_xlabel("Window index")
    ax.set_ylabel("L2 distance to centroid")
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.6)
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
        plt.close(fig)

    return fig


# ---------------------------------------------------------------------------
# Plot 5: Pairwise mode distance heatmap
# ---------------------------------------------------------------------------

def plot_mode_distance_heatmap(
    embeddings: np.ndarray,
    mode_ids: np.ndarray,
    title: str = "Pairwise Mode Centroid Distance",
    save_path: Optional[Path] = None,
) -> plt.Figure:
    unique_modes = sorted(np.unique(mode_ids).tolist())
    centroids = np.stack([embeddings[mode_ids == m].mean(axis=0) for m in unique_modes])

    n = len(unique_modes)
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            dist_matrix[i, j] = float(np.linalg.norm(centroids[i] - centroids[j]))

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(dist_matrix, cmap="Blues", aspect="auto")
    fig.colorbar(im, ax=ax, label="L2 distance")

    labels = [f"Mode {m}" for m in unique_modes]
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)

    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{dist_matrix[i, j]:.2f}", ha="center", va="center", fontsize=8)

    ax.set_title(title)
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
        plt.close(fig)

    return fig


# ---------------------------------------------------------------------------
# Batch runner
# ---------------------------------------------------------------------------

def save_all_plots(
    embeddings: np.ndarray,
    mode_ids: np.ndarray,
    change_point_windows: List[int],
    output_dir: Path,
    prefix: str = "",
) -> None:
    """Save all 5 plot types to output_dir with the given prefix."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    p = f"{prefix}_" if prefix else ""

    plot_worm(
        embeddings, mode_ids,
        title=f"PCA Worm — {prefix}",
        save_path=output_dir / f"{p}worm.png",
    )
    plot_mode_loops(
        embeddings, mode_ids, change_point_windows,
        title=f"Mode Loops — {prefix}",
        save_path=output_dir / f"{p}mode_loops.png",
    )
    plot_centroids(
        embeddings, mode_ids,
        title=f"Centroids — {prefix}",
        save_path=output_dir / f"{p}centroids.png",
    )
    plot_centroid_distance_over_time(
        embeddings, mode_ids, change_point_windows,
        title=f"Centroid Distance Over Time — {prefix}",
        save_path=output_dir / f"{p}centroid_distance.png",
    )
    plot_mode_distance_heatmap(
        embeddings, mode_ids,
        title=f"Mode Distance Heatmap — {prefix}",
        save_path=output_dir / f"{p}distance_heatmap.png",
    )
