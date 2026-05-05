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
from sklearn.manifold import TSNE

try:
    import umap as _umap_module
    _HAS_UMAP = True
except ImportError:
    _HAS_UMAP = False


# Consistent color palette for up to 6 modes
_MODE_COLORS = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00", "#a65628"]


def _fit_pca_2d(embeddings: np.ndarray) -> Tuple[np.ndarray, str]:
    """Returns (emb_2d, subtitle) where subtitle reports variance explained."""
    n_comp = min(2, embeddings.shape[1], embeddings.shape[0] - 1)
    pca = PCA(n_components=n_comp, random_state=0)
    emb_2d = pca.fit_transform(embeddings)
    var = pca.explained_variance_ratio_.sum()
    subtitle = f"PC1+PC2 explain {var:.1%} of variance"
    return emb_2d, subtitle


def _fit_umap_2d(embeddings: np.ndarray) -> Tuple[np.ndarray, str]:
    """Returns (emb_2d, method_label). Uses UMAP if available, t-SNE otherwise."""
    if _HAS_UMAP:
        reducer = _umap_module.UMAP(
            n_components=2,
            n_neighbors=15,
            min_dist=0.1,
            random_state=42,
            verbose=False,
        )
        emb_2d = reducer.fit_transform(embeddings)
        return emb_2d, "UMAP"
    else:
        perplexity = min(30, max(5, len(embeddings) // 4))
        reducer = TSNE(
            n_components=2,
            perplexity=perplexity,
            random_state=42,
            n_iter=500,
        )
        emb_2d = reducer.fit_transform(embeddings)
        return emb_2d, f"t-SNE (perplexity={perplexity})"


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
# Shared 2D rendering helpers (projection-agnostic)
# ---------------------------------------------------------------------------

def _render_worm(ax: plt.Axes, emb_2d: np.ndarray, mode_ids: np.ndarray,
                 xlabel: str, ylabel: str) -> None:
    N = len(emb_2d)
    points = emb_2d.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = LineCollection(segments, cmap="plasma", linewidth=1.2, alpha=0.8)
    lc.set_array(np.linspace(0, 1, N - 1))
    ax.add_collection(lc)
    for m in np.unique(mode_ids):
        mask = mode_ids == m
        ax.scatter(emb_2d[mask, 0], emb_2d[mask, 1],
                   color=_mode_color(m), s=12, alpha=0.5, label=f"Mode {m}", zorder=3)
    ax.scatter(*emb_2d[0], color="black", s=60, marker="o", zorder=5, label="Start")
    ax.scatter(*emb_2d[-1], color="black", s=60, marker="X", zorder=5, label="End")
    cbar = ax.get_figure().colorbar(cm.ScalarMappable(cmap="plasma"), ax=ax,
                                    fraction=0.04, pad=0.02)
    cbar.set_label("Time →")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.6)
    ax.set_aspect("equal", adjustable="datalim")


def _render_mode_loops(ax: plt.Axes, emb_2d: np.ndarray, mode_ids: np.ndarray,
                       change_point_windows: List[int], xlabel: str, ylabel: str) -> None:
    boundaries = sorted(set([0] + list(change_point_windows) + [len(emb_2d)]))
    seen_modes: set = set()
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        if end <= start:
            continue
        seg_labels = mode_ids[start:end]
        unique, counts = np.unique(seg_labels, return_counts=True)
        m = int(unique[counts.argmax()])
        seg_2d = emb_2d[start:end]
        label = f"Mode {m}" if m not in seen_modes else None
        ax.plot(seg_2d[:, 0], seg_2d[:, 1], color=_mode_color(m),
                alpha=0.4, linewidth=1.0, label=label)
        seen_modes.add(m)
    for m in np.unique(mode_ids):
        pts = emb_2d[mode_ids == m]
        ax.scatter(*pts.mean(axis=0), color=_mode_color(m), s=80,
                   marker="+", linewidths=2, zorder=5)
        _draw_confidence_ellipse(ax, pts, color=_mode_color(m))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.6)
    ax.set_aspect("equal", adjustable="datalim")


def _render_centroids(ax: plt.Axes, emb_2d: np.ndarray, mode_ids: np.ndarray,
                      xlabel: str, ylabel: str) -> None:
    for m in np.unique(mode_ids):
        pts = emb_2d[mode_ids == m]
        color = _mode_color(m)
        ax.scatter(pts[:, 0], pts[:, 1], color=color, s=8, alpha=0.25)
        ax.scatter(*pts.mean(axis=0), color=color, s=120, marker="*",
                   zorder=5, label=f"Mode {m}")
        _draw_confidence_ellipse(ax, pts, color=color, n_std=1.0)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.6)
    ax.set_aspect("equal", adjustable="datalim")


# ---------------------------------------------------------------------------
# Plot 1: PCA worm plot
# ---------------------------------------------------------------------------

def plot_worm(
    embeddings: np.ndarray,
    mode_ids: np.ndarray,
    title: str = "PCA Worm Plot",
    save_path: Optional[Path] = None,
) -> plt.Figure:
    emb_2d, subtitle = _fit_pca_2d(embeddings)
    fig, ax = plt.subplots(figsize=(7, 6))
    _render_worm(ax, emb_2d, mode_ids, "PC 1", "PC 2")
    ax.set_title(f"{title}\n{subtitle}", fontsize=10)
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
    return fig


# ---------------------------------------------------------------------------
# Plot 2: Mode loop overlay
# ---------------------------------------------------------------------------

def plot_mode_loops(
    embeddings: np.ndarray,
    mode_ids: np.ndarray,
    change_point_windows: List[int],
    title: str = "Mode Loop Overlay",
    save_path: Optional[Path] = None,
) -> plt.Figure:
    emb_2d, subtitle = _fit_pca_2d(embeddings)
    fig, ax = plt.subplots(figsize=(7, 6))
    _render_mode_loops(ax, emb_2d, mode_ids, change_point_windows, "PC 1", "PC 2")
    ax.set_title(f"{title}\n{subtitle}", fontsize=10)
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
    emb_2d, subtitle = _fit_pca_2d(embeddings)
    fig, ax = plt.subplots(figsize=(6, 5))
    _render_centroids(ax, emb_2d, mode_ids, "PC 1", "PC 2")
    ax.set_title(f"{title}\n{subtitle}", fontsize=10)
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
    return fig


# ---------------------------------------------------------------------------
# Plots 1–3 UMAP/t-SNE variants
# ---------------------------------------------------------------------------

def plot_umap_worm(
    embeddings: np.ndarray,
    mode_ids: np.ndarray,
    title: str = "UMAP Worm Plot",
    save_path: Optional[Path] = None,
) -> plt.Figure:
    emb_2d, method = _fit_umap_2d(embeddings)
    fig, ax = plt.subplots(figsize=(7, 6))
    _render_worm(ax, emb_2d, mode_ids, f"{method} 1", f"{method} 2")
    ax.set_title(f"{title}\n({method})", fontsize=10)
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
    return fig


def plot_umap_mode_loops(
    embeddings: np.ndarray,
    mode_ids: np.ndarray,
    change_point_windows: List[int],
    title: str = "Mode Loop Overlay",
    save_path: Optional[Path] = None,
) -> plt.Figure:
    emb_2d, method = _fit_umap_2d(embeddings)
    fig, ax = plt.subplots(figsize=(7, 6))
    _render_mode_loops(ax, emb_2d, mode_ids, change_point_windows,
                       f"{method} 1", f"{method} 2")
    ax.set_title(f"{title}\n({method})", fontsize=10)
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
    return fig


def plot_umap_centroids(
    embeddings: np.ndarray,
    mode_ids: np.ndarray,
    title: str = "Mode Centroids",
    save_path: Optional[Path] = None,
) -> plt.Figure:
    emb_2d, method = _fit_umap_2d(embeddings)
    fig, ax = plt.subplots(figsize=(6, 5))
    _render_centroids(ax, emb_2d, mode_ids, f"{method} 1", f"{method} 2")
    ax.set_title(f"{title}\n({method})", fontsize=10)
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
    """Save all plot types to output_dir with the given prefix.

    PCA plots (5): worm, mode_loops, centroids, centroid_distance, distance_heatmap
    UMAP/t-SNE plots (3): umap_worm, umap_mode_loops, umap_centroids
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    p = f"{prefix}_" if prefix else ""

    # PCA plots
    plot_worm(
        embeddings, mode_ids,
        title=f"PCA Worm — {prefix}",
        save_path=output_dir / f"{p}worm.png",
    )
    plot_mode_loops(
        embeddings, mode_ids, change_point_windows,
        title=f"Mode Loops (PCA) — {prefix}",
        save_path=output_dir / f"{p}mode_loops.png",
    )
    plot_centroids(
        embeddings, mode_ids,
        title=f"Centroids (PCA) — {prefix}",
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

    # UMAP / t-SNE plots
    plot_umap_worm(
        embeddings, mode_ids,
        title=f"UMAP Worm — {prefix}",
        save_path=output_dir / f"{p}umap_worm.png",
    )
    plot_umap_mode_loops(
        embeddings, mode_ids, change_point_windows,
        title=f"Mode Loops (UMAP) — {prefix}",
        save_path=output_dir / f"{p}umap_mode_loops.png",
    )
    plot_umap_centroids(
        embeddings, mode_ids,
        title=f"Centroids (UMAP) — {prefix}",
        save_path=output_dir / f"{p}umap_centroids.png",
    )
