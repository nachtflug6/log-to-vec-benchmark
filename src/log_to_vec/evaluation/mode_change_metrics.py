"""Metrics for baseline mode-change detection and segment clustering."""

from __future__ import annotations

from typing import Dict, List

import numpy as np


def mode_change_metrics(
    scores: np.ndarray,
    change_points: List[int],
    segment_labels: np.ndarray,
) -> Dict[str, float]:
    """Return compact summary metrics for unsupervised mode-change analysis."""
    if scores.ndim != 1:
        raise ValueError("scores must be 1D")

    metrics = {
        "mode_change/num_change_points": float(len(change_points)),
        "mode_change/mean_change_score": float(np.mean(scores)),
        "mode_change/max_change_score": float(np.max(scores)),
        "mode_change/score_std": float(np.std(scores)),
    }

    if segment_labels.size > 0:
        unique_labels = np.unique(segment_labels)
        metrics["mode_change/num_mode_clusters"] = float(len(unique_labels))
    else:
        metrics["mode_change/num_mode_clusters"] = 0.0

    return metrics
