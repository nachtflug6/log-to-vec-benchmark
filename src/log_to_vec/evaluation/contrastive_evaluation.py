"""
Evaluation for contrastive learning.

Metrics:
1) Positive Cosine Similarity: mean cosine(z1, z2)
2) Alignment: mean squared distance ||z1 - z2||^2
3) Uniformity (Wang & Isola paper): log mean exp(-t * ||zi - zj||^2)
4) Embedding Variance: mean variance across embedding dimensions (anti-collapse proxy)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional
import torch
import torch.nn.functional as F


@dataclass
class ContrastiveEvalConfig:
    """Configuration for contrastive evaluation metrics."""
    uniformity_t: float = 2.0
    uniformity_max_samples: int = 2048  # subsample size for pairwise computations
    eps: float = 1e-8


def _subsample_rows(x: torch.Tensor, max_samples: int) -> torch.Tensor:
    """Randomly subsample rows to reduce O(N^2) cost."""
    n = x.shape[0]
    if max_samples is None or max_samples <= 0 or n <= max_samples:
        return x
    idx = torch.randperm(n, device=x.device)[:max_samples]
    return x[idx]


def positive_cosine(z1: torch.Tensor, z2: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Mean cosine similarity between paired positives."""
    z1 = F.normalize(z1, dim=1, eps=eps)
    z2 = F.normalize(z2, dim=1, eps=eps)
    return (z1 * z2).sum(dim=1).mean()


def alignment(z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
    """Mean squared L2 distance between paired positives (lower is better)."""
    return (z1 - z2).pow(2).sum(dim=1).mean()


def uniformity(z: torch.Tensor,
               t: float = 2.0,
               max_samples: int = 2048,
               eps: float = 1e-8) -> torch.Tensor:
    """
    Uniformity metric: log E[exp(-t * ||zi - zj||^2)] over random pairs.

    Lower (more negative) is better: indicates embeddings are spread out.

    To avoid O(N^2) explosion, we subsample and compute full pairwise on the subsample.
    """
    z = _subsample_rows(z, max_samples=max_samples)
    z = F.normalize(z, dim=1, eps=eps)

    # Pairwise squared distances using (a-b)^2 = a^2 + b^2 - 2ab
    # For normalized vectors: ||a||^2 = ||b||^2 = 1, so dist^2 = 2 - 2*(a·b)
    sim = z @ z.T  # cosine similarity matrix
    dist2 = 2.0 - 2.0 * sim

    # Exclude diagonal pairs (i==j)
    n = dist2.shape[0]
    mask = ~torch.eye(n, device=z.device, dtype=torch.bool)
    dist2 = dist2[mask]

    # log mean exp(-t * dist2)
    return torch.log(torch.mean(torch.exp(-t * dist2)) + eps)


def embedding_variance(z: torch.Tensor) -> torch.Tensor:
    """
    Mean variance across embedding dimensions.
    Higher is generally better (anti-collapse proxy).
    """
    # variance over batch dimension
    return z.var(dim=0, unbiased=False).mean()


@torch.no_grad()
def compute_contrastive_metrics(
    z1: torch.Tensor,
    z2: torch.Tensor,
    embeddings_for_uniformity: Optional[torch.Tensor] = None,
    cfg: Optional[ContrastiveEvalConfig] = None
) -> Dict[str, float]:
    """
    Compute mode-agnostic contrastive evaluation metrics.

    Args:
        z1, z2: (B, D) projected embeddings for view1/view2 (paired)
        embeddings_for_uniformity: (N, D) embeddings to compute uniformity on.
                                  If None, will use concat([z1, z2]).
        cfg: configuration for uniformity and numerical stability

    Returns:
        Dict of scalar floats.
    """
    if cfg is None:
        cfg = ContrastiveEvalConfig()

    if z1.ndim != 2 or z2.ndim != 2:
        raise ValueError(f"z1 and z2 must be 2D, got {z1.shape}, {z2.shape}")
    if z1.shape != z2.shape:
        raise ValueError(f"z1 and z2 must have same shape, got {z1.shape} vs {z2.shape}")

    pos_cos = positive_cosine(z1, z2, eps=cfg.eps)
    align = alignment(z1, z2)

    if embeddings_for_uniformity is None:
        Z = torch.cat([z1, z2], dim=0)
    else:
        Z = embeddings_for_uniformity

    uni = uniformity(Z, t=cfg.uniformity_t, max_samples=cfg.uniformity_max_samples, eps=cfg.eps)
    var = embedding_variance(Z)

    return {
        "positive_cosine": float(pos_cos.item()),
        "alignment": float(align.item()),
        "uniformity": float(uni.item()),
        "embedding_variance": float(var.item()),
    }