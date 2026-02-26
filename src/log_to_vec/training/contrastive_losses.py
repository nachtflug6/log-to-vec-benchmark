"""
Contrastive loss functions for representation learning.

Implements NT-Xent (Normalized Temperature-scaled Cross Entropy),
a common InfoNCE-style loss used in SimCLR-like contrastive learning.

Given two batches of projected embeddings z1 and z2 (paired views),
the loss encourages each z1[i] to be closest to z2[i] among all negatives
in the batch (in-batch negatives).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def nt_xent_loss(
    z1: torch.Tensor,
    z2: torch.Tensor,
    temperature: float = 0.1,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Compute the NT-Xent loss using in-batch negatives.

    This implementation uses the standard 2B formulation:
      - Concatenate z1 and z2 into a single batch Z of size 2B
      - Each sample's positive is its paired view (i <-> i+B)
      - All other samples are treated as negatives

    Args:
        z1: Tensor of shape (B, D) - projected embeddings for view 1
        z2: Tensor of shape (B, D) - projected embeddings for view 2
        temperature: Temperature scaling factor (tau)
        eps: Small constant for numerical stability

    Returns:
        Scalar loss (mean over 2B anchors)
    """
    if z1.ndim != 2 or z2.ndim != 2:
        raise ValueError(f"z1 and z2 must be 2D tensors, got {z1.shape}, {z2.shape}")
    if z1.shape != z2.shape:
        raise ValueError(f"z1 and z2 must have the same shape, got {z1.shape} vs {z2.shape}")
    if temperature <= 0:
        raise ValueError("temperature must be > 0")

    B, D = z1.shape
    if B < 2:
        raise ValueError("Batch size must be >= 2 to use in-batch negatives effectively.")

    # Normalize embeddings to unit length (cosine similarity via dot product)
    z1 = F.normalize(z1, dim=1, eps=eps)
    z2 = F.normalize(z2, dim=1, eps=eps)

    # Concatenate: Z has shape (2B, D)
    Z = torch.cat([z1, z2], dim=0)

    # Similarity matrix: (2B, 2B)
    # sim[i, j] = cosine(Z[i], Z[j]) / temperature
    sim = torch.matmul(Z, Z.T) / temperature

    # Mask out self-similarity (i == j) so it won't be selected as a positive/negative
    # Set diagonal to a large negative value so exp(sim) ~ 0
    diag_mask = torch.eye(2 * B, device=Z.device, dtype=torch.bool)
    sim = sim.masked_fill(diag_mask, -1e9)

    # Positive indices:
    # For i in [0, B-1] (z1), positive is i+B (z2)
    # For i in [B, 2B-1] (z2), positive is i-B (z1)
    pos_idx = torch.arange(2 * B, device=Z.device)
    pos_idx = (pos_idx + B) % (2 * B)

    # Cross-entropy over the row-wise logits:
    # Each row i: logits = sim[i, :]
    # Target class is pos_idx[i]
    loss = F.cross_entropy(sim, pos_idx)

    return loss