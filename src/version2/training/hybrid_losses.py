from __future__ import annotations

import torch
import torch.nn.functional as F


def nt_xent_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.2) -> torch.Tensor:
    if z1.shape != z2.shape:
        raise ValueError(f"Shape mismatch: z1={z1.shape}, z2={z2.shape}")

    batch_size = z1.shape[0]
    z = torch.cat([z1, z2], dim=0)
    sim = torch.matmul(z, z.T) / temperature
    eye = torch.eye(2 * batch_size, device=z.device, dtype=torch.bool)
    sim = sim.masked_fill(eye, -1e9)

    targets = torch.arange(batch_size, device=z.device)
    targets = torch.cat([targets + batch_size, targets], dim=0)
    return F.cross_entropy(sim, targets)


def masked_reconstruction_loss(
    recon: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    mask = mask.float()
    sq_err = (recon - target) ** 2
    masked_sq_err = sq_err * mask
    denom = mask.sum().clamp_min(1.0)
    return masked_sq_err.sum() / denom


def variance_loss(emb: torch.Tensor, target_std: float = 1.0, eps: float = 1e-4) -> torch.Tensor:
    std = torch.sqrt(emb.var(dim=0, unbiased=False) + eps)
    return torch.mean(F.relu(target_std - std))


def covariance_loss(emb: torch.Tensor) -> torch.Tensor:
    n, d = emb.shape
    if n <= 1:
        return emb.new_tensor(0.0)
    x = emb - emb.mean(dim=0, keepdim=True)
    cov = (x.T @ x) / (n - 1)
    off_diag = cov - torch.diag(torch.diag(cov))
    return (off_diag ** 2).sum() / d


def temporal_consistency_loss(anchor_emb: torch.Tensor, neighbor_emb: torch.Tensor) -> torch.Tensor:
    anchor = F.normalize(anchor_emb, dim=1)
    neighbor = F.normalize(neighbor_emb, dim=1)
    cosine = (anchor * neighbor).sum(dim=1)
    return 1.0 - cosine.mean()


def total_hybrid_unsup_loss(
    out1: dict,
    out2: dict,
    out_recon: dict,
    x_target: torch.Tensor,
    recon_mask: torch.Tensor,
    anchor_emb: torch.Tensor,
    neighbor_emb: torch.Tensor,
    contrastive_weight: float = 1.0,
    reconstruction_weight: float = 1.0,
    temporal_weight: float = 0.25,
    variance_weight: float = 0.10,
    covariance_weight: float = 0.02,
    temperature: float = 0.2,
) -> tuple[torch.Tensor, dict]:
    loss_contrastive = nt_xent_loss(out1["projections"], out2["projections"], temperature=temperature)
    loss_reconstruction = masked_reconstruction_loss(
        recon=out_recon["reconstructions"],
        target=x_target,
        mask=recon_mask,
    )
    loss_temporal = temporal_consistency_loss(anchor_emb, neighbor_emb)

    emb_cat = torch.cat([out1["embeddings"], out2["embeddings"]], dim=0)
    loss_variance = variance_loss(emb_cat)
    loss_covariance = covariance_loss(emb_cat)

    total = (
        contrastive_weight * loss_contrastive
        + reconstruction_weight * loss_reconstruction
        + temporal_weight * loss_temporal
        + variance_weight * loss_variance
        + covariance_weight * loss_covariance
    )

    stats = {
        "loss_total": float(total.detach().item()),
        "loss_contrastive": float(loss_contrastive.detach().item()),
        "loss_reconstruction": float(loss_reconstruction.detach().item()),
        "loss_temporal": float(loss_temporal.detach().item()),
        "loss_variance": float(loss_variance.detach().item()),
        "loss_covariance": float(loss_covariance.detach().item()),
    }
    return total, stats
