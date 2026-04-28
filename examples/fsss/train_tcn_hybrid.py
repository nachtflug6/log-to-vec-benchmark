from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from version2.data.fsss_dataset import FSSSWindowDataset
from version2.models.tcn_hybrid import TCNHybridEncoder
from version2.training.hybrid_losses import total_hybrid_unsup_loss


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# -----------------------------------------------------------------------------
# Conservative augmentations
# -----------------------------------------------------------------------------

def weak_augment(
    x: torch.Tensor,
    noise_std: float = 0.01,
    max_time_shift: int = 1,
) -> torch.Tensor:
    out = x.clone()
    out = out + torch.randn_like(out) * noise_std

    if max_time_shift > 0:
        shifts = torch.randint(
            low=-max_time_shift,
            high=max_time_shift + 1,
            size=(out.shape[0],),
            device=out.device,
        )
        out = torch.stack(
            [torch.roll(out[i], shifts=int(shifts[i].item()), dims=0) for i in range(out.shape[0])],
            dim=0,
        )

    return out


def build_span_mask(
    x: torch.Tensor,
    mask_ratio: float = 0.15,
    span_len: int = 6,
) -> torch.Tensor:
    """Continuous temporal span masking for reconstruction."""
    B, T, D = x.shape
    mask = torch.zeros_like(x)

    num_mask_steps = max(1, int(T * mask_ratio))
    num_spans = max(1, num_mask_steps // max(1, span_len))

    for b in range(B):
        for _ in range(num_spans):
            start = int(torch.randint(0, max(1, T - span_len + 1), (1,), device=x.device).item())
            mask[b, start:start + span_len, :] = 1.0

    return mask


def apply_mask_token(x: torch.Tensor, mask: torch.Tensor, mask_value: float = 0.0) -> torch.Tensor:
    return x * (1.0 - mask) + mask_value * mask


# -----------------------------------------------------------------------------
# Temporal positive mining without labels
# -----------------------------------------------------------------------------

def build_temporal_neighbor_index(dataset: FSSSWindowDataset) -> Dict[tuple[int, int], int]:
    mapping: Dict[tuple[int, int], int] = {}
    for idx in range(len(dataset)):
        mapping[(int(dataset.trajectory_id[idx]), int(dataset.window_start[idx]))] = idx
    return mapping


def choose_temporal_neighbor_indices(
    batch: Dict[str, torch.Tensor | List[int]],
    dataset: FSSSWindowDataset,
    index_map: Dict[tuple[int, int], int],
    window_stride: int,
    max_neighbor_hops: int,
) -> List[int]:
    """
    Choose positives from the same trajectory using only trajectory structure.
    """
    batch_indices = batch["index"]
    if isinstance(batch_indices, torch.Tensor):
        batch_indices = batch_indices.tolist()

    result: List[int] = []
    for idx in batch_indices:
        traj = int(dataset.trajectory_id[idx])
        start = int(dataset.window_start[idx])

        candidates: List[int] = []
        for hop in range(1, max_neighbor_hops + 1):
            for delta in (-hop * window_stride, hop * window_stride):
                j = index_map.get((traj, start + delta))
                if j is not None:
                    candidates.append(j)

        if len(candidates) == 0:
            result.append(idx)
        else:
            result.append(int(np.random.choice(candidates)))

    return result


def gather_neighbor_windows(
    dataset: FSSSWindowDataset,
    neighbor_indices: List[int],
    device: torch.device,
) -> torch.Tensor:
    arr = dataset.X[neighbor_indices]
    return torch.from_numpy(arr).to(device=device, dtype=torch.float32)


# -----------------------------------------------------------------------------
# Unsupervised validation helpers
# -----------------------------------------------------------------------------

def compute_batch_neighbor_gap(anchor_emb: torch.Tensor, neighbor_emb: torch.Tensor) -> float:
    anchor = F.normalize(anchor_emb, dim=1)
    neighbor = F.normalize(neighbor_emb, dim=1)
    pos = (anchor * neighbor).sum(dim=1).mean().item()

    perm = torch.randperm(anchor.shape[0], device=anchor.device)
    neg = (anchor * neighbor[perm]).sum(dim=1).mean().item()
    return float(pos - neg)


def compute_embedding_health(embeddings: List[torch.Tensor]) -> Dict[str, float]:
    if len(embeddings) == 0:
        return {"overall_std": 0.0, "mean_dim_std": 0.0, "frac_low_variance_dims": 1.0}

    z = torch.cat(embeddings, dim=0)
    dim_std = z.std(dim=0, unbiased=False)
    overall_std = z.std(unbiased=False).item()
    frac_low_variance_dims = (dim_std < 1e-3).float().mean().item()
    return {
        "overall_std": float(overall_std),
        "mean_dim_std": float(dim_std.mean().item()),
        "frac_low_variance_dims": float(frac_low_variance_dims),
    }


def compute_validation_selection_score(val_stats: Dict[str, float], emb_health: Dict[str, float], neighbor_gap: float) -> float:
    # Higher is better.
    return (
        -1.0 * val_stats["loss_total"]
        + 0.50 * neighbor_gap
        + 0.20 * emb_health["mean_dim_std"]
        - 2.0 * emb_health["frac_low_variance_dims"]
    )


# -----------------------------------------------------------------------------
# Train / eval loop
# -----------------------------------------------------------------------------

def run_epoch(
    model: TCNHybridEncoder,
    dataloader: DataLoader,
    dataset: FSSSWindowDataset,
    neighbor_index: Dict[tuple[int, int], int],
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    train: bool,
    temperature: float,
    contrastive_weight: float,
    reconstruction_weight: float,
    temporal_weight: float,
    variance_weight: float,
    covariance_weight: float,
    mask_ratio: float,
    span_len: int,
    positive_mode: str,
    window_stride: int,
    max_neighbor_hops: int,
) -> tuple[Dict[str, float], Dict[str, float], float]:
    if train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_contrastive = 0.0
    total_reconstruction = 0.0
    total_temporal = 0.0
    total_variance = 0.0
    total_covariance = 0.0
    total_neighbor_gap = 0.0
    num_batches = 0
    collected_embeddings: List[torch.Tensor] = []

    context = torch.enable_grad() if train else torch.no_grad()

    with context:
        for batch in tqdm(dataloader, leave=False):
            x = batch["x"].to(device, non_blocking=True)

            x1 = weak_augment(x)
            if positive_mode == "instance":
                x2 = weak_augment(x)
                neighbor_for_temporal = x2
            else:
                neighbor_indices = choose_temporal_neighbor_indices(
                    batch=batch,
                    dataset=dataset,
                    index_map=neighbor_index,
                    window_stride=window_stride,
                    max_neighbor_hops=max_neighbor_hops,
                )
                x2 = gather_neighbor_windows(dataset, neighbor_indices, device)
                x2 = weak_augment(x2)
                neighbor_for_temporal = x2

            recon_mask = build_span_mask(x, mask_ratio=mask_ratio, span_len=span_len)
            x_masked = apply_mask_token(x, recon_mask)

            out1 = model(x1)
            out2 = model(x2)
            out_recon = model(x_masked)
            out_anchor = model(x)
            out_neighbor = model(neighbor_for_temporal)

            loss, stats = total_hybrid_unsup_loss(
                out1=out1,
                out2=out2,
                out_recon=out_recon,
                x_target=x,
                recon_mask=recon_mask,
                anchor_emb=out_anchor["embeddings"],
                neighbor_emb=out_neighbor["embeddings"],
                contrastive_weight=contrastive_weight,
                reconstruction_weight=reconstruction_weight,
                temporal_weight=temporal_weight,
                variance_weight=variance_weight,
                covariance_weight=covariance_weight,
                temperature=temperature,
            )

            if train and optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            total_loss += stats["loss_total"]
            total_contrastive += stats["loss_contrastive"]
            total_reconstruction += stats["loss_reconstruction"]
            total_temporal += stats["loss_temporal"]
            total_variance += stats["loss_variance"]
            total_covariance += stats["loss_covariance"]
            total_neighbor_gap += compute_batch_neighbor_gap(out_anchor["embeddings"], out_neighbor["embeddings"])
            collected_embeddings.append(out_anchor["embeddings"].detach().cpu())
            num_batches += 1

    if num_batches == 0:
        stats = {
            "loss_total": 0.0,
            "loss_contrastive": 0.0,
            "loss_reconstruction": 0.0,
            "loss_temporal": 0.0,
            "loss_variance": 0.0,
            "loss_covariance": 0.0,
        }
        emb_health = {"overall_std": 0.0, "mean_dim_std": 0.0, "frac_low_variance_dims": 1.0}
        return stats, emb_health, 0.0

    stats = {
        "loss_total": total_loss / num_batches,
        "loss_contrastive": total_contrastive / num_batches,
        "loss_reconstruction": total_reconstruction / num_batches,
        "loss_temporal": total_temporal / num_batches,
        "loss_variance": total_variance / num_batches,
        "loss_covariance": total_covariance / num_batches,
    }
    emb_health = compute_embedding_health(collected_embeddings)
    avg_neighbor_gap = total_neighbor_gap / num_batches
    return stats, emb_health, avg_neighbor_gap


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train unsupervised TCN hybrid encoder on FSSS splits.")
    parser.add_argument("--train_file", type=str, required=True)
    parser.add_argument("--val_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)

    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--embedding_dim", type=int, default=128)
    parser.add_argument("--projection_dim", type=int, default=64)
    parser.add_argument("--num_blocks", type=int, default=4)
    parser.add_argument("--kernel_size", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.1)

    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--contrastive_weight", type=float, default=1.0)
    parser.add_argument("--reconstruction_weight", type=float, default=1.0)
    parser.add_argument("--temporal_weight", type=float, default=0.25)
    parser.add_argument("--variance_weight", type=float, default=0.10)
    parser.add_argument("--covariance_weight", type=float, default=0.02)

    parser.add_argument("--mask_ratio", type=float, default=0.15)
    parser.add_argument("--span_len", type=int, default=6)

    parser.add_argument(
        "--positive_mode",
        type=str,
        default="temporal_neighbor",
        choices=["instance", "temporal_neighbor"],
    )
    parser.add_argument("--window_stride", type=int, default=12)
    parser.add_argument("--max_neighbor_hops", type=int, default=2)

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_ds = FSSSWindowDataset(args.train_file)
    val_ds = FSSSWindowDataset(args.val_file)
    train_neighbor_index = build_temporal_neighbor_index(train_ds)
    val_neighbor_index = build_temporal_neighbor_index(val_ds)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, drop_last=False)

    input_dim = train_ds.X.shape[2]
    model = TCNHybridEncoder(
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        embedding_dim=args.embedding_dim,
        projection_dim=args.projection_dim,
        num_blocks=args.num_blocks,
        kernel_size=args.kernel_size,
        dropout=args.dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_score = -float("inf")
    history = []

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")

        train_stats, train_emb_health, train_neighbor_gap = run_epoch(
            model=model,
            dataloader=train_loader,
            dataset=train_ds,
            neighbor_index=train_neighbor_index,
            optimizer=optimizer,
            device=device,
            train=True,
            temperature=args.temperature,
            contrastive_weight=args.contrastive_weight,
            reconstruction_weight=args.reconstruction_weight,
            temporal_weight=args.temporal_weight,
            variance_weight=args.variance_weight,
            covariance_weight=args.covariance_weight,
            mask_ratio=args.mask_ratio,
            span_len=args.span_len,
            positive_mode=args.positive_mode,
            window_stride=args.window_stride,
            max_neighbor_hops=args.max_neighbor_hops,
        )

        val_stats, val_emb_health, val_neighbor_gap = run_epoch(
            model=model,
            dataloader=val_loader,
            dataset=val_ds,
            neighbor_index=val_neighbor_index,
            optimizer=None,
            device=device,
            train=False,
            temperature=args.temperature,
            contrastive_weight=args.contrastive_weight,
            reconstruction_weight=args.reconstruction_weight,
            temporal_weight=args.temporal_weight,
            variance_weight=args.variance_weight,
            covariance_weight=args.covariance_weight,
            mask_ratio=args.mask_ratio,
            span_len=args.span_len,
            positive_mode=args.positive_mode,
            window_stride=args.window_stride,
            max_neighbor_hops=args.max_neighbor_hops,
        )

        selection_score = compute_validation_selection_score(
            val_stats=val_stats,
            emb_health=val_emb_health,
            neighbor_gap=val_neighbor_gap,
        )

        print(
            f"train_total={train_stats['loss_total']:.4f} "
            f"train_c={train_stats['loss_contrastive']:.4f} "
            f"train_r={train_stats['loss_reconstruction']:.4f} "
            f"train_t={train_stats['loss_temporal']:.4f}"
        )
        print(
            f"val_total={val_stats['loss_total']:.4f} "
            f"val_c={val_stats['loss_contrastive']:.4f} "
            f"val_r={val_stats['loss_reconstruction']:.4f} "
            f"val_t={val_stats['loss_temporal']:.4f}"
        )
        print(
            f"val_neighbor_gap={val_neighbor_gap:.4f} "
            f"val_mean_dim_std={val_emb_health['mean_dim_std']:.4f} "
            f"val_frac_low_var={val_emb_health['frac_low_variance_dims']:.4f} "
            f"selection_score={selection_score:.4f}"
        )

        history.append(
            {
                "epoch": epoch + 1,
                "train": train_stats,
                "train_embedding_health": train_emb_health,
                "train_neighbor_gap": train_neighbor_gap,
                "val": val_stats,
                "val_embedding_health": val_emb_health,
                "val_neighbor_gap": val_neighbor_gap,
                "selection_score": selection_score,
            }
        )

        ckpt = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "selection_score": selection_score,
            "val_loss": val_stats["loss_total"],
            "input_dim": input_dim,
            "hidden_dim": args.hidden_dim,
            "embedding_dim": args.embedding_dim,
            "projection_dim": args.projection_dim,
            "num_blocks": args.num_blocks,
            "kernel_size": args.kernel_size,
            "dropout": args.dropout,
        }

        torch.save(ckpt, output_dir / "last_model.pt")

        if selection_score > best_score:
            best_score = selection_score
            torch.save(ckpt, output_dir / "best_model.pt")
            print(f"Saved best checkpoint -> {output_dir / 'best_model.pt'}")

    with open(output_dir / "training_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    print(f"\nDone. Best selection score: {best_score:.4f}")


if __name__ == "__main__":
    main()
