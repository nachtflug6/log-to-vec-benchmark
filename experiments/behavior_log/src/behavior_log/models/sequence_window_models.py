"""Learned sequence models for behavior-log window embeddings."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset


class ResidualTemporalBlock(nn.Module):
    """Simple residual 1D convolution block with dilation."""

    def __init__(self, channels: int, kernel_size: int, dilation: int, dropout: float) -> None:
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation)
        self.norm1 = nn.BatchNorm1d(channels)
        self.norm2 = nn.BatchNorm1d(channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.dropout(F.relu(self.norm1(self.conv1(x))))
        x = self.dropout(self.norm2(self.conv2(x)))
        return F.relu(x + residual)


class TCNEncoder(nn.Module):
    """Small TCN-style encoder that preserves sequence length."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        depth: int,
        kernel_size: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Conv1d(input_dim, hidden_dim, kernel_size=1)
        self.blocks = nn.ModuleList(
            [
                ResidualTemporalBlock(
                    channels=hidden_dim,
                    kernel_size=kernel_size,
                    dilation=2**layer_idx,
                    dropout=dropout,
                )
                for layer_idx in range(depth)
            ]
        )
        self.output_proj = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x)
        x = self.output_proj(x)
        return x.transpose(1, 2)


class CategoricalNumericInputEncoder(nn.Module):
    """Encode categorical ids and numeric features into per-event vectors."""

    def __init__(
        self,
        event_type_vocab_size: int,
        component_vocab_size: int,
        severity_vocab_size: int,
        state_vocab_size: int,
        event_type_emb_dim: int,
        component_emb_dim: int,
        severity_emb_dim: int,
        state_emb_dim: int,
        numeric_input_dim: int,
        output_dim: int,
    ) -> None:
        super().__init__()
        self.event_type_vocab_size = event_type_vocab_size
        self.component_vocab_size = component_vocab_size
        self.severity_vocab_size = severity_vocab_size
        self.state_vocab_size = state_vocab_size

        self.event_type_embedding = nn.Embedding(event_type_vocab_size + 1, event_type_emb_dim)
        self.component_embedding = nn.Embedding(component_vocab_size + 1, component_emb_dim)
        self.severity_embedding = nn.Embedding(severity_vocab_size + 1, severity_emb_dim)
        self.state_embedding = nn.Embedding(state_vocab_size + 1, state_emb_dim)

        fused_dim = event_type_emb_dim + component_emb_dim + severity_emb_dim + state_emb_dim + numeric_input_dim
        self.output_proj = nn.Sequential(
            nn.Linear(fused_dim, output_dim),
            nn.ReLU(),
        )

    def forward(self, X: torch.Tensor, masked_event_positions: torch.Tensor | None = None) -> torch.Tensor:
        event_type_ids = X[:, :, 0].long()
        component_ids = X[:, :, 1].long()
        severity_ids = X[:, :, 2].long()
        state_ids = X[:, :, 3].long()
        numeric_values = X[:, :, 4:].float()

        if masked_event_positions is not None:
            event_type_ids = torch.where(masked_event_positions, torch.full_like(event_type_ids, self.event_type_vocab_size), event_type_ids)
            component_ids = torch.where(masked_event_positions, torch.full_like(component_ids, self.component_vocab_size), component_ids)
            severity_ids = torch.where(masked_event_positions, torch.full_like(severity_ids, self.severity_vocab_size), severity_ids)
            state_ids = torch.where(masked_event_positions, torch.full_like(state_ids, self.state_vocab_size), state_ids)
            numeric_values = torch.where(masked_event_positions.unsqueeze(-1), torch.zeros_like(numeric_values), numeric_values)

        encoded = torch.cat(
            [
                self.event_type_embedding(event_type_ids),
                self.component_embedding(component_ids),
                self.severity_embedding(severity_ids),
                self.state_embedding(state_ids),
                numeric_values,
            ],
            dim=-1,
        )
        return self.output_proj(encoded)


def masked_mean_pool(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Pool sequence states using the valid-event mask."""

    mask = mask.unsqueeze(-1)
    summed = torch.sum(hidden * mask, dim=1)
    denom = torch.clamp(torch.sum(mask, dim=1), min=1.0)
    return summed / denom


def masked_max_pool(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Pool sequence states with a masked max over valid events."""

    expanded_mask = mask.unsqueeze(-1).bool()
    filled = hidden.masked_fill(~expanded_mask, float("-inf"))
    pooled = torch.max(filled, dim=1).values
    return torch.where(torch.isfinite(pooled), pooled, torch.zeros_like(pooled))


class AttentionPool(nn.Module):
    """Learn attention weights over valid timesteps."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.score = nn.Linear(hidden_dim, 1)

    def forward(self, hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        logits = self.score(hidden).squeeze(-1)
        logits = logits.masked_fill(mask <= 0, float("-inf"))
        attn = torch.softmax(logits, dim=1)
        attn = torch.where(mask > 0, attn, torch.zeros_like(attn))
        return torch.sum(hidden * attn.unsqueeze(-1), dim=1)


def pool_sequence(
    hidden: torch.Tensor,
    mask: torch.Tensor,
    pooling_mode: str,
    attention_pool: AttentionPool | None = None,
) -> torch.Tensor:
    """Pool temporal states into one window embedding."""

    if pooling_mode == "mean":
        return masked_mean_pool(hidden, mask)
    if pooling_mode == "max":
        return masked_max_pool(hidden, mask)
    if pooling_mode == "mean_max":
        return torch.cat([masked_mean_pool(hidden, mask), masked_max_pool(hidden, mask)], dim=1)
    if pooling_mode == "attention":
        if attention_pool is None:
            raise ValueError("attention_pool must be provided for attention pooling.")
        return attention_pool(hidden, mask)
    raise ValueError(f"Unknown pooling_mode: {pooling_mode}")


def multi_positive_contrastive_loss(
    z1: torch.Tensor,
    z2: torch.Tensor,
    temperature: float,
    temporal_positive_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute a contrastive loss with same-window and temporal-neighbor positives."""

    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    reps = torch.cat([z1, z2], dim=0)
    batch_size = z1.size(0)
    total = batch_size * 2

    positive_mask = torch.zeros((total, total), dtype=torch.bool, device=reps.device)
    pair_indices = torch.arange(batch_size, device=reps.device)
    positive_mask[pair_indices, pair_indices + batch_size] = True
    positive_mask[pair_indices + batch_size, pair_indices] = True

    if temporal_positive_mask is not None and temporal_positive_mask.numel() > 0:
        temporal_positive_mask = temporal_positive_mask.to(reps.device)
        positive_mask[:batch_size, :batch_size] |= temporal_positive_mask
        positive_mask[batch_size:, batch_size:] |= temporal_positive_mask
        positive_mask[:batch_size, batch_size:] |= temporal_positive_mask
        positive_mask[batch_size:, :batch_size] |= temporal_positive_mask

    identity = torch.eye(total, device=reps.device, dtype=torch.bool)
    positive_mask = positive_mask & ~identity

    logits = torch.matmul(reps, reps.transpose(0, 1)) / temperature
    logits = logits.masked_fill(identity, -1e9)
    log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)

    positive_counts = positive_mask.sum(dim=1)
    valid = positive_counts > 0
    per_anchor_loss = -(positive_mask.float() * log_prob).sum(dim=1) / torch.clamp(positive_counts.float(), min=1.0)
    return per_anchor_loss[valid].mean()


def conservative_hard_negative_loss(
    z1: torch.Tensor,
    z2: torch.Tensor,
    trajectory_id: torch.Tensor,
    window_start: torch.Tensor,
    temporal_positive_mask: torch.Tensor | None,
    far_negative_radius: int,
    similarity_margin: float,
) -> torch.Tensor:
    """Push down only conservative hard negatives: different trajectory or far-apart same trajectory."""

    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    reps = torch.cat([z1, z2], dim=0)
    batch_size = z1.size(0)
    total = batch_size * 2

    repeated_trajectory = torch.cat([trajectory_id, trajectory_id], dim=0)
    repeated_start = torch.cat([window_start, window_start], dim=0)
    same_trajectory = repeated_trajectory[:, None] == repeated_trajectory[None, :]
    start_distance = torch.abs(repeated_start[:, None] - repeated_start[None, :])
    conservative_negative_mask = (~same_trajectory) | (start_distance >= far_negative_radius)

    identity = torch.eye(total, device=reps.device, dtype=torch.bool)
    conservative_negative_mask = conservative_negative_mask & ~identity

    positive_mask = torch.zeros((total, total), dtype=torch.bool, device=reps.device)
    pair_indices = torch.arange(batch_size, device=reps.device)
    positive_mask[pair_indices, pair_indices + batch_size] = True
    positive_mask[pair_indices + batch_size, pair_indices] = True
    if temporal_positive_mask is not None and temporal_positive_mask.numel() > 0:
        temporal_positive_mask = temporal_positive_mask.to(reps.device)
        positive_mask[:batch_size, :batch_size] |= temporal_positive_mask
        positive_mask[batch_size:, batch_size:] |= temporal_positive_mask
        positive_mask[:batch_size, batch_size:] |= temporal_positive_mask
        positive_mask[batch_size:, :batch_size] |= temporal_positive_mask
    conservative_negative_mask = conservative_negative_mask & ~positive_mask

    similarity = torch.matmul(reps, reps.transpose(0, 1))
    masked_similarity = similarity.masked_fill(~conservative_negative_mask, float("-inf"))
    hardest_negative = torch.max(masked_similarity, dim=1).values
    valid = torch.isfinite(hardest_negative)
    if not torch.any(valid):
        return torch.tensor(0.0, device=reps.device)
    return torch.relu(hardest_negative[valid] - similarity_margin).mean()


def sample_mask_span_positions(
    valid_positions: torch.Tensor,
    masked_event_prob: float,
    mask_span_length: int,
) -> torch.Tensor:
    """Sample masked positions with contiguous spans inside valid timesteps."""

    if masked_event_prob <= 0:
        return torch.zeros_like(valid_positions, dtype=torch.bool)

    batch_size, seq_len = valid_positions.shape
    masked = torch.zeros_like(valid_positions, dtype=torch.bool)
    mask_span_length = max(1, int(mask_span_length))

    for row_idx in range(batch_size):
        valid_idx = torch.where(valid_positions[row_idx])[0]
        if len(valid_idx) == 0:
            continue
        target_count = max(1, int(torch.ceil(torch.tensor(len(valid_idx) * masked_event_prob)).item()))
        attempts = 0
        while int(masked[row_idx].sum().item()) < target_count and attempts < seq_len * 4:
            start = int(valid_idx[torch.randint(len(valid_idx), (1,), device=valid_positions.device)].item())
            end = min(seq_len, start + mask_span_length)
            span_positions = torch.arange(start, end, device=valid_positions.device)
            span_positions = span_positions[valid_positions[row_idx, span_positions]]
            masked[row_idx, span_positions] = True
            attempts += 1
    return masked & valid_positions


def apply_local_order_shuffle(
    encoded: torch.Tensor,
    valid_positions: torch.Tensor,
    span_length: int,
) -> torch.Tensor:
    """Reverse one local span inside each sequence to create an order-corrupted view."""

    shuffled = encoded.clone()
    batch_size, seq_len, _ = encoded.shape
    span_length = max(2, int(span_length))

    for row_idx in range(batch_size):
        valid_idx = torch.where(valid_positions[row_idx])[0]
        if len(valid_idx) < span_length:
            continue
        max_start = int(valid_idx[-span_length].item())
        min_start = int(valid_idx[0].item())
        if max_start < min_start:
            continue
        start = int(torch.randint(min_start, max_start + 1, (1,), device=encoded.device).item())
        span_positions = torch.arange(start, min(seq_len, start + span_length), device=encoded.device)
        span_positions = span_positions[valid_positions[row_idx, span_positions]]
        if len(span_positions) < 2:
            continue
        shuffled[row_idx, span_positions] = encoded[row_idx, span_positions.flip(0)]
    return shuffled


@dataclass
class LearnedModelResult:
    """Saved training summary."""

    summary: dict


class TS2VecStyleWindowEncoder:
    """Contrastive log-window encoder with a TCN backbone."""

    model_type = "ts2vec_style"

    def __init__(
        self,
        input_dim: int,
        window_length: int,
        hidden_dim: int = 64,
        depth: int = 4,
        kernel_size: int = 3,
        dropout: float = 0.1,
        event_type_vocab_size: int = 8,
        component_vocab_size: int = 4,
        severity_vocab_size: int = 4,
        state_vocab_size: int = 4,
        event_type_emb_dim: int = 16,
        component_emb_dim: int = 8,
        severity_emb_dim: int = 8,
        state_emb_dim: int = 8,
        projection_dim: int = 32,
        batch_size: int = 64,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        max_epochs: int = 20,
        temperature: float = 0.2,
        time_mask_prob: float = 0.15,
        feature_dropout_prob: float = 0.1,
        noise_std: float = 0.02,
        temporal_positive_radius: int = 10,
        pooling_mode: str = "mean",
        masked_event_prob: float = 0.15,
        masked_prediction_weight: float = 1.0,
        mask_span_length: int = 1,
        order_prediction_weight: float = 0.0,
        order_span_length: int = 3,
        hard_negative_weight: float = 0.0,
        hard_negative_margin: float = 0.3,
        hard_negative_far_radius: int = 20,
        device: str = "cpu",
    ) -> None:
        self.input_dim = input_dim
        self.window_length = window_length
        self.hidden_dim = hidden_dim
        self.depth = depth
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.event_type_vocab_size = event_type_vocab_size
        self.component_vocab_size = component_vocab_size
        self.severity_vocab_size = severity_vocab_size
        self.state_vocab_size = state_vocab_size
        self.event_type_emb_dim = event_type_emb_dim
        self.component_emb_dim = component_emb_dim
        self.severity_emb_dim = severity_emb_dim
        self.state_emb_dim = state_emb_dim
        self.projection_dim = projection_dim
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs
        self.temperature = temperature
        self.time_mask_prob = time_mask_prob
        self.feature_dropout_prob = feature_dropout_prob
        self.noise_std = noise_std
        self.temporal_positive_radius = temporal_positive_radius
        self.pooling_mode = pooling_mode
        self.masked_event_prob = masked_event_prob
        self.masked_prediction_weight = masked_prediction_weight
        self.mask_span_length = mask_span_length
        self.order_prediction_weight = order_prediction_weight
        self.order_span_length = order_span_length
        self.hard_negative_weight = hard_negative_weight
        self.hard_negative_margin = hard_negative_margin
        self.hard_negative_far_radius = hard_negative_far_radius
        self.device = self._resolve_device(device)
        self.pooled_dim = hidden_dim if pooling_mode in {"mean", "max", "attention"} else hidden_dim * 2

        self.input_encoder = CategoricalNumericInputEncoder(
            event_type_vocab_size=event_type_vocab_size,
            component_vocab_size=component_vocab_size,
            severity_vocab_size=severity_vocab_size,
            state_vocab_size=state_vocab_size,
            event_type_emb_dim=event_type_emb_dim,
            component_emb_dim=component_emb_dim,
            severity_emb_dim=severity_emb_dim,
            state_emb_dim=state_emb_dim,
            numeric_input_dim=max(0, input_dim - 4),
            output_dim=hidden_dim,
        ).to(self.device)
        self.encoder = TCNEncoder(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            depth=depth,
            kernel_size=kernel_size,
            dropout=dropout,
        ).to(self.device)
        self.projection_head = nn.Sequential(
            nn.Linear(self.pooled_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, projection_dim),
        ).to(self.device)
        self.attention_pool = AttentionPool(hidden_dim).to(self.device) if pooling_mode == "attention" else None
        self.masked_event_heads = nn.ModuleDict(
            {
                "event_type": nn.Linear(hidden_dim, event_type_vocab_size),
                "component": nn.Linear(hidden_dim, component_vocab_size),
                "severity": nn.Linear(hidden_dim, severity_vocab_size),
                "state": nn.Linear(hidden_dim, state_vocab_size),
            }
        ).to(self.device)
        self.order_prediction_head = nn.Linear(self.pooled_dim, 1).to(self.device)
        self.summary: dict = {}

    def fit(
        self,
        X_windows: np.ndarray,
        mask: np.ndarray,
        trajectory_id: np.ndarray | None = None,
        window_start: np.ndarray | None = None,
    ) -> LearnedModelResult:
        dataset = TensorDataset(
            torch.tensor(X_windows, dtype=torch.float32),
            torch.tensor(mask, dtype=torch.float32),
            torch.tensor(np.zeros(len(X_windows)) if trajectory_id is None else trajectory_id, dtype=torch.long),
            torch.tensor(np.arange(len(X_windows)) if window_start is None else window_start, dtype=torch.long),
        )
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=False)
        optimizer = torch.optim.Adam(
            list(self.input_encoder.parameters())
            + list(self.encoder.parameters())
            + list(self.projection_head.parameters())
            + list(self.masked_event_heads.parameters())
            + list(self.order_prediction_head.parameters()),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        if self.attention_pool is not None:
            optimizer.add_param_group({"params": self.attention_pool.parameters()})

        self.input_encoder.train()
        self.encoder.train()
        self.projection_head.train()
        epoch_losses: list[float] = []
        for epoch_idx in range(self.max_epochs):
            batch_losses: list[float] = []
            for batch_X, batch_mask, batch_trajectory_id, batch_window_start in loader:
                batch_X = batch_X.to(self.device)
                batch_mask = batch_mask.to(self.device)
                batch_trajectory_id = batch_trajectory_id.to(self.device)
                batch_window_start = batch_window_start.to(self.device)
                view1 = self._augment_encoded(self.input_encoder(batch_X), batch_mask)
                view2 = self._augment_encoded(self.input_encoder(batch_X), batch_mask)

                pooled1 = pool_sequence(self.encoder(view1), batch_mask, self.pooling_mode, self.attention_pool)
                pooled2 = pool_sequence(self.encoder(view2), batch_mask, self.pooling_mode, self.attention_pool)
                z1 = self.projection_head(pooled1)
                z2 = self.projection_head(pooled2)
                temporal_positive_mask = self._build_temporal_positive_mask(batch_trajectory_id, batch_window_start)
                contrastive_loss = multi_positive_contrastive_loss(
                    z1,
                    z2,
                    self.temperature,
                    temporal_positive_mask=temporal_positive_mask,
                )
                masked_prediction_loss = self._masked_event_prediction_loss(batch_X, batch_mask)
                order_prediction_loss = self._order_prediction_loss(batch_X, batch_mask)
                hard_negative_loss = self._hard_negative_loss(
                    z1=z1,
                    z2=z2,
                    batch_trajectory_id=batch_trajectory_id,
                    batch_window_start=batch_window_start,
                    temporal_positive_mask=temporal_positive_mask,
                )
                loss = (
                    contrastive_loss
                    + self.masked_prediction_weight * masked_prediction_loss
                    + self.order_prediction_weight * order_prediction_loss
                    + self.hard_negative_weight * hard_negative_loss
                )

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                batch_losses.append(float(loss.item()))

            epoch_losses.append(float(np.mean(batch_losses)))

        self.summary = {
            "model_type": self.model_type,
            "input_dim": self.input_dim,
            "window_length": self.window_length,
            "embedding_dim": self.pooled_dim,
            "projection_dim": self.projection_dim,
            "pooled_dim": self.pooled_dim,
            "max_epochs": self.max_epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "temperature": self.temperature,
            "temporal_positive_radius": self.temporal_positive_radius,
            "pooling_mode": self.pooling_mode,
            "masked_event_prob": self.masked_event_prob,
            "masked_prediction_weight": self.masked_prediction_weight,
            "mask_span_length": self.mask_span_length,
            "order_prediction_weight": self.order_prediction_weight,
            "order_span_length": self.order_span_length,
            "hard_negative_weight": self.hard_negative_weight,
            "hard_negative_margin": self.hard_negative_margin,
            "hard_negative_far_radius": self.hard_negative_far_radius,
            "train_loss_by_epoch": epoch_losses,
            "final_train_loss": epoch_losses[-1] if epoch_losses else None,
            "device": self.device,
        }
        return LearnedModelResult(summary=self.summary)

    @torch.no_grad()
    def transform(self, X_windows: np.ndarray, mask: np.ndarray) -> np.ndarray:
        self.input_encoder.eval()
        self.encoder.eval()
        X_tensor = torch.tensor(X_windows, dtype=torch.float32)
        mask_tensor = torch.tensor(mask, dtype=torch.float32)
        loader = DataLoader(TensorDataset(X_tensor, mask_tensor), batch_size=self.batch_size, shuffle=False)
        outputs: list[np.ndarray] = []
        for batch_X, batch_mask in loader:
            batch_X = batch_X.to(self.device)
            batch_mask = batch_mask.to(self.device)
            encoded = self.input_encoder(batch_X)
            pooled = pool_sequence(self.encoder(encoded), batch_mask, self.pooling_mode, self.attention_pool)
            outputs.append(pooled.cpu().numpy().astype(np.float32))
        return np.concatenate(outputs, axis=0)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_type": self.model_type,
                "config": self._config_dict(),
                "input_encoder_state_dict": self.input_encoder.state_dict(),
                "encoder_state_dict": self.encoder.state_dict(),
                "projection_head_state_dict": self.projection_head.state_dict(),
                "attention_pool_state_dict": None if self.attention_pool is None else self.attention_pool.state_dict(),
                "masked_event_heads_state_dict": self.masked_event_heads.state_dict(),
                "order_prediction_head_state_dict": self.order_prediction_head.state_dict(),
                "summary": self.summary,
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path, device: str = "cpu") -> "TS2VecStyleWindowEncoder":
        checkpoint = torch.load(Path(path), map_location=cls._resolve_device(device), weights_only=True)
        model = cls(device=device, **checkpoint["config"])
        model.input_encoder.load_state_dict(checkpoint["input_encoder_state_dict"])
        model.encoder.load_state_dict(checkpoint["encoder_state_dict"])
        model.projection_head.load_state_dict(checkpoint["projection_head_state_dict"])
        if model.attention_pool is not None and checkpoint.get("attention_pool_state_dict") is not None:
            model.attention_pool.load_state_dict(checkpoint["attention_pool_state_dict"])
        model.masked_event_heads.load_state_dict(checkpoint["masked_event_heads_state_dict"])
        model.order_prediction_head.load_state_dict(checkpoint["order_prediction_head_state_dict"])
        model.summary = checkpoint.get("summary", {})
        return model

    def training_summary(self) -> dict:
        return self.summary

    def _config_dict(self) -> dict:
        return {
            "input_dim": self.input_dim,
            "window_length": self.window_length,
            "hidden_dim": self.hidden_dim,
            "depth": self.depth,
            "kernel_size": self.kernel_size,
            "dropout": self.dropout,
            "event_type_vocab_size": self.event_type_vocab_size,
            "component_vocab_size": self.component_vocab_size,
            "severity_vocab_size": self.severity_vocab_size,
            "state_vocab_size": self.state_vocab_size,
            "event_type_emb_dim": self.event_type_emb_dim,
            "component_emb_dim": self.component_emb_dim,
            "severity_emb_dim": self.severity_emb_dim,
            "state_emb_dim": self.state_emb_dim,
            "projection_dim": self.projection_dim,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "max_epochs": self.max_epochs,
            "temperature": self.temperature,
            "time_mask_prob": self.time_mask_prob,
            "feature_dropout_prob": self.feature_dropout_prob,
            "noise_std": self.noise_std,
            "temporal_positive_radius": self.temporal_positive_radius,
            "pooling_mode": self.pooling_mode,
            "masked_event_prob": self.masked_event_prob,
            "masked_prediction_weight": self.masked_prediction_weight,
            "mask_span_length": self.mask_span_length,
            "order_prediction_weight": self.order_prediction_weight,
            "order_span_length": self.order_span_length,
            "hard_negative_weight": self.hard_negative_weight,
            "hard_negative_margin": self.hard_negative_margin,
            "hard_negative_far_radius": self.hard_negative_far_radius,
        }

    def _augment_encoded(self, encoded: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        augmented = encoded.clone()
        valid = mask.unsqueeze(-1)
        if self.noise_std > 0:
            augmented = augmented + torch.randn_like(augmented) * self.noise_std * valid

        if self.feature_dropout_prob > 0:
            feature_keep = (torch.rand(augmented.size(0), 1, augmented.size(2), device=augmented.device) > self.feature_dropout_prob).float()
            augmented = augmented * feature_keep

        if self.time_mask_prob > 0:
            time_keep = (torch.rand(augmented.size(0), augmented.size(1), 1, device=augmented.device) > self.time_mask_prob).float()
            augmented = augmented * time_keep

        return augmented * valid

    def _masked_event_prediction_loss(self, batch_X: torch.Tensor, batch_mask: torch.Tensor) -> torch.Tensor:
        if self.masked_event_prob <= 0:
            return torch.tensor(0.0, device=self.device)

        valid_positions = batch_mask > 0
        masked_positions = sample_mask_span_positions(
            valid_positions=valid_positions,
            masked_event_prob=self.masked_event_prob,
            mask_span_length=self.mask_span_length,
        )
        if not torch.any(masked_positions):
            return torch.tensor(0.0, device=self.device)

        encoded = self.input_encoder(batch_X, masked_event_positions=masked_positions)
        hidden = self.encoder(encoded)
        loss_terms = []
        targets = {
            "event_type": batch_X[:, :, 0].long(),
            "component": batch_X[:, :, 1].long(),
            "severity": batch_X[:, :, 2].long(),
            "state": batch_X[:, :, 3].long(),
        }
        for name, head in self.masked_event_heads.items():
            logits = head(hidden[masked_positions])
            target = targets[name][masked_positions]
            loss_terms.append(F.cross_entropy(logits, target))
        return torch.stack(loss_terms).mean()

    def _order_prediction_loss(self, batch_X: torch.Tensor, batch_mask: torch.Tensor) -> torch.Tensor:
        if self.order_prediction_weight <= 0:
            return torch.tensor(0.0, device=self.device)

        valid_positions = batch_mask > 0
        encoded = self.input_encoder(batch_X)
        shuffled = apply_local_order_shuffle(
            encoded=encoded,
            valid_positions=valid_positions,
            span_length=self.order_span_length,
        )

        original_pooled = pool_sequence(self.encoder(encoded), batch_mask, self.pooling_mode, self.attention_pool)
        shuffled_pooled = pool_sequence(self.encoder(shuffled), batch_mask, self.pooling_mode, self.attention_pool)
        logits = torch.cat(
            [
                self.order_prediction_head(original_pooled),
                self.order_prediction_head(shuffled_pooled),
            ],
            dim=0,
        ).squeeze(-1)
        labels = torch.cat(
            [
                torch.ones(original_pooled.size(0), device=self.device),
                torch.zeros(shuffled_pooled.size(0), device=self.device),
            ],
            dim=0,
        )
        return F.binary_cross_entropy_with_logits(logits, labels)

    def _hard_negative_loss(
        self,
        z1: torch.Tensor,
        z2: torch.Tensor,
        batch_trajectory_id: torch.Tensor,
        batch_window_start: torch.Tensor,
        temporal_positive_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        if self.hard_negative_weight <= 0:
            return torch.tensor(0.0, device=self.device)
        return conservative_hard_negative_loss(
            z1=z1,
            z2=z2,
            trajectory_id=batch_trajectory_id,
            window_start=batch_window_start,
            temporal_positive_mask=temporal_positive_mask,
            far_negative_radius=self.hard_negative_far_radius,
            similarity_margin=self.hard_negative_margin,
        )

    def _build_temporal_positive_mask(
        self,
        trajectory_id: torch.Tensor,
        window_start: torch.Tensor,
    ) -> torch.Tensor:
        same_trajectory = trajectory_id[:, None] == trajectory_id[None, :]
        start_distance = torch.abs(window_start[:, None] - window_start[None, :])
        nearby = (start_distance > 0) & (start_distance <= self.temporal_positive_radius)
        return same_trajectory & nearby

    @staticmethod
    def _resolve_device(device: str) -> str:
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda" and not torch.cuda.is_available():
            return "cpu"
        return device


class SequenceAutoencoder:
    """TCN encoder with mean pooling and a reconstruction decoder."""

    model_type = "sequence_autoencoder"

    def __init__(
        self,
        input_dim: int,
        window_length: int,
        hidden_dim: int = 64,
        depth: int = 4,
        kernel_size: int = 3,
        dropout: float = 0.1,
        event_type_vocab_size: int = 8,
        component_vocab_size: int = 4,
        severity_vocab_size: int = 4,
        state_vocab_size: int = 4,
        event_type_emb_dim: int = 16,
        component_emb_dim: int = 8,
        severity_emb_dim: int = 8,
        state_emb_dim: int = 8,
        batch_size: int = 64,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        max_epochs: int = 20,
        pooling_mode: str = "mean",
        device: str = "cpu",
    ) -> None:
        self.input_dim = input_dim
        self.window_length = window_length
        self.hidden_dim = hidden_dim
        self.depth = depth
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.event_type_vocab_size = event_type_vocab_size
        self.component_vocab_size = component_vocab_size
        self.severity_vocab_size = severity_vocab_size
        self.state_vocab_size = state_vocab_size
        self.event_type_emb_dim = event_type_emb_dim
        self.component_emb_dim = component_emb_dim
        self.severity_emb_dim = severity_emb_dim
        self.state_emb_dim = state_emb_dim
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs
        self.pooling_mode = pooling_mode
        self.device = TS2VecStyleWindowEncoder._resolve_device(device)
        self.pooled_dim = hidden_dim if pooling_mode in {"mean", "max"} else hidden_dim * 2

        self.input_encoder = CategoricalNumericInputEncoder(
            event_type_vocab_size=event_type_vocab_size,
            component_vocab_size=component_vocab_size,
            severity_vocab_size=severity_vocab_size,
            state_vocab_size=state_vocab_size,
            event_type_emb_dim=event_type_emb_dim,
            component_emb_dim=component_emb_dim,
            severity_emb_dim=severity_emb_dim,
            state_emb_dim=state_emb_dim,
            numeric_input_dim=max(0, input_dim - 4),
            output_dim=hidden_dim,
        ).to(self.device)
        self.encoder = TCNEncoder(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            depth=depth,
            kernel_size=kernel_size,
            dropout=dropout,
        ).to(self.device)
        self.decoder_input_proj = nn.Linear(self.pooled_dim, hidden_dim).to(self.device)
        self.decoder_position_embedding = nn.Embedding(window_length, hidden_dim).to(self.device)
        self.decoder_backbone = TCNEncoder(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            depth=max(1, depth // 2),
            kernel_size=kernel_size,
            dropout=dropout,
        ).to(self.device)
        self.decoder_heads = nn.ModuleDict(
            {
                "event_type": nn.Linear(hidden_dim, event_type_vocab_size),
                "component": nn.Linear(hidden_dim, component_vocab_size),
                "severity": nn.Linear(hidden_dim, severity_vocab_size),
                "state": nn.Linear(hidden_dim, state_vocab_size),
                "numeric": nn.Linear(hidden_dim, max(0, input_dim - 4)),
            }
        ).to(self.device)
        self.summary: dict = {}

    def fit(
        self,
        X_windows: np.ndarray,
        mask: np.ndarray,
        trajectory_id: np.ndarray | None = None,
        window_start: np.ndarray | None = None,
    ) -> LearnedModelResult:
        dataset = TensorDataset(
            torch.tensor(X_windows, dtype=torch.float32),
            torch.tensor(mask, dtype=torch.float32),
        )
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=False)
        optimizer = torch.optim.Adam(
            list(self.input_encoder.parameters())
            + list(self.encoder.parameters())
            + list(self.decoder_input_proj.parameters())
            + list(self.decoder_position_embedding.parameters())
            + list(self.decoder_backbone.parameters())
            + list(self.decoder_heads.parameters()),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        self.input_encoder.train()
        self.encoder.train()
        self.decoder_input_proj.train()
        self.decoder_position_embedding.train()
        self.decoder_backbone.train()
        self.decoder_heads.train()
        epoch_losses: list[float] = []
        for _ in range(self.max_epochs):
            batch_losses: list[float] = []
            for batch_X, batch_mask in loader:
                batch_X = batch_X.to(self.device)
                batch_mask = batch_mask.to(self.device)
                encoded = self.input_encoder(batch_X)
                pooled = pool_sequence(self.encoder(encoded), batch_mask, self.pooling_mode)
                decoded = self._decode_sequence(pooled)
                loss = self._reconstruction_loss(decoded, batch_X, batch_mask)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                batch_losses.append(float(loss.item()))

            epoch_losses.append(float(np.mean(batch_losses)))

        self.summary = {
            "model_type": self.model_type,
            "input_dim": self.input_dim,
            "window_length": self.window_length,
            "embedding_dim": self.pooled_dim,
            "pooled_dim": self.pooled_dim,
            "max_epochs": self.max_epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "pooling_mode": self.pooling_mode,
            "train_loss_by_epoch": epoch_losses,
            "final_train_loss": epoch_losses[-1] if epoch_losses else None,
            "device": self.device,
        }
        return LearnedModelResult(summary=self.summary)

    @torch.no_grad()
    def transform(self, X_windows: np.ndarray, mask: np.ndarray) -> np.ndarray:
        self.input_encoder.eval()
        self.encoder.eval()
        X_tensor = torch.tensor(X_windows, dtype=torch.float32)
        mask_tensor = torch.tensor(mask, dtype=torch.float32)
        loader = DataLoader(TensorDataset(X_tensor, mask_tensor), batch_size=self.batch_size, shuffle=False)
        outputs: list[np.ndarray] = []
        for batch_X, batch_mask in loader:
            batch_X = batch_X.to(self.device)
            batch_mask = batch_mask.to(self.device)
            encoded = self.input_encoder(batch_X)
            pooled = pool_sequence(self.encoder(encoded), batch_mask, self.pooling_mode)
            outputs.append(pooled.cpu().numpy().astype(np.float32))
        return np.concatenate(outputs, axis=0)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_type": self.model_type,
                "config": self._config_dict(),
                "input_encoder_state_dict": self.input_encoder.state_dict(),
                "encoder_state_dict": self.encoder.state_dict(),
                "decoder_input_proj_state_dict": self.decoder_input_proj.state_dict(),
                "decoder_position_embedding_state_dict": self.decoder_position_embedding.state_dict(),
                "decoder_backbone_state_dict": self.decoder_backbone.state_dict(),
                "decoder_heads_state_dict": self.decoder_heads.state_dict(),
                "summary": self.summary,
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path, device: str = "cpu") -> "SequenceAutoencoder":
        checkpoint = torch.load(
            Path(path),
            map_location=TS2VecStyleWindowEncoder._resolve_device(device),
            weights_only=True,
        )
        model = cls(device=device, **checkpoint["config"])
        model.input_encoder.load_state_dict(checkpoint["input_encoder_state_dict"])
        model.encoder.load_state_dict(checkpoint["encoder_state_dict"])
        model.decoder_input_proj.load_state_dict(checkpoint["decoder_input_proj_state_dict"])
        model.decoder_position_embedding.load_state_dict(checkpoint["decoder_position_embedding_state_dict"])
        model.decoder_backbone.load_state_dict(checkpoint["decoder_backbone_state_dict"])
        model.decoder_heads.load_state_dict(checkpoint["decoder_heads_state_dict"])
        model.summary = checkpoint.get("summary", {})
        return model

    def training_summary(self) -> dict:
        return self.summary

    def _config_dict(self) -> dict:
        return {
            "input_dim": self.input_dim,
            "window_length": self.window_length,
            "hidden_dim": self.hidden_dim,
            "depth": self.depth,
            "kernel_size": self.kernel_size,
            "dropout": self.dropout,
            "event_type_vocab_size": self.event_type_vocab_size,
            "component_vocab_size": self.component_vocab_size,
            "severity_vocab_size": self.severity_vocab_size,
            "state_vocab_size": self.state_vocab_size,
            "event_type_emb_dim": self.event_type_emb_dim,
            "component_emb_dim": self.component_emb_dim,
            "severity_emb_dim": self.severity_emb_dim,
            "state_emb_dim": self.state_emb_dim,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "max_epochs": self.max_epochs,
            "pooling_mode": self.pooling_mode,
        }

    def _decode_sequence(self, pooled: torch.Tensor) -> dict[str, torch.Tensor]:
        batch_size = pooled.shape[0]
        base = self.decoder_input_proj(pooled).unsqueeze(1).expand(batch_size, self.window_length, self.hidden_dim)
        positions = torch.arange(self.window_length, device=self.device)
        position_embeddings = self.decoder_position_embedding(positions).unsqueeze(0).expand(batch_size, -1, -1)
        hidden = self.decoder_backbone(base + position_embeddings)
        return {
            "event_type": self.decoder_heads["event_type"](hidden),
            "component": self.decoder_heads["component"](hidden),
            "severity": self.decoder_heads["severity"](hidden),
            "state": self.decoder_heads["state"](hidden),
            "numeric": self.decoder_heads["numeric"](hidden),
        }

    @staticmethod
    def _reconstruction_loss(decoded: dict[str, torch.Tensor], target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        valid_positions = mask > 0
        loss_terms = []
        categorical_targets = {
            "event_type": target[:, :, 0].long(),
            "component": target[:, :, 1].long(),
            "severity": target[:, :, 2].long(),
            "state": target[:, :, 3].long(),
        }
        for name, labels in categorical_targets.items():
            logits = decoded[name][valid_positions]
            refs = labels[valid_positions]
            loss_terms.append(F.cross_entropy(logits, refs))

        numeric_pred = decoded["numeric"][valid_positions]
        numeric_target = target[:, :, 4:][valid_positions]
        if numeric_pred.numel() > 0:
            loss_terms.append(F.mse_loss(numeric_pred, numeric_target))
        return torch.stack(loss_terms).mean()


class MaskedEventAutoencoder:
    """Masked-event sequence model that predicts only masked event fields."""

    model_type = "masked_event_autoencoder"

    def __init__(
        self,
        input_dim: int,
        window_length: int,
        hidden_dim: int = 64,
        depth: int = 4,
        kernel_size: int = 3,
        dropout: float = 0.1,
        event_type_vocab_size: int = 8,
        component_vocab_size: int = 4,
        severity_vocab_size: int = 4,
        state_vocab_size: int = 4,
        event_type_emb_dim: int = 16,
        component_emb_dim: int = 8,
        severity_emb_dim: int = 8,
        state_emb_dim: int = 8,
        batch_size: int = 64,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        max_epochs: int = 20,
        pooling_mode: str = "mean",
        masked_event_prob: float = 0.15,
        mask_span_length: int = 1,
        device: str = "cpu",
    ) -> None:
        self.input_dim = input_dim
        self.window_length = window_length
        self.hidden_dim = hidden_dim
        self.depth = depth
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.event_type_vocab_size = event_type_vocab_size
        self.component_vocab_size = component_vocab_size
        self.severity_vocab_size = severity_vocab_size
        self.state_vocab_size = state_vocab_size
        self.event_type_emb_dim = event_type_emb_dim
        self.component_emb_dim = component_emb_dim
        self.severity_emb_dim = severity_emb_dim
        self.state_emb_dim = state_emb_dim
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs
        self.pooling_mode = pooling_mode
        self.masked_event_prob = masked_event_prob
        self.mask_span_length = mask_span_length
        self.device = TS2VecStyleWindowEncoder._resolve_device(device)
        self.pooled_dim = hidden_dim if pooling_mode in {"mean", "max"} else hidden_dim * 2

        self.input_encoder = CategoricalNumericInputEncoder(
            event_type_vocab_size=event_type_vocab_size,
            component_vocab_size=component_vocab_size,
            severity_vocab_size=severity_vocab_size,
            state_vocab_size=state_vocab_size,
            event_type_emb_dim=event_type_emb_dim,
            component_emb_dim=component_emb_dim,
            severity_emb_dim=severity_emb_dim,
            state_emb_dim=state_emb_dim,
            numeric_input_dim=max(0, input_dim - 4),
            output_dim=hidden_dim,
        ).to(self.device)
        self.encoder = TCNEncoder(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            depth=depth,
            kernel_size=kernel_size,
            dropout=dropout,
        ).to(self.device)
        self.masked_event_heads = nn.ModuleDict(
            {
                "event_type": nn.Linear(hidden_dim, event_type_vocab_size),
                "component": nn.Linear(hidden_dim, component_vocab_size),
                "severity": nn.Linear(hidden_dim, severity_vocab_size),
                "state": nn.Linear(hidden_dim, state_vocab_size),
                "numeric": nn.Linear(hidden_dim, max(0, input_dim - 4)),
            }
        ).to(self.device)
        self.summary: dict = {}

    def fit(
        self,
        X_windows: np.ndarray,
        mask: np.ndarray,
        trajectory_id: np.ndarray | None = None,
        window_start: np.ndarray | None = None,
    ) -> LearnedModelResult:
        dataset = TensorDataset(
            torch.tensor(X_windows, dtype=torch.float32),
            torch.tensor(mask, dtype=torch.float32),
        )
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=False)
        optimizer = torch.optim.Adam(
            list(self.input_encoder.parameters())
            + list(self.encoder.parameters())
            + list(self.masked_event_heads.parameters()),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        self.input_encoder.train()
        self.encoder.train()
        self.masked_event_heads.train()
        epoch_losses: list[float] = []
        for _ in range(self.max_epochs):
            batch_losses: list[float] = []
            for batch_X, batch_mask in loader:
                batch_X = batch_X.to(self.device)
                batch_mask = batch_mask.to(self.device)
                loss = self._masked_event_prediction_loss(batch_X, batch_mask)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                batch_losses.append(float(loss.item()))

            epoch_losses.append(float(np.mean(batch_losses)))

        self.summary = {
            "model_type": self.model_type,
            "input_dim": self.input_dim,
            "window_length": self.window_length,
            "embedding_dim": self.pooled_dim,
            "pooled_dim": self.pooled_dim,
            "max_epochs": self.max_epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "pooling_mode": self.pooling_mode,
            "masked_event_prob": self.masked_event_prob,
            "mask_span_length": self.mask_span_length,
            "train_loss_by_epoch": epoch_losses,
            "final_train_loss": epoch_losses[-1] if epoch_losses else None,
            "device": self.device,
        }
        return LearnedModelResult(summary=self.summary)

    @torch.no_grad()
    def transform(self, X_windows: np.ndarray, mask: np.ndarray) -> np.ndarray:
        self.input_encoder.eval()
        self.encoder.eval()
        X_tensor = torch.tensor(X_windows, dtype=torch.float32)
        mask_tensor = torch.tensor(mask, dtype=torch.float32)
        loader = DataLoader(TensorDataset(X_tensor, mask_tensor), batch_size=self.batch_size, shuffle=False)
        outputs: list[np.ndarray] = []
        for batch_X, batch_mask in loader:
            batch_X = batch_X.to(self.device)
            batch_mask = batch_mask.to(self.device)
            encoded = self.input_encoder(batch_X)
            pooled = pool_sequence(self.encoder(encoded), batch_mask, self.pooling_mode)
            outputs.append(pooled.cpu().numpy().astype(np.float32))
        return np.concatenate(outputs, axis=0)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_type": self.model_type,
                "config": self._config_dict(),
                "input_encoder_state_dict": self.input_encoder.state_dict(),
                "encoder_state_dict": self.encoder.state_dict(),
                "masked_event_heads_state_dict": self.masked_event_heads.state_dict(),
                "summary": self.summary,
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path, device: str = "cpu") -> "MaskedEventAutoencoder":
        checkpoint = torch.load(
            Path(path),
            map_location=TS2VecStyleWindowEncoder._resolve_device(device),
            weights_only=True,
        )
        model = cls(device=device, **checkpoint["config"])
        model.input_encoder.load_state_dict(checkpoint["input_encoder_state_dict"])
        model.encoder.load_state_dict(checkpoint["encoder_state_dict"])
        model.masked_event_heads.load_state_dict(checkpoint["masked_event_heads_state_dict"])
        model.summary = checkpoint.get("summary", {})
        return model

    def training_summary(self) -> dict:
        return self.summary

    def _config_dict(self) -> dict:
        return {
            "input_dim": self.input_dim,
            "window_length": self.window_length,
            "hidden_dim": self.hidden_dim,
            "depth": self.depth,
            "kernel_size": self.kernel_size,
            "dropout": self.dropout,
            "event_type_vocab_size": self.event_type_vocab_size,
            "component_vocab_size": self.component_vocab_size,
            "severity_vocab_size": self.severity_vocab_size,
            "state_vocab_size": self.state_vocab_size,
            "event_type_emb_dim": self.event_type_emb_dim,
            "component_emb_dim": self.component_emb_dim,
            "severity_emb_dim": self.severity_emb_dim,
            "state_emb_dim": self.state_emb_dim,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "max_epochs": self.max_epochs,
            "pooling_mode": self.pooling_mode,
            "masked_event_prob": self.masked_event_prob,
            "mask_span_length": self.mask_span_length,
        }

    def _masked_event_prediction_loss(self, batch_X: torch.Tensor, batch_mask: torch.Tensor) -> torch.Tensor:
        valid_positions = batch_mask > 0
        masked_positions = sample_mask_span_positions(
            valid_positions=valid_positions,
            masked_event_prob=self.masked_event_prob,
            mask_span_length=self.mask_span_length,
        )
        if not torch.any(masked_positions):
            return torch.tensor(0.0, device=self.device)

        encoded = self.input_encoder(batch_X, masked_event_positions=masked_positions)
        hidden = self.encoder(encoded)
        loss_terms = []

        categorical_targets = {
            "event_type": batch_X[:, :, 0].long(),
            "component": batch_X[:, :, 1].long(),
            "severity": batch_X[:, :, 2].long(),
            "state": batch_X[:, :, 3].long(),
        }
        for name, labels in categorical_targets.items():
            logits = self.masked_event_heads[name](hidden[masked_positions])
            refs = labels[masked_positions]
            loss_terms.append(F.cross_entropy(logits, refs))

        numeric_pred = self.masked_event_heads["numeric"](hidden[masked_positions])
        numeric_target = batch_X[:, :, 4:][masked_positions]
        if numeric_pred.numel() > 0:
            loss_terms.append(F.mse_loss(numeric_pred, numeric_target))
        return torch.stack(loss_terms).mean()


class LogBERTStyleWindowEncoder:
    """Masked Transformer encoder for log windows with CLS or attention pooling."""

    model_type = "logbert_style"

    def __init__(
        self,
        input_dim: int,
        window_length: int,
        hidden_dim: int = 128,
        depth: int = 4,
        kernel_size: int = 3,
        dropout: float = 0.1,
        event_type_vocab_size: int = 8,
        component_vocab_size: int = 4,
        severity_vocab_size: int = 4,
        state_vocab_size: int = 4,
        event_type_emb_dim: int = 16,
        component_emb_dim: int = 8,
        severity_emb_dim: int = 8,
        state_emb_dim: int = 8,
        batch_size: int = 64,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        max_epochs: int = 20,
        pooling_mode: str = "cls",
        masked_event_prob: float = 0.15,
        mask_span_length: int = 1,
        num_attention_heads: int = 4,
        feedforward_dim: int = 256,
        device: str = "cpu",
    ) -> None:
        self.input_dim = input_dim
        self.window_length = window_length
        self.hidden_dim = hidden_dim
        self.depth = depth
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.event_type_vocab_size = event_type_vocab_size
        self.component_vocab_size = component_vocab_size
        self.severity_vocab_size = severity_vocab_size
        self.state_vocab_size = state_vocab_size
        self.event_type_emb_dim = event_type_emb_dim
        self.component_emb_dim = component_emb_dim
        self.severity_emb_dim = severity_emb_dim
        self.state_emb_dim = state_emb_dim
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs
        self.pooling_mode = pooling_mode
        self.masked_event_prob = masked_event_prob
        self.mask_span_length = mask_span_length
        self.num_attention_heads = num_attention_heads
        self.feedforward_dim = feedforward_dim
        self.device = TS2VecStyleWindowEncoder._resolve_device(device)
        self.pooled_dim = hidden_dim

        self.input_encoder = CategoricalNumericInputEncoder(
            event_type_vocab_size=event_type_vocab_size,
            component_vocab_size=component_vocab_size,
            severity_vocab_size=severity_vocab_size,
            state_vocab_size=state_vocab_size,
            event_type_emb_dim=event_type_emb_dim,
            component_emb_dim=component_emb_dim,
            severity_emb_dim=severity_emb_dim,
            state_emb_dim=state_emb_dim,
            numeric_input_dim=max(0, input_dim - 4),
            output_dim=hidden_dim,
        ).to(self.device)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim, device=self.device))
        self.position_embedding = nn.Embedding(window_length + 1, hidden_dim).to(self.device)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_attention_heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth).to(self.device)
        self.attention_pool = AttentionPool(hidden_dim).to(self.device) if pooling_mode == "attention" else None
        self.masked_event_heads = nn.ModuleDict(
            {
                "event_type": nn.Linear(hidden_dim, event_type_vocab_size),
                "component": nn.Linear(hidden_dim, component_vocab_size),
                "severity": nn.Linear(hidden_dim, severity_vocab_size),
                "state": nn.Linear(hidden_dim, state_vocab_size),
                "numeric": nn.Linear(hidden_dim, max(0, input_dim - 4)),
            }
        ).to(self.device)
        self.summary: dict = {}

    def fit(
        self,
        X_windows: np.ndarray,
        mask: np.ndarray,
        trajectory_id: np.ndarray | None = None,
        window_start: np.ndarray | None = None,
    ) -> LearnedModelResult:
        dataset = TensorDataset(
            torch.tensor(X_windows, dtype=torch.float32),
            torch.tensor(mask, dtype=torch.float32),
        )
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=False)
        params = (
            list(self.input_encoder.parameters())
            + list(self.position_embedding.parameters())
            + list(self.transformer.parameters())
            + list(self.masked_event_heads.parameters())
        )
        if self.attention_pool is not None:
            params += list(self.attention_pool.parameters())
        optimizer = torch.optim.Adam(params, lr=self.learning_rate, weight_decay=self.weight_decay)

        self.input_encoder.train()
        self.transformer.train()
        self.masked_event_heads.train()
        if self.attention_pool is not None:
            self.attention_pool.train()

        epoch_losses: list[float] = []
        for _ in range(self.max_epochs):
            batch_losses: list[float] = []
            for batch_X, batch_mask in loader:
                batch_X = batch_X.to(self.device)
                batch_mask = batch_mask.to(self.device)
                loss = self._masked_event_prediction_loss(batch_X, batch_mask)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                batch_losses.append(float(loss.item()))

            epoch_losses.append(float(np.mean(batch_losses)))

        self.summary = {
            "model_type": self.model_type,
            "input_dim": self.input_dim,
            "window_length": self.window_length,
            "embedding_dim": self.pooled_dim,
            "max_epochs": self.max_epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "pooling_mode": self.pooling_mode,
            "masked_event_prob": self.masked_event_prob,
            "mask_span_length": self.mask_span_length,
            "num_attention_heads": self.num_attention_heads,
            "feedforward_dim": self.feedforward_dim,
            "train_loss_by_epoch": epoch_losses,
            "final_train_loss": epoch_losses[-1] if epoch_losses else None,
            "device": self.device,
        }
        return LearnedModelResult(summary=self.summary)

    @torch.no_grad()
    def transform(self, X_windows: np.ndarray, mask: np.ndarray) -> np.ndarray:
        self.input_encoder.eval()
        self.transformer.eval()
        if self.attention_pool is not None:
            self.attention_pool.eval()
        X_tensor = torch.tensor(X_windows, dtype=torch.float32)
        mask_tensor = torch.tensor(mask, dtype=torch.float32)
        loader = DataLoader(TensorDataset(X_tensor, mask_tensor), batch_size=self.batch_size, shuffle=False)
        outputs: list[np.ndarray] = []
        for batch_X, batch_mask in loader:
            batch_X = batch_X.to(self.device)
            batch_mask = batch_mask.to(self.device)
            hidden = self._encode_sequence(batch_X, batch_mask)
            pooled = self._pool_hidden(hidden, batch_mask)
            outputs.append(pooled.cpu().numpy().astype(np.float32))
        return np.concatenate(outputs, axis=0)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_type": self.model_type,
                "config": self._config_dict(),
                "input_encoder_state_dict": self.input_encoder.state_dict(),
                "cls_token": self.cls_token.detach().cpu(),
                "position_embedding_state_dict": self.position_embedding.state_dict(),
                "transformer_state_dict": self.transformer.state_dict(),
                "attention_pool_state_dict": None if self.attention_pool is None else self.attention_pool.state_dict(),
                "masked_event_heads_state_dict": self.masked_event_heads.state_dict(),
                "summary": self.summary,
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path, device: str = "cpu") -> "LogBERTStyleWindowEncoder":
        checkpoint = torch.load(
            Path(path),
            map_location=TS2VecStyleWindowEncoder._resolve_device(device),
            weights_only=True,
        )
        model = cls(device=device, **checkpoint["config"])
        model.input_encoder.load_state_dict(checkpoint["input_encoder_state_dict"])
        model.cls_token.data.copy_(checkpoint["cls_token"].to(model.device))
        model.position_embedding.load_state_dict(checkpoint["position_embedding_state_dict"])
        model.transformer.load_state_dict(checkpoint["transformer_state_dict"])
        if model.attention_pool is not None and checkpoint.get("attention_pool_state_dict") is not None:
            model.attention_pool.load_state_dict(checkpoint["attention_pool_state_dict"])
        model.masked_event_heads.load_state_dict(checkpoint["masked_event_heads_state_dict"])
        model.summary = checkpoint.get("summary", {})
        return model

    def training_summary(self) -> dict:
        return self.summary

    def _config_dict(self) -> dict:
        return {
            "input_dim": self.input_dim,
            "window_length": self.window_length,
            "hidden_dim": self.hidden_dim,
            "depth": self.depth,
            "kernel_size": self.kernel_size,
            "dropout": self.dropout,
            "event_type_vocab_size": self.event_type_vocab_size,
            "component_vocab_size": self.component_vocab_size,
            "severity_vocab_size": self.severity_vocab_size,
            "state_vocab_size": self.state_vocab_size,
            "event_type_emb_dim": self.event_type_emb_dim,
            "component_emb_dim": self.component_emb_dim,
            "severity_emb_dim": self.severity_emb_dim,
            "state_emb_dim": self.state_emb_dim,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "max_epochs": self.max_epochs,
            "pooling_mode": self.pooling_mode,
            "masked_event_prob": self.masked_event_prob,
            "mask_span_length": self.mask_span_length,
            "num_attention_heads": self.num_attention_heads,
            "feedforward_dim": self.feedforward_dim,
        }

    def _encode_sequence(
        self,
        batch_X: torch.Tensor,
        batch_mask: torch.Tensor,
        masked_positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        token_hidden = self.input_encoder(batch_X, masked_event_positions=masked_positions)
        batch_size = token_hidden.shape[0]
        cls = self.cls_token.expand(batch_size, -1, -1)
        sequence = torch.cat([cls, token_hidden], dim=1)
        positions = torch.arange(self.window_length + 1, device=self.device)
        sequence = sequence + self.position_embedding(positions).unsqueeze(0)
        key_padding_mask = torch.cat(
            [
                torch.zeros((batch_size, 1), dtype=torch.bool, device=self.device),
                batch_mask <= 0,
            ],
            dim=1,
        )
        return self.transformer(sequence, src_key_padding_mask=key_padding_mask)

    def _pool_hidden(self, hidden: torch.Tensor, batch_mask: torch.Tensor) -> torch.Tensor:
        if self.pooling_mode == "cls":
            return hidden[:, 0, :]
        if self.pooling_mode == "attention":
            return self.attention_pool(hidden[:, 1:, :], batch_mask)
        raise ValueError(f"Unknown pooling_mode: {self.pooling_mode}")

    def _masked_event_prediction_loss(self, batch_X: torch.Tensor, batch_mask: torch.Tensor) -> torch.Tensor:
        valid_positions = batch_mask > 0
        masked_positions = sample_mask_span_positions(
            valid_positions=valid_positions,
            masked_event_prob=self.masked_event_prob,
            mask_span_length=self.mask_span_length,
        )
        if not torch.any(masked_positions):
            return torch.tensor(0.0, device=self.device)

        hidden = self._encode_sequence(batch_X, batch_mask, masked_positions=masked_positions)
        token_hidden = hidden[:, 1:, :]
        loss_terms = []
        categorical_targets = {
            "event_type": batch_X[:, :, 0].long(),
            "component": batch_X[:, :, 1].long(),
            "severity": batch_X[:, :, 2].long(),
            "state": batch_X[:, :, 3].long(),
        }
        for name, labels in categorical_targets.items():
            logits = self.masked_event_heads[name](token_hidden[masked_positions])
            refs = labels[masked_positions]
            loss_terms.append(F.cross_entropy(logits, refs))

        numeric_pred = self.masked_event_heads["numeric"](token_hidden[masked_positions])
        numeric_target = batch_X[:, :, 4:][masked_positions]
        if numeric_pred.numel() > 0:
            loss_terms.append(F.mse_loss(numeric_pred, numeric_target))
        return torch.stack(loss_terms).mean()


class TimeDRLStyleWindowEncoder:
    """CLS-Transformer with masked event modeling plus instance contrastive learning."""

    model_type = "timedrl_style"

    def __init__(
        self,
        input_dim: int,
        window_length: int,
        hidden_dim: int = 128,
        depth: int = 3,
        kernel_size: int = 3,
        dropout: float = 0.1,
        event_type_vocab_size: int = 8,
        component_vocab_size: int = 4,
        severity_vocab_size: int = 4,
        state_vocab_size: int = 4,
        event_type_emb_dim: int = 16,
        component_emb_dim: int = 8,
        severity_emb_dim: int = 8,
        state_emb_dim: int = 8,
        batch_size: int = 128,
        learning_rate: float = 5e-4,
        weight_decay: float = 1e-4,
        max_epochs: int = 80,
        pooling_mode: str = "cls",
        masked_event_prob: float = 0.15,
        mask_span_length: int = 1,
        temperature: float = 0.15,
        projection_dim: int = 128,
        num_attention_heads: int = 4,
        feedforward_dim: int = 256,
        time_mask_prob: float = 0.05,
        feature_dropout_prob: float = 0.05,
        noise_std: float = 0.01,
        masked_prediction_weight: float = 1.0,
        contrastive_weight: float = 1.0,
        device: str = "cpu",
    ) -> None:
        self.input_dim = input_dim
        self.window_length = window_length
        self.hidden_dim = hidden_dim
        self.depth = depth
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.event_type_vocab_size = event_type_vocab_size
        self.component_vocab_size = component_vocab_size
        self.severity_vocab_size = severity_vocab_size
        self.state_vocab_size = state_vocab_size
        self.event_type_emb_dim = event_type_emb_dim
        self.component_emb_dim = component_emb_dim
        self.severity_emb_dim = severity_emb_dim
        self.state_emb_dim = state_emb_dim
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs
        self.pooling_mode = pooling_mode
        self.masked_event_prob = masked_event_prob
        self.mask_span_length = mask_span_length
        self.temperature = temperature
        self.projection_dim = projection_dim
        self.num_attention_heads = num_attention_heads
        self.feedforward_dim = feedforward_dim
        self.time_mask_prob = time_mask_prob
        self.feature_dropout_prob = feature_dropout_prob
        self.noise_std = noise_std
        self.masked_prediction_weight = masked_prediction_weight
        self.contrastive_weight = contrastive_weight
        self.device = TS2VecStyleWindowEncoder._resolve_device(device)
        self.pooled_dim = hidden_dim if pooling_mode in {"cls", "mean", "max", "attention"} else hidden_dim * 2

        self.input_encoder = CategoricalNumericInputEncoder(
            event_type_vocab_size=event_type_vocab_size,
            component_vocab_size=component_vocab_size,
            severity_vocab_size=severity_vocab_size,
            state_vocab_size=state_vocab_size,
            event_type_emb_dim=event_type_emb_dim,
            component_emb_dim=component_emb_dim,
            severity_emb_dim=severity_emb_dim,
            state_emb_dim=state_emb_dim,
            numeric_input_dim=max(0, input_dim - 4),
            output_dim=hidden_dim,
        ).to(self.device)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim, device=self.device))
        self.position_embedding = nn.Embedding(window_length + 1, hidden_dim).to(self.device)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_attention_heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth).to(self.device)
        self.attention_pool = AttentionPool(hidden_dim).to(self.device) if pooling_mode == "attention" else None
        self.projection_head = nn.Sequential(
            nn.Linear(self.pooled_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, projection_dim),
        ).to(self.device)
        self.masked_event_heads = nn.ModuleDict(
            {
                "event_type": nn.Linear(hidden_dim, event_type_vocab_size),
                "component": nn.Linear(hidden_dim, component_vocab_size),
                "severity": nn.Linear(hidden_dim, severity_vocab_size),
                "state": nn.Linear(hidden_dim, state_vocab_size),
                "numeric": nn.Linear(hidden_dim, max(0, input_dim - 4)),
            }
        ).to(self.device)
        self.summary: dict = {}

    def fit(
        self,
        X_windows: np.ndarray,
        mask: np.ndarray,
        trajectory_id: np.ndarray | None = None,
        window_start: np.ndarray | None = None,
    ) -> LearnedModelResult:
        dataset = TensorDataset(
            torch.tensor(X_windows, dtype=torch.float32),
            torch.tensor(mask, dtype=torch.float32),
        )
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=False)
        optimizer = torch.optim.Adam(
            list(self.input_encoder.parameters())
            + list(self.position_embedding.parameters())
            + list(self.transformer.parameters())
            + list(self.projection_head.parameters())
            + list(self.masked_event_heads.parameters()),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        if self.attention_pool is not None:
            optimizer.add_param_group({"params": self.attention_pool.parameters()})

        self.input_encoder.train()
        self.transformer.train()
        self.projection_head.train()
        self.masked_event_heads.train()
        if self.attention_pool is not None:
            self.attention_pool.train()

        epoch_losses: list[float] = []
        for _ in range(self.max_epochs):
            batch_losses: list[float] = []
            for batch_X, batch_mask in loader:
                batch_X = batch_X.to(self.device)
                batch_mask = batch_mask.to(self.device)

                view1 = self._augment_encoded(self.input_encoder(batch_X), batch_mask)
                view2 = self._augment_encoded(self.input_encoder(batch_X), batch_mask)
                pooled1 = self._pool_hidden(self._encode_from_hidden(view1, batch_mask), batch_mask)
                pooled2 = self._pool_hidden(self._encode_from_hidden(view2, batch_mask), batch_mask)
                z1 = self.projection_head(pooled1)
                z2 = self.projection_head(pooled2)
                contrastive_loss = multi_positive_contrastive_loss(
                    z1,
                    z2,
                    self.temperature,
                )
                masked_loss = self._masked_event_prediction_loss(batch_X, batch_mask)
                loss = self.contrastive_weight * contrastive_loss + self.masked_prediction_weight * masked_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                batch_losses.append(float(loss.item()))

            epoch_losses.append(float(np.mean(batch_losses)))

        self.summary = {
            "model_type": self.model_type,
            "input_dim": self.input_dim,
            "window_length": self.window_length,
            "embedding_dim": self.pooled_dim,
            "projection_dim": self.projection_dim,
            "max_epochs": self.max_epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "pooling_mode": self.pooling_mode,
            "masked_event_prob": self.masked_event_prob,
            "mask_span_length": self.mask_span_length,
            "temperature": self.temperature,
            "masked_prediction_weight": self.masked_prediction_weight,
            "contrastive_weight": self.contrastive_weight,
            "train_loss_by_epoch": epoch_losses,
            "final_train_loss": epoch_losses[-1] if epoch_losses else None,
            "device": self.device,
        }
        return LearnedModelResult(summary=self.summary)

    @torch.no_grad()
    def transform(self, X_windows: np.ndarray, mask: np.ndarray) -> np.ndarray:
        self.input_encoder.eval()
        self.transformer.eval()
        if self.attention_pool is not None:
            self.attention_pool.eval()
        X_tensor = torch.tensor(X_windows, dtype=torch.float32)
        mask_tensor = torch.tensor(mask, dtype=torch.float32)
        loader = DataLoader(TensorDataset(X_tensor, mask_tensor), batch_size=self.batch_size, shuffle=False)
        outputs: list[np.ndarray] = []
        for batch_X, batch_mask in loader:
            batch_X = batch_X.to(self.device)
            batch_mask = batch_mask.to(self.device)
            hidden = self._encode_sequence(batch_X, batch_mask)
            pooled = self._pool_hidden(hidden, batch_mask)
            outputs.append(pooled.cpu().numpy().astype(np.float32))
        return np.concatenate(outputs, axis=0)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_type": self.model_type,
                "config": self._config_dict(),
                "input_encoder_state_dict": self.input_encoder.state_dict(),
                "cls_token": self.cls_token.detach().cpu(),
                "position_embedding_state_dict": self.position_embedding.state_dict(),
                "transformer_state_dict": self.transformer.state_dict(),
                "projection_head_state_dict": self.projection_head.state_dict(),
                "masked_event_heads_state_dict": self.masked_event_heads.state_dict(),
                "attention_pool_state_dict": None if self.attention_pool is None else self.attention_pool.state_dict(),
                "summary": self.summary,
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path, device: str = "cpu") -> "TimeDRLStyleWindowEncoder":
        checkpoint = torch.load(
            Path(path),
            map_location=TS2VecStyleWindowEncoder._resolve_device(device),
            weights_only=True,
        )
        model = cls(device=device, **checkpoint["config"])
        model.input_encoder.load_state_dict(checkpoint["input_encoder_state_dict"])
        model.cls_token.data.copy_(checkpoint["cls_token"].to(model.device))
        model.position_embedding.load_state_dict(checkpoint["position_embedding_state_dict"])
        model.transformer.load_state_dict(checkpoint["transformer_state_dict"])
        model.projection_head.load_state_dict(checkpoint["projection_head_state_dict"])
        model.masked_event_heads.load_state_dict(checkpoint["masked_event_heads_state_dict"])
        if model.attention_pool is not None and checkpoint.get("attention_pool_state_dict") is not None:
            model.attention_pool.load_state_dict(checkpoint["attention_pool_state_dict"])
        model.summary = checkpoint.get("summary", {})
        return model

    def training_summary(self) -> dict:
        return self.summary

    def _config_dict(self) -> dict:
        return {
            "input_dim": self.input_dim,
            "window_length": self.window_length,
            "hidden_dim": self.hidden_dim,
            "depth": self.depth,
            "kernel_size": self.kernel_size,
            "dropout": self.dropout,
            "event_type_vocab_size": self.event_type_vocab_size,
            "component_vocab_size": self.component_vocab_size,
            "severity_vocab_size": self.severity_vocab_size,
            "state_vocab_size": self.state_vocab_size,
            "event_type_emb_dim": self.event_type_emb_dim,
            "component_emb_dim": self.component_emb_dim,
            "severity_emb_dim": self.severity_emb_dim,
            "state_emb_dim": self.state_emb_dim,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "max_epochs": self.max_epochs,
            "pooling_mode": self.pooling_mode,
            "masked_event_prob": self.masked_event_prob,
            "mask_span_length": self.mask_span_length,
            "temperature": self.temperature,
            "projection_dim": self.projection_dim,
            "num_attention_heads": self.num_attention_heads,
            "feedforward_dim": self.feedforward_dim,
            "time_mask_prob": self.time_mask_prob,
            "feature_dropout_prob": self.feature_dropout_prob,
            "noise_std": self.noise_std,
            "masked_prediction_weight": self.masked_prediction_weight,
            "contrastive_weight": self.contrastive_weight,
        }

    def _encode_from_hidden(self, token_hidden: torch.Tensor, batch_mask: torch.Tensor) -> torch.Tensor:
        batch_size = token_hidden.shape[0]
        cls = self.cls_token.expand(batch_size, -1, -1)
        sequence = torch.cat([cls, token_hidden], dim=1)
        positions = torch.arange(self.window_length + 1, device=self.device)
        sequence = sequence + self.position_embedding(positions).unsqueeze(0)
        key_padding_mask = torch.cat(
            [
                torch.zeros((batch_size, 1), dtype=torch.bool, device=self.device),
                batch_mask <= 0,
            ],
            dim=1,
        )
        return self.transformer(sequence, src_key_padding_mask=key_padding_mask)

    def _encode_sequence(
        self,
        batch_X: torch.Tensor,
        batch_mask: torch.Tensor,
        masked_positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        token_hidden = self.input_encoder(batch_X, masked_event_positions=masked_positions)
        return self._encode_from_hidden(token_hidden, batch_mask)

    def _pool_hidden(self, hidden: torch.Tensor, batch_mask: torch.Tensor) -> torch.Tensor:
        if self.pooling_mode == "cls":
            return hidden[:, 0, :]
        token_hidden = hidden[:, 1:, :]
        return pool_sequence(token_hidden, batch_mask, self.pooling_mode, self.attention_pool)

    def _augment_encoded(self, encoded: torch.Tensor, batch_mask: torch.Tensor) -> torch.Tensor:
        augmented = encoded.clone()
        valid = batch_mask.unsqueeze(-1)
        if self.noise_std > 0:
            augmented = augmented + torch.randn_like(augmented) * self.noise_std * valid
        if self.feature_dropout_prob > 0:
            feature_keep = (
                torch.rand(augmented.size(0), 1, augmented.size(2), device=augmented.device) > self.feature_dropout_prob
            ).float()
            augmented = augmented * feature_keep
        if self.time_mask_prob > 0:
            time_keep = (
                torch.rand(augmented.size(0), augmented.size(1), 1, device=augmented.device) > self.time_mask_prob
            ).float()
            augmented = augmented * time_keep
        return augmented * valid

    def _masked_event_prediction_loss(self, batch_X: torch.Tensor, batch_mask: torch.Tensor) -> torch.Tensor:
        valid_positions = batch_mask > 0
        masked_positions = sample_mask_span_positions(
            valid_positions=valid_positions,
            masked_event_prob=self.masked_event_prob,
            mask_span_length=self.mask_span_length,
        )
        if not torch.any(masked_positions):
            return torch.tensor(0.0, device=self.device)

        hidden = self._encode_sequence(batch_X, batch_mask, masked_positions=masked_positions)
        token_hidden = hidden[:, 1:, :]
        loss_terms = []
        categorical_targets = {
            "event_type": batch_X[:, :, 0].long(),
            "component": batch_X[:, :, 1].long(),
            "severity": batch_X[:, :, 2].long(),
            "state": batch_X[:, :, 3].long(),
        }
        for name, labels in categorical_targets.items():
            logits = self.masked_event_heads[name](token_hidden[masked_positions])
            refs = labels[masked_positions]
            loss_terms.append(F.cross_entropy(logits, refs))

        numeric_pred = self.masked_event_heads["numeric"](token_hidden[masked_positions])
        numeric_target = batch_X[:, :, 4:][masked_positions]
        if numeric_pred.numel() > 0:
            loss_terms.append(F.mse_loss(numeric_pred, numeric_target))
        return torch.stack(loss_terms).mean()


def build_sequence_model(model_type: str, cfg: dict, input_dim: int, window_length: int):
    """Instantiate one learned sequence model from a flat YAML config."""

    common_kwargs = {
        "input_dim": input_dim,
        "window_length": window_length,
        "hidden_dim": int(cfg.get("hidden_dim", 64)),
        "depth": int(cfg.get("depth", 4)),
        "kernel_size": int(cfg.get("kernel_size", 3)),
        "dropout": float(cfg.get("dropout", 0.1)),
        "event_type_vocab_size": int(cfg["event_type_vocab_size"]),
        "component_vocab_size": int(cfg["component_vocab_size"]),
        "severity_vocab_size": int(cfg["severity_vocab_size"]),
        "state_vocab_size": int(cfg["state_vocab_size"]),
        "event_type_emb_dim": int(cfg.get("event_type_emb_dim", 16)),
        "component_emb_dim": int(cfg.get("component_emb_dim", 8)),
        "severity_emb_dim": int(cfg.get("severity_emb_dim", 8)),
        "state_emb_dim": int(cfg.get("state_emb_dim", 8)),
        "batch_size": int(cfg.get("batch_size", 64)),
        "learning_rate": float(cfg.get("learning_rate", 1e-3)),
        "weight_decay": float(cfg.get("weight_decay", 1e-4)),
        "max_epochs": int(cfg.get("max_epochs", 20)),
        "pooling_mode": str(cfg.get("pooling_mode", "mean")),
        "device": str(cfg.get("device", "cpu")),
    }
    if model_type == TS2VecStyleWindowEncoder.model_type:
        return TS2VecStyleWindowEncoder(
            projection_dim=int(cfg.get("projection_dim", 32)),
            temperature=float(cfg.get("temperature", 0.2)),
            time_mask_prob=float(cfg.get("time_mask_prob", 0.05)),
            feature_dropout_prob=float(cfg.get("feature_dropout_prob", 0.05)),
            noise_std=float(cfg.get("noise_std", 0.01)),
            temporal_positive_radius=int(cfg.get("temporal_positive_radius", 10)),
            masked_event_prob=float(cfg.get("masked_event_prob", 0.15)),
            masked_prediction_weight=float(cfg.get("masked_prediction_weight", 1.0)),
            mask_span_length=int(cfg.get("mask_span_length", 1)),
            order_prediction_weight=float(cfg.get("order_prediction_weight", 0.0)),
            order_span_length=int(cfg.get("order_span_length", 3)),
            hard_negative_weight=float(cfg.get("hard_negative_weight", 0.0)),
            hard_negative_margin=float(cfg.get("hard_negative_margin", 0.3)),
            hard_negative_far_radius=int(cfg.get("hard_negative_far_radius", 20)),
            **common_kwargs,
        )
    if model_type == SequenceAutoencoder.model_type:
        return SequenceAutoencoder(**common_kwargs)
    if model_type == MaskedEventAutoencoder.model_type:
        return MaskedEventAutoencoder(
            masked_event_prob=float(cfg.get("masked_event_prob", 0.15)),
            mask_span_length=int(cfg.get("mask_span_length", 1)),
            **common_kwargs,
        )
    if model_type == LogBERTStyleWindowEncoder.model_type:
        return LogBERTStyleWindowEncoder(
            masked_event_prob=float(cfg.get("masked_event_prob", 0.15)),
            mask_span_length=int(cfg.get("mask_span_length", 1)),
            num_attention_heads=int(cfg.get("num_attention_heads", 4)),
            feedforward_dim=int(cfg.get("feedforward_dim", 256)),
            **common_kwargs,
        )
    if model_type == TimeDRLStyleWindowEncoder.model_type:
        return TimeDRLStyleWindowEncoder(
            masked_event_prob=float(cfg.get("masked_event_prob", 0.15)),
            mask_span_length=int(cfg.get("mask_span_length", 1)),
            temperature=float(cfg.get("temperature", 0.15)),
            projection_dim=int(cfg.get("projection_dim", 128)),
            num_attention_heads=int(cfg.get("num_attention_heads", 4)),
            feedforward_dim=int(cfg.get("feedforward_dim", 256)),
            time_mask_prob=float(cfg.get("time_mask_prob", 0.05)),
            feature_dropout_prob=float(cfg.get("feature_dropout_prob", 0.05)),
            noise_std=float(cfg.get("noise_std", 0.01)),
            masked_prediction_weight=float(cfg.get("masked_prediction_weight", 1.0)),
            contrastive_weight=float(cfg.get("contrastive_weight", 1.0)),
            **common_kwargs,
        )
    raise ValueError(f"Unknown model_type: {model_type}")


def load_sequence_model(path: str | Path, model_type: str, device: str = "cpu"):
    """Load one learned sequence model from disk."""

    if model_type == TS2VecStyleWindowEncoder.model_type:
        return TS2VecStyleWindowEncoder.load(path, device=device)
    if model_type == SequenceAutoencoder.model_type:
        return SequenceAutoencoder.load(path, device=device)
    if model_type == MaskedEventAutoencoder.model_type:
        return MaskedEventAutoencoder.load(path, device=device)
    if model_type == LogBERTStyleWindowEncoder.model_type:
        return LogBERTStyleWindowEncoder.load(path, device=device)
    if model_type == TimeDRLStyleWindowEncoder.model_type:
        return TimeDRLStyleWindowEncoder.load(path, device=device)
    raise ValueError(f"Unknown model_type: {model_type}")
