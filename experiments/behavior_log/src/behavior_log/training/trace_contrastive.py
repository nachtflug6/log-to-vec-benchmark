"""SSL training components for trace-level event sequences."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.feature_extraction.text import TfidfVectorizer

from behavior_log.data.trace_data import tokenize_sequence


@dataclass
class TraceTrainingResult:
    checkpoint_path: str
    training_summary: dict[str, Any]


class TraceSequenceDataset(Dataset):
    def __init__(
        self,
        traces_df: pd.DataFrame,
        *,
        token_vocab: dict[str, int],
        max_len: int,
    ) -> None:
        self.sample_ids = traces_df["sample_id"].astype(str).tolist()
        self.labels = traces_df["label"].astype(str).tolist()
        self.max_len = max_len
        self.pad_id = token_vocab["[PAD]"]
        self.unk_id = token_vocab["[UNK]"]
        self.sequences = [
            self._encode_sequence(tokenize_sequence(sequence), token_vocab)
            for sequence in traces_df["sequence"].fillna("").astype(str)
        ]
        template_keys = [tuple(sequence[: self.max_len]) for sequence in self.sequences]
        template_vocab = {key: index for index, key in enumerate(sorted(set(template_keys)))}
        self.template_ids = [template_vocab[key] for key in template_keys]

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sequence = self.sequences[idx]
        length = min(len(sequence), self.max_len)
        input_ids = torch.full((self.max_len,), self.pad_id, dtype=torch.long)
        attention_mask = torch.zeros((self.max_len,), dtype=torch.float32)
        if length > 0:
            input_ids[:length] = torch.tensor(sequence[:length], dtype=torch.long)
            attention_mask[:length] = 1.0
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "row_index": torch.tensor(idx, dtype=torch.long),
            "sample_id": self.sample_ids[idx],
            "label": self.labels[idx],
            "template_id": torch.tensor(self.template_ids[idx], dtype=torch.long),
        }

    def _encode_sequence(self, tokens: list[str], token_vocab: dict[str, int]) -> list[int]:
        return [token_vocab.get(token, self.unk_id) for token in tokens]


class TokenSequenceViewGenerator:
    def __init__(
        self,
        *,
        pad_id: int,
        mask_id: int,
        token_mask_ratio: float,
        token_dropout_ratio: float,
    ) -> None:
        self.pad_id = pad_id
        self.mask_id = mask_id
        self.token_mask_ratio = token_mask_ratio
        self.token_dropout_ratio = token_dropout_ratio

    def __call__(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        view_ids = input_ids.clone()
        view_mask = attention_mask.clone()
        valid_positions = view_mask > 0

        if self.token_dropout_ratio > 0:
            drop = (torch.rand_like(view_mask) < self.token_dropout_ratio) & valid_positions
            view_mask = torch.where(drop, torch.zeros_like(view_mask), view_mask)
            view_ids = torch.where(drop, torch.full_like(view_ids, self.pad_id), view_ids)

            empty_rows = view_mask.sum(dim=1) == 0
            if torch.any(empty_rows):
                first_valid = torch.argmax(valid_positions.float(), dim=1)
                rows = torch.where(empty_rows)[0]
                cols = first_valid[rows]
                view_mask[rows, cols] = 1.0
                view_ids[rows, cols] = input_ids[rows, cols]

        valid_positions = view_mask > 0
        if self.token_mask_ratio > 0:
            mask = (torch.rand_like(view_mask) < self.token_mask_ratio) & valid_positions
            view_ids = torch.where(mask, torch.full_like(view_ids, self.mask_id), view_ids)

        return view_ids, view_mask


class MaskedEventViewGenerator:
    def __init__(self, *, mask_id: int, mask_ratio: float) -> None:
        self.mask_id = mask_id
        self.mask_ratio = mask_ratio

    def __call__(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        corrupted = input_ids.clone()
        labels = torch.full_like(input_ids, -100)
        valid_positions = attention_mask > 0
        mask_positions = (torch.rand_like(attention_mask) < self.mask_ratio) & valid_positions

        empty_rows = mask_positions.sum(dim=1) == 0
        if torch.any(empty_rows):
            first_valid = torch.argmax(valid_positions.float(), dim=1)
            rows = torch.where(empty_rows & (valid_positions.sum(dim=1) > 0))[0]
            cols = first_valid[rows]
            mask_positions[rows, cols] = True

        labels = torch.where(mask_positions, input_ids, labels)
        corrupted = torch.where(mask_positions, torch.full_like(corrupted, self.mask_id), corrupted)
        return corrupted, labels


class ResidualTCNBlock(nn.Module):
    def __init__(self, hidden_dim: int, *, kernel_size: int, dilation: int, dropout: float) -> None:
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2
        self.net = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size, padding=padding, dilation=dilation),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size, padding=padding, dilation=dilation),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        y = self.net(x.transpose(1, 2)).transpose(1, 2)
        return self.norm(F.relu(y + residual))


def masked_mean(hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1)
    return (hidden * mask).sum(dim=1) / torch.clamp(mask.sum(dim=1), min=1.0)


def masked_max(hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    expanded_mask = attention_mask.unsqueeze(-1).bool()
    filled = hidden.masked_fill(~expanded_mask, float("-inf"))
    pooled = torch.max(filled, dim=1).values
    return torch.where(torch.isfinite(pooled), pooled, torch.zeros_like(pooled))


def pool_hidden(hidden: torch.Tensor, attention_mask: torch.Tensor, pooling: str) -> torch.Tensor:
    if pooling == "mean":
        return masked_mean(hidden, attention_mask)
    if pooling == "max":
        return masked_max(hidden, attention_mask)
    if pooling == "mean_max":
        return torch.cat([masked_mean(hidden, attention_mask), masked_max(hidden, attention_mask)], dim=1)
    raise ValueError(f"Unsupported pooling mode: {pooling}")


class TraceTCNBackbone(nn.Module):
    def __init__(
        self,
        *,
        vocab_size: int,
        hidden_dim: int,
        embedding_dim: int,
        padding_idx: int,
        max_len: int,
        pooling: str,
        depth: int = 3,
        kernel_size: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if pooling not in {"mean", "max", "mean_max"}:
            raise ValueError("TraceTCNBackbone supports pooling in {'mean', 'max', 'mean_max'}.")
        self.pooling = pooling
        self.token_embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx=padding_idx)
        self.position_embedding = nn.Embedding(max_len, hidden_dim)
        self.blocks = nn.ModuleList(
            [
                ResidualTCNBlock(
                    hidden_dim,
                    kernel_size=kernel_size,
                    dilation=2**layer_idx,
                    dropout=dropout,
                )
                for layer_idx in range(depth)
            ]
        )
        pooled_dim = hidden_dim * 2 if pooling == "mean_max" else hidden_dim
        self.projection = nn.Sequential(
            nn.Linear(pooled_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim),
        )
        self.hidden_size = hidden_dim

    def encode_hidden(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0)
        hidden = self.token_embedding(input_ids) + self.position_embedding(positions)
        hidden = hidden * attention_mask.unsqueeze(-1)
        for block in self.blocks:
            hidden = block(hidden)
            hidden = hidden * attention_mask.unsqueeze(-1)
        return hidden

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.encode_hidden(input_ids, attention_mask)
        pooled = pool_hidden(hidden, attention_mask, self.pooling)
        return hidden, self.projection(pooled)


class TraceBiGRUBackbone(nn.Module):
    def __init__(
        self,
        *,
        vocab_size: int,
        hidden_dim: int,
        embedding_dim: int,
        padding_idx: int,
        max_len: int,
        pooling: str,
        depth: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if pooling not in {"mean", "max", "mean_max"}:
            raise ValueError("TraceBiGRUBackbone supports pooling in {'mean', 'max', 'mean_max'}.")
        if hidden_dim % 2 != 0:
            raise ValueError("BiGRU hidden_dim must be even because it is split across directions.")
        self.pooling = pooling
        self.token_embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx=padding_idx)
        self.position_embedding = nn.Embedding(max_len, hidden_dim)
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim // 2,
            num_layers=depth,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if depth > 1 else 0.0,
        )
        pooled_dim = hidden_dim * 2 if pooling == "mean_max" else hidden_dim
        self.projection = nn.Sequential(
            nn.Linear(pooled_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim),
        )
        self.hidden_size = hidden_dim

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        positions = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0)
        embedded = self.token_embedding(input_ids) + self.position_embedding(positions)
        embedded = embedded * attention_mask.unsqueeze(-1)
        hidden, _ = self.gru(embedded)
        hidden = hidden * attention_mask.unsqueeze(-1)
        pooled = pool_hidden(hidden, attention_mask, self.pooling)
        return hidden, self.projection(pooled)


class TraceTransformerBackbone(nn.Module):
    def __init__(
        self,
        *,
        vocab_size: int,
        hidden_dim: int,
        embedding_dim: int,
        padding_idx: int,
        max_len: int,
        pooling: str,
        depth: int = 2,
        dropout: float = 0.1,
        nhead: int = 4,
        feedforward_dim: int | None = None,
    ) -> None:
        super().__init__()
        if pooling not in {"mean", "max", "mean_max"}:
            raise ValueError("TraceTransformerBackbone supports pooling in {'mean', 'max', 'mean_max'}.")
        if hidden_dim % nhead != 0:
            raise ValueError("Transformer hidden_dim must be divisible by nhead.")
        self.pooling = pooling
        self.token_embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx=padding_idx)
        self.position_embedding = nn.Embedding(max_len, hidden_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=int(feedforward_dim or hidden_dim * 4),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=depth)
        pooled_dim = hidden_dim * 2 if pooling == "mean_max" else hidden_dim
        self.projection = nn.Sequential(
            nn.Linear(pooled_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim),
        )
        self.hidden_size = hidden_dim

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        positions = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0)
        hidden = self.token_embedding(input_ids) + self.position_embedding(positions)
        key_padding_mask = attention_mask <= 0
        hidden = self.encoder(hidden, src_key_padding_mask=key_padding_mask)
        hidden = hidden * attention_mask.unsqueeze(-1)
        pooled = pool_hidden(hidden, attention_mask, self.pooling)
        return hidden, self.projection(pooled)


class TraceSSLModel(nn.Module):
    def __init__(self, *, backbone: nn.Module, vocab_size: int) -> None:
        super().__init__()
        self.backbone = backbone
        hidden_size = int(getattr(backbone, "hidden_size"))
        self.reconstruction_head = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden, embedding = self.backbone(input_ids, attention_mask)
        logits = self.reconstruction_head(hidden)
        return logits, embedding


class TraceSeq2SeqAutoencoder(nn.Module):
    """GRU encoder-decoder autoencoder with a single sequence embedding bottleneck."""

    def __init__(
        self,
        *,
        vocab_size: int,
        hidden_dim: int,
        embedding_dim: int,
        padding_idx: int,
        mask_idx: int,
        max_len: int,
        depth: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if hidden_dim % 2 != 0:
            raise ValueError("Seq2Seq GRU hidden_dim must be even because the encoder is bidirectional.")
        self.max_len = max_len
        self.depth = depth
        self.mask_idx = mask_idx
        self.token_embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx=padding_idx)
        self.position_embedding = nn.Embedding(max_len, hidden_dim)
        self.encoder = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim // 2,
            num_layers=depth,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if depth > 1 else 0.0,
        )
        self.to_embedding = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim),
        )
        self.decoder_init = nn.Linear(embedding_dim, depth * hidden_dim)
        self.decoder = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=depth,
            batch_first=True,
            dropout=dropout if depth > 1 else 0.0,
        )
        self.output_head = nn.Linear(hidden_dim, vocab_size)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len = input_ids.shape
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        embedded = self.token_embedding(input_ids) + self.position_embedding(positions)
        embedded = embedded * attention_mask.unsqueeze(-1)
        _, final_hidden = self.encoder(embedded)
        final_hidden = final_hidden.view(self.depth, 2, batch_size, -1)
        encoder_state = torch.cat([final_hidden[-1, 0], final_hidden[-1, 1]], dim=1)
        sequence_embedding = self.to_embedding(encoder_state)

        decoder_init = self.decoder_init(sequence_embedding).view(self.depth, batch_size, -1).contiguous()
        decoder_token_ids = torch.full_like(input_ids, self.mask_idx)
        decoder_inputs = self.token_embedding(decoder_token_ids) + self.position_embedding(positions)
        decoder_hidden, _ = self.decoder(decoder_inputs, decoder_init)
        logits = self.output_head(decoder_hidden)
        return logits, sequence_embedding


def build_trace_ssl_model(
    *,
    vocab_size: int,
    token_vocab: dict[str, int],
    encoder_cfg: dict[str, Any],
    max_len: int,
) -> TraceSSLModel:
    architecture = str(encoder_cfg["architecture"])
    common = {
        "vocab_size": vocab_size,
        "hidden_dim": int(encoder_cfg["hidden_dim"]),
        "embedding_dim": int(encoder_cfg["embedding_dim"]),
        "padding_idx": token_vocab["[PAD]"],
        "max_len": max_len,
        "pooling": str(encoder_cfg["pooling"]),
        "depth": int(encoder_cfg.get("depth", 3 if architecture == "tcn" else 1)),
        "dropout": float(encoder_cfg.get("dropout", 0.1)),
    }
    if architecture == "tcn":
        backbone = TraceTCNBackbone(
            **common,
            kernel_size=int(encoder_cfg.get("kernel_size", 3)),
        )
    elif architecture == "bigru":
        backbone = TraceBiGRUBackbone(**common)
    elif architecture == "transformer":
        backbone = TraceTransformerBackbone(
            **common,
            nhead=int(encoder_cfg.get("nhead", 4)),
            feedforward_dim=encoder_cfg.get("feedforward_dim"),
        )
    elif architecture == "seq2seq_gru":
        return TraceSeq2SeqAutoencoder(
            vocab_size=vocab_size,
            hidden_dim=int(encoder_cfg["hidden_dim"]),
            embedding_dim=int(encoder_cfg["embedding_dim"]),
            padding_idx=token_vocab["[PAD]"],
            mask_idx=token_vocab["[MASK]"],
            max_len=max_len,
            depth=int(encoder_cfg.get("depth", 1)),
            dropout=float(encoder_cfg.get("dropout", 0.1)),
        )
    else:
        raise ValueError(f"Unsupported encoder architecture: {architecture}")
    return TraceSSLModel(backbone=backbone, vocab_size=vocab_size)


def nt_xent_loss(z1: torch.Tensor, z2: torch.Tensor, *, temperature: float) -> torch.Tensor:
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    logits = torch.matmul(z1, z2.transpose(0, 1)) / temperature
    labels = torch.arange(z1.shape[0], device=z1.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.transpose(0, 1), labels))


def template_aware_contrastive_loss(
    z1: torch.Tensor,
    z2: torch.Tensor,
    template_ids: torch.Tensor,
    *,
    temperature: float,
) -> torch.Tensor:
    """Supervised contrastive loss where positives share the same sequence template."""

    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    reps = torch.cat([z1, z2], dim=0)
    labels = torch.cat([template_ids, template_ids], dim=0)
    total = reps.shape[0]

    logits = torch.matmul(reps, reps.transpose(0, 1)) / temperature
    identity = torch.eye(total, dtype=torch.bool, device=reps.device)
    positive_mask = labels.unsqueeze(0) == labels.unsqueeze(1)
    positive_mask = positive_mask & ~identity

    logits = logits.masked_fill(identity, -1e9)
    log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)

    positive_counts = positive_mask.sum(dim=1)
    valid = positive_counts > 0
    per_anchor_loss = -(positive_mask.float() * log_prob).sum(dim=1) / torch.clamp(
        positive_counts.float(),
        min=1.0,
    )
    return per_anchor_loss[valid].mean()


def similarity_aware_contrastive_loss(
    z1: torch.Tensor,
    z2: torch.Tensor,
    target_similarity: torch.Tensor,
    *,
    temperature: float,
    target_temperature: float,
    target_top_k: int,
) -> torch.Tensor:
    """Match embedding similarities to soft sequence-similarity targets."""

    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    reps = torch.cat([z1, z2], dim=0)
    target = torch.cat(
        [
            torch.cat([target_similarity, target_similarity], dim=1),
            torch.cat([target_similarity, target_similarity], dim=1),
        ],
        dim=0,
    )
    total = reps.shape[0]
    identity = torch.eye(total, dtype=torch.bool, device=reps.device)

    model_logits = torch.matmul(reps, reps.transpose(0, 1)) / temperature
    model_logits = model_logits.masked_fill(identity, -1e9)

    target_logits = target / target_temperature
    target_logits = target_logits.masked_fill(identity, -1e9)
    if target_top_k > 0 and target_top_k < total - 1:
        keep = torch.zeros_like(target_logits, dtype=torch.bool)
        topk_indices = torch.topk(target_logits, k=target_top_k, dim=1).indices
        keep.scatter_(1, topk_indices, True)
        target_logits = target_logits.masked_fill(~keep, -1e9)

    target_probs = torch.softmax(target_logits, dim=1)
    log_probs = torch.log_softmax(model_logits, dim=1)
    return -(target_probs * log_probs).sum(dim=1).mean()


def build_sequence_tfidf_matrix(
    traces_df: pd.DataFrame,
    *,
    ngram_min: int,
    ngram_max: int,
):
    docs = []
    for sequence in traces_df["sequence"].fillna("").astype(str):
        tokens = tokenize_sequence(sequence)
        docs.append(" ".join(_encode_tfidf_token(token) for token in tokens))
    vectorizer = TfidfVectorizer(
        analyzer="word",
        token_pattern=r"[^ ]+",
        lowercase=False,
        ngram_range=(ngram_min, ngram_max),
    )
    return vectorizer.fit_transform(docs)


def _encode_tfidf_token(token: str) -> str:
    return token.replace("\\", "\\\\").replace(" ", "\\s")


def train_hdfs_contrastive_encoder(
    *,
    train_df: pd.DataFrame,
    token_vocab: dict[str, int],
    cfg: dict[str, Any],
    output_dir: Path,
) -> TraceTrainingResult:
    optimization = cfg["optimization"]
    augmentation = cfg["augmentation"]
    encoder_cfg = cfg["encoder"]
    objective_cfg = cfg.get("objective", {})
    objective_type = str(objective_cfg.get("type", "contrastive"))
    max_len = int(cfg["max_len"])
    requested_device = str(cfg.get("device", "auto"))
    if requested_device == "auto":
        requested_device = "cuda" if torch.cuda.is_available() else "cpu"
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA was requested but is not available; falling back to CPU.")
        requested_device = "cpu"
    device = torch.device(requested_device)

    dataset = TraceSequenceDataset(train_df, token_vocab=token_vocab, max_len=max_len)
    positive_target = str(objective_cfg.get("positive_target", "instance"))
    sequence_tfidf_matrix = None
    if positive_target == "sequence_similarity":
        sequence_tfidf_matrix = build_sequence_tfidf_matrix(
            train_df,
            ngram_min=int(objective_cfg.get("similarity_ngram_min", 1)),
            ngram_max=int(objective_cfg.get("similarity_ngram_max", 3)),
        )
    generator = torch.Generator()
    generator.manual_seed(int(cfg.get("seed", 42)))
    loader = DataLoader(
        dataset,
        batch_size=int(optimization["batch_size"]),
        shuffle=True,
        drop_last=True,
        generator=generator,
    )

    model = build_trace_ssl_model(
        vocab_size=len(token_vocab),
        token_vocab=token_vocab,
        encoder_cfg=encoder_cfg,
        max_len=max_len,
    ).to(device)
    view1_generator = TokenSequenceViewGenerator(
        pad_id=token_vocab["[PAD]"],
        mask_id=token_vocab["[MASK]"],
        token_mask_ratio=float(augmentation.get("view1_token_mask_ratio", augmentation["token_mask_ratio"])),
        token_dropout_ratio=float(augmentation.get("view1_token_dropout_ratio", augmentation["token_dropout_ratio"])),
    )
    view2_generator = TokenSequenceViewGenerator(
        pad_id=token_vocab["[PAD]"],
        mask_id=token_vocab["[MASK]"],
        token_mask_ratio=float(augmentation.get("view2_token_mask_ratio", augmentation["token_mask_ratio"])),
        token_dropout_ratio=float(augmentation.get("view2_token_dropout_ratio", augmentation["token_dropout_ratio"])),
    )
    reconstruction_generator = MaskedEventViewGenerator(
        mask_id=token_vocab["[MASK]"],
        mask_ratio=float(augmentation.get("reconstruction_mask_ratio", augmentation["token_mask_ratio"])),
    )
    denoising_generator = TokenSequenceViewGenerator(
        pad_id=token_vocab["[PAD]"],
        mask_id=token_vocab["[MASK]"],
        token_mask_ratio=float(augmentation.get("denoising_token_mask_ratio", augmentation.get("token_mask_ratio", 0.0))),
        token_dropout_ratio=float(augmentation.get("denoising_token_dropout_ratio", augmentation.get("token_dropout_ratio", 0.0))),
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(optimization["lr"]),
        weight_decay=float(optimization.get("weight_decay", 0.0)),
    )
    epochs = int(optimization["epochs"])
    temperature = float(optimization["temperature"])
    contrastive_weight = float(objective_cfg.get("contrastive_weight", 1.0 if objective_type in {"contrastive", "hybrid"} else 0.0))
    reconstruction_weight = float(
        objective_cfg.get("reconstruction_weight", 1.0 if objective_type in {"reconstruction", "hybrid", "autoencoder"} else 0.0)
    )
    losses: list[float] = []
    contrastive_losses: list[float] = []
    reconstruction_losses: list[float] = []

    model.train()
    for epoch in range(1, epochs + 1):
        epoch_losses: list[float] = []
        epoch_contrastive_losses: list[float] = []
        epoch_reconstruction_losses: list[float] = []
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            template_ids = batch["template_id"].to(device)
            row_indices = batch["row_index"].cpu().numpy()
            loss_terms: list[torch.Tensor] = []

            if contrastive_weight > 0:
                view1_ids, view1_mask = view1_generator(input_ids, attention_mask)
                view2_ids, view2_mask = view2_generator(input_ids, attention_mask)
                _, z1 = model(view1_ids, view1_mask)
                _, z2 = model(view2_ids, view2_mask)
                if positive_target == "template_id":
                    contrastive_loss = template_aware_contrastive_loss(
                        z1,
                        z2,
                        template_ids,
                        temperature=temperature,
                    )
                elif positive_target == "sequence_similarity":
                    if sequence_tfidf_matrix is None:
                        raise RuntimeError("sequence_tfidf_matrix was not initialized.")
                    batch_similarity = (
                        sequence_tfidf_matrix[row_indices] @ sequence_tfidf_matrix[row_indices].T
                    ).toarray()
                    target_similarity = torch.tensor(batch_similarity, dtype=torch.float32, device=device)
                    contrastive_loss = similarity_aware_contrastive_loss(
                        z1,
                        z2,
                        target_similarity,
                        temperature=temperature,
                        target_temperature=float(objective_cfg.get("target_temperature", 0.1)),
                        target_top_k=int(objective_cfg.get("target_top_k", 8)),
                    )
                else:
                    contrastive_loss = nt_xent_loss(z1, z2, temperature=temperature)
                loss_terms.append(contrastive_weight * contrastive_loss)
                epoch_contrastive_losses.append(float(contrastive_loss.detach().cpu().item()))

            if reconstruction_weight > 0:
                if objective_type == "autoencoder":
                    corrupted_ids, corrupted_mask = denoising_generator(input_ids, attention_mask)
                    reconstruction_labels = torch.where(
                        attention_mask > 0,
                        input_ids,
                        torch.full_like(input_ids, -100),
                    )
                    logits, _ = model(corrupted_ids, corrupted_mask)
                else:
                    masked_ids, reconstruction_labels = reconstruction_generator(input_ids, attention_mask)
                    logits, _ = model(masked_ids, attention_mask)
                reconstruction_loss = F.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]),
                    reconstruction_labels.reshape(-1),
                    ignore_index=-100,
                )
                loss_terms.append(reconstruction_weight * reconstruction_loss)
                epoch_reconstruction_losses.append(float(reconstruction_loss.detach().cpu().item()))

            if not loss_terms:
                raise ValueError("At least one of contrastive_weight or reconstruction_weight must be positive.")
            loss = sum(loss_terms)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(optimization.get("max_grad_norm", 1.0)))
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu().item()))

        mean_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0
        mean_contrastive_loss = float(np.mean(epoch_contrastive_losses)) if epoch_contrastive_losses else 0.0
        mean_reconstruction_loss = float(np.mean(epoch_reconstruction_losses)) if epoch_reconstruction_losses else 0.0
        losses.append(mean_loss)
        contrastive_losses.append(mean_contrastive_loss)
        reconstruction_losses.append(mean_reconstruction_loss)
        print(
            f"epoch={epoch} loss={mean_loss:.6f} "
            f"contrastive={mean_contrastive_loss:.6f} reconstruction={mean_reconstruction_loss:.6f}"
        )

    checkpoint_path = output_dir / str(cfg.get("model_file", "model.pt"))
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "token_vocab": token_vocab,
        "config": cfg,
        "model_class": "TraceSSLModel",
        "encoder_architecture": str(encoder_cfg["architecture"]),
        "objective_type": objective_type,
    }
    torch.save(checkpoint, checkpoint_path)

    summary = {
        "status": "completed",
        "device": str(device),
        "epochs": epochs,
        "objective_type": objective_type,
        "contrastive_weight": contrastive_weight,
        "reconstruction_weight": reconstruction_weight,
        "train_samples": len(dataset),
        "steps_per_epoch": len(loader),
        "loss_history": losses,
        "contrastive_loss_history": contrastive_losses,
        "reconstruction_loss_history": reconstruction_losses,
        "final_loss": losses[-1] if losses else None,
        "checkpoint_path": str(checkpoint_path),
    }
    return TraceTrainingResult(checkpoint_path=str(checkpoint_path), training_summary=summary)
