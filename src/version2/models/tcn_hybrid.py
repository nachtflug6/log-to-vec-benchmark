from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ProjectionHeadMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ResidualDilatedBlock(nn.Module):
    """Residual dilated 1D conv block for short fixed windows."""

    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2

        self.conv1 = nn.Conv1d(
            channels, channels, kernel_size=kernel_size, dilation=dilation, padding=padding
        )
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(
            channels, channels, kernel_size=kernel_size, dilation=dilation, padding=padding
        )
        self.bn2 = nn.BatchNorm1d(channels)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out)
        out = self.dropout(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.dropout(out)

        out = out + residual
        out = self.act(out)
        return out


class TCNBackbone(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_blocks: int = 4,
        kernel_size: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_proj = nn.Conv1d(input_dim, hidden_dim, kernel_size=1)
        dilations = [2 ** i for i in range(num_blocks)]
        self.blocks = nn.ModuleList(
            [
                ResidualDilatedBlock(
                    channels=hidden_dim,
                    kernel_size=kernel_size,
                    dilation=d,
                    dropout=dropout,
                )
                for d in dilations
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [B, T, D] -> [B, D, T]
        x = x.transpose(1, 2)
        h = self.input_proj(x)
        for block in self.blocks:
            h = block(h)
        return h


class TCNHybridEncoder(nn.Module):
    """
    Unsupervised TCN encoder with:
    - embedding head for downstream representation
    - projection head for contrastive objective
    - reconstruction head for masked/partial reconstruction
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        embedding_dim: int = 128,
        projection_dim: int = 64,
        num_blocks: int = 4,
        kernel_size: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.projection_dim = projection_dim
        self.num_blocks = num_blocks
        self.kernel_size = kernel_size
        self.dropout = dropout

        self.backbone = TCNBackbone(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_blocks=num_blocks,
            kernel_size=kernel_size,
            dropout=dropout,
        )

        self.embedding_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, embedding_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.projection_head = ProjectionHeadMLP(
            input_dim=embedding_dim,
            hidden_dim=embedding_dim,
            output_dim=projection_dim,
        )

        self.reconstruction_head = nn.Conv1d(hidden_dim, input_dim, kernel_size=1)

    def _pool(self, h_seq: torch.Tensor) -> torch.Tensor:
        h_mean = h_seq.mean(dim=-1)
        h_max = h_seq.max(dim=-1).values
        return torch.cat([h_mean, h_max], dim=1)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        h_seq = self.backbone(x)
        pooled = self._pool(h_seq)
        emb = self.embedding_head(pooled)
        return emb

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        h_seq = self.backbone(x)
        pooled = self._pool(h_seq)
        emb = self.embedding_head(pooled)
        proj = F.normalize(self.projection_head(emb), dim=1)
        recon = self.reconstruction_head(h_seq).transpose(1, 2)
        return {
            "embeddings": emb,
            "projections": proj,
            "reconstructions": recon,
            "hidden_seq": h_seq,
        }
