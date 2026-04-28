"""
Encoder-only models for contrastive learning based objectives
on numerical sequences (B, T, D).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ProjectionHeadMLP(nn.Module):
    """
    Small MLP used to project encoder representations
    into a space where contrastive loss is applied.
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)


class LSTMContrastiveEncoder(nn.Module):
    """
    LSTM encoder for numerical sequence inputs.

    Input:
        x: (B, T, D)

    Output:
        dict:
            "embeddings": h (B, E)
            "projections": z (B, P)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        projection_dim: int = 64,
        dropout: float = 0.1
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )

        # Final representation dimension
        self.embedding_dim = hidden_dim

        # Projection head
        self.projection_head = ProjectionHeadMLP(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            output_dim=projection_dim
        )

    def encode(self, x):
        """
        x: (B, T, D)
        return: h (B, hidden_dim)
        """
        outputs, (h_n, c_n) = self.lstm(x)

        # Use last layer's hidden state
        h = h_n[-1]  # (B, hidden_dim)

        return h

    def forward(self, x):
        """
        x: (B, T, D)
        """
        h = self.encode(x)
        z = self.projection_head(h)

        # Normalize for cosine similarity
        z = F.normalize(z, dim=1)

        return {
            "embeddings": h,
            "projections": z
        }