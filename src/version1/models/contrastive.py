"""
Defines encoder architectures used for contrastive representation learning
on numerical sequence data.

Input:
    sequence tensor of shape (B, T, D)

Output:
    embeddings  : representation vector (B, E) used for downstream tasks
    projections : projected vector (B, P) used for contrastive loss

The projection head follows the SimCLR design.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ProjectionHeadMLP(nn.Module):
    """
    Projection head used for contrastive learning.

    Maps encoder embeddings into a space where contrastive loss is applied.

    Architecture:
        Linear → ReLU → Linear
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LSTMContrastiveEncoder(nn.Module):
    """
    LSTM encoder for sequence data.

    Input:
        x: (B, T, D)

    Output dictionary:
        embeddings  : sequence representation (B, hidden_dim)
        projections : contrastive projection (B, projection_dim)
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

        self.embedding_dim = hidden_dim

        self.projection_head = ProjectionHeadMLP(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            output_dim=projection_dim
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode sequence into embedding.

        Args:
            x: (B, T, D)

        Returns:
            h: (B, hidden_dim)
        """

        _, (h_n, _) = self.lstm(x)

        # last layer hidden state
        h = h_n[-1]

        return h

    def forward(self, x: torch.Tensor):

        # encoder representation
        h = self.encode(x)

        # projection head
        z = self.projection_head(h)

        # normalize for cosine similarity
        z = F.normalize(z, dim=1)

        return {
            "embeddings": h,
            "projections": z
        }