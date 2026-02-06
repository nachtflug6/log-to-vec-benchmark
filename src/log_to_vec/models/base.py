"""
Base Model Interface

Abstract base class for all embedding models.
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, Optional


class BaseEmbeddingModel(nn.Module, ABC):
    """Abstract base class for log embedding models."""
    
    def __init__(self, 
                 vocab_size: int,
                 embedding_dim: int,
                 padding_idx: int = 0):
        """Initialize base model.
        
        Args:
            vocab_size: Size of vocabulary
            embedding_dim: Dimension of embeddings
            padding_idx: Index for padding token
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        
        # Token embeddings
        self.token_embedding = nn.Embedding(
            vocab_size, 
            embedding_dim,
            padding_idx=padding_idx
        )
    
    @abstractmethod
    def encode(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Encode a batch of sequences to embeddings.
        
        Args:
            batch: Dictionary containing input tensors
            
        Returns:
            Tensor of shape [batch_size, embedding_dim]
        """
        pass
    
    @abstractmethod
    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Forward pass through the model.
        
        Args:
            batch: Dictionary containing input tensors
            
        Returns:
            Dictionary containing model outputs
        """
        pass
    
    def get_embeddings(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Get embeddings for a batch (convenience method).
        
        Args:
            batch: Dictionary containing input tensors
            
        Returns:
            Embeddings tensor [batch_size, embedding_dim]
        """
        self.eval()
        with torch.no_grad():
            embeddings = self.encode(batch)
        return embeddings


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for sequence position."""
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        """Initialize positional encoding.
        
        Args:
            d_model: Dimension of the model
            max_len: Maximum sequence length
            dropout: Dropout probability
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model)
        )
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        
        # Register as buffer (not a parameter, but part of state)
        self.register_buffer("pe", pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input.
        
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            
        Returns:
            Tensor with positional encoding added
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TemporalEncoding(nn.Module):
    """Encode time deltas between events."""
    
    def __init__(self, d_model: int, dropout: float = 0.1):
        """Initialize temporal encoding.
        
        Args:
            d_model: Dimension of the model
            dropout: Dropout probability
        """
        super().__init__()
        self.time_projection = nn.Linear(1, d_model)
        self.dropout = nn.Dropout(p=dropout)
    
    def forward(self, time_deltas: torch.Tensor) -> torch.Tensor:
        """Encode time deltas.
        
        Args:
            time_deltas: Time deltas [batch_size, seq_len]
            
        Returns:
            Encoded time features [batch_size, seq_len, d_model]
        """
        # Reshape for linear layer
        time_features = time_deltas.unsqueeze(-1)  # [batch, seq_len, 1]
        time_encoded = self.time_projection(time_features)
        return self.dropout(time_encoded)
