"""
Autoencoder-based models for log sequence embeddings.

Implements reconstruction-based objectives.
"""

import torch
import torch.nn as nn
from typing import Dict
from .base import BaseEmbeddingModel, PositionalEncoding, TemporalEncoding


class LSTMAutoencoder(BaseEmbeddingModel):
    """LSTM-based autoencoder for log sequences."""
    
    def __init__(self,
                 vocab_size: int,
                 embedding_dim: int,
                 hidden_dim: int,
                 num_layers: int = 2,
                 dropout: float = 0.1,
                 use_temporal: bool = True,
                 padding_idx: int = 0):
        """Initialize LSTM autoencoder.
        
        Args:
            vocab_size: Size of vocabulary
            embedding_dim: Dimension of token embeddings
            hidden_dim: Dimension of LSTM hidden states
            num_layers: Number of LSTM layers
            dropout: Dropout probability
            use_temporal: Whether to use temporal (time delta) features
            padding_idx: Index for padding token
        """
        super().__init__(vocab_size, embedding_dim, padding_idx)
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.use_temporal = use_temporal
        
        # Temporal encoding
        if use_temporal:
            self.temporal_encoding = TemporalEncoding(embedding_dim, dropout)
        
        # Encoder LSTM
        self.encoder = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False
        )
        
        # Decoder LSTM
        self.decoder = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False
        )
        
        # Output projection to vocabulary
        self.output_projection = nn.Linear(hidden_dim, vocab_size)
        
    def encode(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Encode sequences to fixed-size embeddings.
        
        Args:
            batch: Dictionary with 'events' and optionally 'time_deltas'
            
        Returns:
            Embeddings [batch_size, hidden_dim]
        """
        events = batch["events"]  # [batch_size, seq_len]
        
        # Embed tokens
        embedded = self.token_embedding(events)  # [batch_size, seq_len, embedding_dim]
        
        # Add temporal information
        if self.use_temporal and "time_deltas" in batch:
            time_encoded = self.temporal_encoding(batch["time_deltas"])
            embedded = embedded + time_encoded
        
        # Encode with LSTM
        _, (hidden, _) = self.encoder(embedded)
        
        # Use final hidden state as embedding (from last layer)
        embedding = hidden[-1]  # [batch_size, hidden_dim]
        
        return embedding
    
    def decode(self, 
               hidden_state: torch.Tensor,
               target_length: int,
               teacher_forcing_events: torch.Tensor = None) -> torch.Tensor:
        """Decode from embedding to sequence.
        
        Args:
            hidden_state: Encoder hidden state [batch_size, hidden_dim]
            target_length: Length of sequence to generate
            teacher_forcing_events: Events for teacher forcing (optional)
            
        Returns:
            Logits [batch_size, seq_len, vocab_size]
        """
        batch_size = hidden_state.size(0)
        device = hidden_state.device
        
        # Prepare decoder hidden state
        decoder_hidden = hidden_state.unsqueeze(0).repeat(self.num_layers, 1, 1)
        decoder_cell = torch.zeros_like(decoder_hidden)
        
        if teacher_forcing_events is not None:
            # Teacher forcing: use ground truth as input
            decoder_input = self.token_embedding(teacher_forcing_events)
            decoder_output, _ = self.decoder(decoder_input, (decoder_hidden, decoder_cell))
            logits = self.output_projection(decoder_output)
        else:
            # Autoregressive generation (for inference)
            # Start with padding token
            current_input = torch.full(
                (batch_size, 1), 
                self.padding_idx, 
                dtype=torch.long, 
                device=device
            )
            
            outputs = []
            for _ in range(target_length):
                embedded = self.token_embedding(current_input)
                decoder_output, (decoder_hidden, decoder_cell) = self.decoder(
                    embedded, (decoder_hidden, decoder_cell)
                )
                logit = self.output_projection(decoder_output)
                outputs.append(logit)
                
                # Use predicted token as next input
                current_input = logit.argmax(dim=-1)
            
            logits = torch.cat(outputs, dim=1)
        
        return logits
    
    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Forward pass: encode and reconstruct.
        
        Args:
            batch: Dictionary containing 'events' and optionally 'time_deltas'
            
        Returns:
            Dictionary with 'embeddings' and 'logits'
        """
        events = batch["events"]
        seq_len = events.size(1)
        
        # Encode
        embeddings = self.encode(batch)
        
        # Decode with teacher forcing
        logits = self.decode(embeddings, seq_len, teacher_forcing_events=events)
        
        return {
            "embeddings": embeddings,
            "logits": logits,
        }


class TransformerAutoencoder(BaseEmbeddingModel):
    """Transformer-based autoencoder for log sequences."""
    
    def __init__(self,
                 vocab_size: int,
                 embedding_dim: int,
                 num_heads: int = 8,
                 num_encoder_layers: int = 4,
                 num_decoder_layers: int = 4,
                 dim_feedforward: int = 512,
                 dropout: float = 0.1,
                 use_temporal: bool = True,
                 padding_idx: int = 0):
        """Initialize Transformer autoencoder.
        
        Args:
            vocab_size: Size of vocabulary
            embedding_dim: Dimension of embeddings
            num_heads: Number of attention heads
            num_encoder_layers: Number of encoder layers
            num_decoder_layers: Number of decoder layers
            dim_feedforward: Dimension of feedforward network
            dropout: Dropout probability
            use_temporal: Whether to use temporal features
            padding_idx: Index for padding token
        """
        super().__init__(vocab_size, embedding_dim, padding_idx)
        
        self.use_temporal = use_temporal
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(embedding_dim, dropout=dropout)
        
        # Temporal encoding
        if use_temporal:
            self.temporal_encoding = TemporalEncoding(embedding_dim, dropout)
        
        # Transformer encoder-decoder
        self.transformer = nn.Transformer(
            d_model=embedding_dim,
            nhead=num_heads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        
        # Output projection
        self.output_projection = nn.Linear(embedding_dim, vocab_size)
        
        # Pooling for fixed-size embedding
        self.pooling = nn.AdaptiveAvgPool1d(1)
    
    def encode(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Encode sequences to fixed-size embeddings.
        
        Args:
            batch: Dictionary with 'events' and optionally 'time_deltas'
            
        Returns:
            Embeddings [batch_size, embedding_dim]
        """
        events = batch["events"]
        
        # Embed and add positional encoding
        embedded = self.token_embedding(events)
        embedded = self.pos_encoding(embedded)
        
        # Add temporal information
        if self.use_temporal and "time_deltas" in batch:
            time_encoded = self.temporal_encoding(batch["time_deltas"])
            embedded = embedded + time_encoded
        
        # Create padding mask
        padding_mask = (events == self.padding_idx)
        
        # Encode (we need to pass through encoder only)
        # Using transformer encoder directly
        encoder = self.transformer.encoder
        memory = encoder(embedded, src_key_padding_mask=padding_mask)
        
        # Pool to fixed-size embedding
        # Transpose for pooling: [batch, seq, dim] -> [batch, dim, seq]
        pooled = self.pooling(memory.transpose(1, 2)).squeeze(-1)
        
        return pooled
    
    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Forward pass: encode and reconstruct.
        
        Args:
            batch: Dictionary containing 'events' and optionally 'time_deltas'
            
        Returns:
            Dictionary with 'embeddings' and 'logits'
        """
        events = batch["events"]
        
        # Embed source
        src_embedded = self.token_embedding(events)
        src_embedded = self.pos_encoding(src_embedded)
        
        if self.use_temporal and "time_deltas" in batch:
            time_encoded = self.temporal_encoding(batch["time_deltas"])
            src_embedded = src_embedded + time_encoded
        
        # Target is same as source (autoencoding)
        tgt_embedded = src_embedded
        
        # Create masks
        src_padding_mask = (events == self.padding_idx)
        tgt_padding_mask = src_padding_mask
        
        # Forward through transformer
        output = self.transformer(
            src_embedded,
            tgt_embedded,
            src_key_padding_mask=src_padding_mask,
            tgt_key_padding_mask=tgt_padding_mask,
        )
        
        # Project to vocabulary
        logits = self.output_projection(output)
        
        # Get embeddings by pooling encoder output
        embeddings = self.encode(batch)
        
        return {
            "embeddings": embeddings,
            "logits": logits,
        }
