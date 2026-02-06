"""
Dataset Module

PyTorch datasets for log sequences.
"""

import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple, List


class LogDataset(Dataset):
    """PyTorch dataset for log sequences."""
    
    def __init__(self,
                 features: Dict[str, np.ndarray],
                 sequence_length: int = 100,
                 stride: Optional[int] = None,
                 include_timestamps: bool = True,
                 include_severity: bool = True,
                 normalize_time: bool = True):
        """Initialize dataset.
        
        Args:
            features: Dictionary of feature arrays from LogParser.extract_features
            sequence_length: Length of each sequence
            stride: Stride for sliding window (defaults to sequence_length)
            include_timestamps: Whether to include temporal features
            include_severity: Whether to include severity information
            normalize_time: Whether to normalize time deltas
        """
        self.features = features
        self.sequence_length = sequence_length
        self.stride = stride if stride is not None else sequence_length
        self.include_timestamps = include_timestamps
        self.include_severity = include_severity
        self.normalize_time = normalize_time
        
        # Create sequences using sliding window
        self.sequences = self._create_sequences()
        
        # Normalization statistics
        if normalize_time and "time_deltas" in features:
            self.time_mean = np.mean(features["time_deltas"])
            self.time_std = np.std(features["time_deltas"]) + 1e-8
        
    def _create_sequences(self) -> List[int]:
        """Create sequence indices using sliding window.
        
        Returns:
            List of starting indices for each sequence
        """
        total_events = len(self.features["events"])
        indices = []
        
        for start in range(0, total_events - self.sequence_length + 1, self.stride):
            indices.append(start)
        
        return indices
    
    def __len__(self) -> int:
        """Return number of sequences."""
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a sequence.
        
        Args:
            idx: Sequence index
            
        Returns:
            Dictionary containing:
                - events: Event token IDs [sequence_length]
                - time_deltas: Time between events [sequence_length] (optional)
                - severity: Severity levels [sequence_length] (optional)
                - component: Component IDs [sequence_length] (optional)
        """
        start_idx = self.sequences[idx]
        end_idx = start_idx + self.sequence_length
        
        sample = {}
        
        # Event sequence
        events = self.features["events"][start_idx:end_idx]
        sample["events"] = torch.tensor(events, dtype=torch.long)
        
        # Temporal features
        if self.include_timestamps and "time_deltas" in self.features:
            time_deltas = self.features["time_deltas"][start_idx:end_idx]
            if self.normalize_time:
                time_deltas = (time_deltas - self.time_mean) / self.time_std
            sample["time_deltas"] = torch.tensor(time_deltas, dtype=torch.float32)
        
        # Severity
        if self.include_severity and "severity" in self.features:
            severity = self.features["severity"][start_idx:end_idx]
            sample["severity"] = torch.tensor(severity, dtype=torch.long)
        
        # Component
        if "component" in self.features:
            component = self.features["component"][start_idx:end_idx]
            sample["component"] = torch.tensor(component, dtype=torch.long)
        
        return sample


class MaskedLogDataset(LogDataset):
    """Dataset with random masking for self-supervised learning."""
    
    def __init__(self, 
                 features: Dict[str, np.ndarray],
                 mask_prob: float = 0.15,
                 mask_token_id: int = 4,  # Assumes <MASK> is at index 4
                 **kwargs):
        """Initialize masked dataset.
        
        Args:
            features: Dictionary of feature arrays
            mask_prob: Probability of masking each token
            mask_token_id: Token ID for mask token
            **kwargs: Arguments for parent LogDataset
        """
        super().__init__(features, **kwargs)
        self.mask_prob = mask_prob
        self.mask_token_id = mask_token_id
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a sequence with random masking.
        
        Returns:
            Dictionary with additional keys:
                - masked_events: Events with some tokens masked
                - mask: Boolean mask indicating which tokens were masked
        """
        sample = super().__getitem__(idx)
        
        # Create mask
        events = sample["events"].clone()
        mask = torch.rand(self.sequence_length) < self.mask_prob
        
        # Apply masking
        masked_events = events.clone()
        masked_events[mask] = self.mask_token_id
        
        sample["original_events"] = events
        sample["masked_events"] = masked_events
        sample["mask"] = mask
        
        return sample


def create_dataloaders(features: Dict[str, np.ndarray],
                      config: Dict,
                      parser) -> Tuple:
    """Create train/val/test dataloaders.
    
    Args:
        features: Dictionary of feature arrays
        config: Configuration dictionary
        parser: LogParser instance
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    from torch.utils.data import DataLoader, random_split
    
    # Create full dataset
    dataset = LogDataset(
        features=features,
        sequence_length=config["data"]["sequence_length"],
        stride=config["data"]["stride"],
    )
    
    # Split into train/val/test
    total_size = len(dataset)
    train_size = int(config["data"]["train_split"] * total_size)
    val_size = int(config["data"]["val_split"] * total_size)
    test_size = total_size - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(config["training"]["seed"])
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=0,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=0,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=0,
    )
    
    return train_loader, val_loader, test_loader
