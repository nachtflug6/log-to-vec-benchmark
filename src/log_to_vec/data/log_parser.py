"""
Log Parser Module

Handles parsing and tokenization of time-stamped log files.
Converts raw log text into sequences of tokens suitable for embedding models.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
from collections import Counter
import re


class LogParser:
    """Parse and tokenize log files into sequences."""
    
    def __init__(self, 
                 vocab_size: Optional[int] = None,
                 special_tokens: Optional[List[str]] = None):
        """Initialize log parser.
        
        Args:
            vocab_size: Maximum vocabulary size (None for unlimited)
            special_tokens: List of special tokens (PAD, UNK, CLS, SEP, MASK)
        """
        if special_tokens is None:
            special_tokens = ["<PAD>", "<UNK>", "<CLS>", "<SEP>", "<MASK>"]
        
        self.special_tokens = special_tokens
        self.vocab_size = vocab_size
        self.token2idx = {}
        self.idx2token = {}
        self.event_counter = Counter()
        
        # Initialize special tokens
        for i, token in enumerate(special_tokens):
            self.token2idx[token] = i
            self.idx2token[i] = token
    
    def build_vocabulary(self, logs_df: pd.DataFrame, 
                        event_col: str = "event_type") -> None:
        """Build vocabulary from log events.
        
        Args:
            logs_df: DataFrame containing log events
            event_col: Column name containing event types
        """
        # Count event occurrences
        self.event_counter = Counter(logs_df[event_col].values)
        
        # Determine vocabulary size
        if self.vocab_size is not None:
            # Keep only top vocab_size - len(special_tokens) events
            max_events = self.vocab_size - len(self.special_tokens)
            most_common = self.event_counter.most_common(max_events)
        else:
            most_common = self.event_counter.most_common()
        
        # Build token mappings
        current_idx = len(self.special_tokens)
        for event, _ in most_common:
            if event not in self.token2idx:
                self.token2idx[event] = current_idx
                self.idx2token[current_idx] = event
                current_idx += 1
        
        print(f"Vocabulary size: {len(self.token2idx)}")
        print(f"Most common events: {most_common[:10]}")
    
    def encode_event(self, event: str) -> int:
        """Encode a single event to its token ID.
        
        Args:
            event: Event string
            
        Returns:
            Token ID (uses <UNK> if event not in vocabulary)
        """
        return self.token2idx.get(event, self.token2idx["<UNK>"])
    
    def decode_event(self, token_id: int) -> str:
        """Decode a token ID back to event string.
        
        Args:
            token_id: Token ID
            
        Returns:
            Event string
        """
        return self.idx2token.get(token_id, "<UNK>")
    
    def encode_sequence(self, events: List[str]) -> List[int]:
        """Encode a sequence of events to token IDs.
        
        Args:
            events: List of event strings
            
        Returns:
            List of token IDs
        """
        return [self.encode_event(event) for event in events]
    
    def decode_sequence(self, token_ids: List[int]) -> List[str]:
        """Decode a sequence of token IDs to event strings.
        
        Args:
            token_ids: List of token IDs
            
        Returns:
            List of event strings
        """
        return [self.decode_event(tid) for tid in token_ids]
    
    def parse_timestamps(self, logs_df: pd.DataFrame, 
                        timestamp_col: str = "timestamp") -> np.ndarray:
        """Parse and normalize timestamps.
        
        Args:
            logs_df: DataFrame containing logs
            timestamp_col: Column name for timestamps
            
        Returns:
            Array of normalized timestamps (seconds since first event)
        """
        timestamps = pd.to_datetime(logs_df[timestamp_col])
        # Convert to seconds since first event
        first_time = timestamps.min()
        relative_times = (timestamps - first_time).dt.total_seconds().values
        return relative_times
    
    def create_time_deltas(self, timestamps: np.ndarray) -> np.ndarray:
        """Compute time differences between consecutive events.
        
        Args:
            timestamps: Array of timestamps
            
        Returns:
            Array of time deltas
        """
        deltas = np.diff(timestamps, prepend=timestamps[0])
        return deltas
    
    def extract_features(self, logs_df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Extract various features from logs.
        
        Args:
            logs_df: DataFrame containing logs
            
        Returns:
            Dictionary of feature arrays
        """
        features = {}
        
        # Event sequence (encoded)
        if "event_type" in logs_df.columns:
            features["events"] = np.array(
                self.encode_sequence(logs_df["event_type"].tolist())
            )
        
        # Timestamps
        if "timestamp" in logs_df.columns:
            features["timestamps"] = self.parse_timestamps(logs_df)
            features["time_deltas"] = self.create_time_deltas(features["timestamps"])
        
        # Severity encoding
        if "severity" in logs_df.columns:
            severity_map = {"INFO": 0, "WARNING": 1, "ERROR": 2, "CRITICAL": 3}
            features["severity"] = np.array([
                severity_map.get(s, 0) for s in logs_df["severity"]
            ])
        
        # Component encoding (if exists)
        if "component" in logs_df.columns:
            unique_components = logs_df["component"].unique()
            component_map = {c: i for i, c in enumerate(unique_components)}
            features["component"] = np.array([
                component_map[c] for c in logs_df["component"]
            ])
        
        return features
    
    def save_vocabulary(self, filepath: str) -> None:
        """Save vocabulary to file.
        
        Args:
            filepath: Path to save vocabulary
        """
        vocab_df = pd.DataFrame([
            {"token": token, "id": idx, "count": self.event_counter.get(token, 0)}
            for token, idx in self.token2idx.items()
        ])
        vocab_df.to_csv(filepath, index=False)
        print(f"Vocabulary saved to {filepath}")
    
    def load_vocabulary(self, filepath: str) -> None:
        """Load vocabulary from file.
        
        Args:
            filepath: Path to vocabulary file
        """
        vocab_df = pd.read_csv(filepath)
        self.token2idx = dict(zip(vocab_df["token"], vocab_df["id"]))
        self.idx2token = {v: k for k, v in self.token2idx.items()}
        print(f"Vocabulary loaded from {filepath}")
