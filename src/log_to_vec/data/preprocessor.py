"""
Log Preprocessor Module

Transforms log data into numerical feature vectors for machine learning.
Handles both categorical (encoded as integers) and continuous numerical features.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import json
import ast
import re
from collections import defaultdict


class LogPreprocessor:
    """Preprocess log data into numerical feature vectors."""
    
    def __init__(self):
        """Initialize the preprocessor with empty encoders."""
        # Categorical encoders - maps values to integer IDs
        self.categorical_encoders = {
            'event_type': {},
            'component': {},
            'severity': {},
            'state': {},
        }
        
        # Reverse mappings for decoding
        self.categorical_decoders = {
            'event_type': {},
            'component': {},
            'severity': {},
            'state': {},
        }
        
        # Normalization statistics for numerical features
        self.numerical_stats = {}
        
        # Feature configuration
        self.feature_names = []
        self.fitted = False
        
    def fit(self, logs_df: pd.DataFrame) -> 'LogPreprocessor':
        """Fit the preprocessor on training data.
        
        Builds categorical encoders and computes normalization statistics.
        
        Args:
            logs_df: DataFrame containing log data
            
        Returns:
            self for method chaining
        """
        print("Fitting preprocessor on training data...")
        
        # Build categorical encoders
        self._fit_categorical_encoders(logs_df)
        
        # Extract and fit numerical features
        numerical_features = self._extract_numerical_features(logs_df)
        self._fit_numerical_stats(numerical_features)
        
        # Build feature names list
        self._build_feature_names()
        
        self.fitted = True
        print(f"Preprocessor fitted. Total features: {len(self.feature_names)}")
        return self
    
    def _fit_categorical_encoders(self, logs_df: pd.DataFrame) -> None:
        """Build categorical encoders from data.
        
        Args:
            logs_df: DataFrame containing log data
        """
        # Event type encoding
        if 'event_type' in logs_df.columns:
            unique_events = logs_df['event_type'].unique()
            self.categorical_encoders['event_type'] = {
                event: idx for idx, event in enumerate(sorted(unique_events))
            }
            self.categorical_decoders['event_type'] = {
                idx: event for event, idx in self.categorical_encoders['event_type'].items()
            }
            print(f"  Event types: {len(unique_events)} unique values")
        
        # Component encoding
        if 'component' in logs_df.columns:
            unique_components = logs_df['component'].unique()
            self.categorical_encoders['component'] = {
                comp: idx for idx, comp in enumerate(sorted(unique_components))
            }
            self.categorical_decoders['component'] = {
                idx: comp for comp, idx in self.categorical_encoders['component'].items()
            }
            print(f"  Components: {len(unique_components)} unique values")
        
        # Severity encoding (with predefined order)
        if 'severity' in logs_df.columns:
            severity_order = ["INFO", "WARNING", "ERROR", "CRITICAL"]
            unique_severities = logs_df['severity'].unique()
            # Use predefined order, but add any unexpected values
            all_severities = severity_order + [s for s in unique_severities if s not in severity_order]
            self.categorical_encoders['severity'] = {
                sev: idx for idx, sev in enumerate(all_severities)
            }
            self.categorical_decoders['severity'] = {
                idx: sev for sev, idx in self.categorical_encoders['severity'].items()
            }
            print(f"  Severity levels: {len(all_severities)} unique values")
        
        # State encoding (extracted from message)
        if 'message' in logs_df.columns:
            states = self._extract_states(logs_df)
            unique_states = set(states)
            self.categorical_encoders['state'] = {
                state: idx for idx, state in enumerate(sorted(unique_states))
            }
            self.categorical_decoders['state'] = {
                idx: state for state, idx in self.categorical_encoders['state'].items()
            }
            print(f"  States: {len(unique_states)} unique values")
    
    def _extract_states(self, logs_df: pd.DataFrame) -> List[str]:
        """Extract state information from message field.
        
        Args:
            logs_df: DataFrame containing log data
            
        Returns:
            List of state strings
        """
        states = []
        for msg in logs_df['message']:
            # Look for pattern "in XXX state"
            match = re.search(r'in (\w+) state', str(msg))
            if match:
                states.append(match.group(1))
            else:
                states.append('UNKNOWN')
        return states
    
    def _extract_numerical_features(self, logs_df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Extract numerical features from data field.
        
        Args:
            logs_df: DataFrame containing log data
            
        Returns:
            Dictionary of numerical feature arrays
        """
        numerical_features = defaultdict(list)
        
        if 'data' not in logs_df.columns:
            return numerical_features
        
        for data_str in logs_df['data']:
            # Parse data field (it's a string representation of a dict)
            try:
                if pd.isna(data_str) or data_str == '' or data_str == '{}':
                    data_dict = {}
                else:
                    # Try to parse as dict literal
                    data_dict = ast.literal_eval(str(data_str))
            except (ValueError, SyntaxError):
                data_dict = {}
            
            # Extract numerical values
            numerical_features['temperature'].append(
                data_dict.get('temperature', np.nan)
            )
            numerical_features['pressure'].append(
                data_dict.get('pressure', np.nan)
            )
            numerical_features['position'].append(
                data_dict.get('position', np.nan)
            )
            numerical_features['value'].append(
                data_dict.get('value', np.nan)
            )
            
            # Boolean features (converted to 0/1)
            numerical_features['threshold_exceeded'].append(
                1.0 if data_dict.get('threshold_exceeded', False) else 0.0
            )
            
            # Actuator state as categorical
            actuator_state = data_dict.get('state', 'UNKNOWN')
            if actuator_state in ['OPEN', 'CLOSED', 'MOVING']:
                state_map = {'OPEN': 0, 'CLOSED': 1, 'MOVING': 2, 'UNKNOWN': 3}
                numerical_features['actuator_state'].append(
                    state_map.get(actuator_state, 3)
                )
            else:
                numerical_features['actuator_state'].append(3)
        
        # Convert to numpy arrays
        for key in numerical_features:
            numerical_features[key] = np.array(numerical_features[key])
        
        return numerical_features
    
    def _fit_numerical_stats(self, numerical_features: Dict[str, np.ndarray]) -> None:
        """Compute normalization statistics for numerical features.
        
        Args:
            numerical_features: Dictionary of numerical feature arrays
        """
        for feature_name, values in numerical_features.items():
            # Skip categorical features encoded as integers
            if feature_name in ['actuator_state', 'threshold_exceeded']:
                self.numerical_stats[feature_name] = {'mean': 0.0, 'std': 1.0}
                continue
            
            # Compute mean and std, ignoring NaN values
            valid_values = values[~np.isnan(values)]
            if len(valid_values) > 0:
                mean = np.mean(valid_values)
                std = np.std(valid_values)
                if std < 1e-8:  # Avoid division by zero
                    std = 1.0
            else:
                mean = 0.0
                std = 1.0
            
            self.numerical_stats[feature_name] = {'mean': mean, 'std': std}
            print(f"  {feature_name}: mean={mean:.3f}, std={std:.3f}")
    
    def _build_feature_names(self) -> None:
        """Build list of all feature names in order."""
        self.feature_names = []
        
        # Categorical features
        if self.categorical_encoders['event_type']:
            self.feature_names.append('event_type_id')
        if self.categorical_encoders['component']:
            self.feature_names.append('component_id')
        if self.categorical_encoders['severity']:
            self.feature_names.append('severity_id')
        if self.categorical_encoders['state']:
            self.feature_names.append('state_id')
        
        # Numerical features
        for feature_name in sorted(self.numerical_stats.keys()):
            self.feature_names.append(feature_name)
    
    def transform(self, logs_df: pd.DataFrame, normalize: bool = True) -> np.ndarray:
        """Transform log data into numerical feature vectors.
        
        Args:
            logs_df: DataFrame containing log data
            normalize: Whether to normalize numerical features
            
        Returns:
            numpy array of shape (n_samples, n_features)
        """
        if not self.fitted:
            raise ValueError("Preprocessor must be fitted before transform")
        
        n_samples = len(logs_df)
        n_features = len(self.feature_names)
        feature_matrix = np.zeros((n_samples, n_features))
        
        feature_idx = 0
        
        # Encode categorical features
        if 'event_type_id' in self.feature_names:
            for i, event in enumerate(logs_df['event_type']):
                feature_matrix[i, feature_idx] = self.categorical_encoders['event_type'].get(
                    event, -1  # Use -1 for unknown values
                )
            feature_idx += 1
        
        if 'component_id' in self.feature_names:
            for i, comp in enumerate(logs_df['component']):
                feature_matrix[i, feature_idx] = self.categorical_encoders['component'].get(
                    comp, -1
                )
            feature_idx += 1
        
        if 'severity_id' in self.feature_names:
            for i, sev in enumerate(logs_df['severity']):
                feature_matrix[i, feature_idx] = self.categorical_encoders['severity'].get(
                    sev, -1
                )
            feature_idx += 1
        
        if 'state_id' in self.feature_names:
            states = self._extract_states(logs_df)
            for i, state in enumerate(states):
                feature_matrix[i, feature_idx] = self.categorical_encoders['state'].get(
                    state, -1
                )
            feature_idx += 1
        
        # Extract and encode numerical features
        numerical_features = self._extract_numerical_features(logs_df)
        
        for feature_name in sorted(self.numerical_stats.keys()):
            if feature_name in numerical_features:
                values = numerical_features[feature_name]
                
                # Replace NaN with 0 (or mean if normalizing)
                if normalize:
                    mean = self.numerical_stats[feature_name]['mean']
                    std = self.numerical_stats[feature_name]['std']
                    # Normalize: (x - mean) / std
                    normalized_values = np.where(
                        np.isnan(values),
                        0.0,  # NaN becomes 0 after normalization
                        (values - mean) / std
                    )
                    feature_matrix[:, feature_idx] = normalized_values
                else:
                    feature_matrix[:, feature_idx] = np.where(
                        np.isnan(values),
                        0.0,
                        values
                    )
            
            feature_idx += 1
        
        return feature_matrix
    
    def fit_transform(self, logs_df: pd.DataFrame, normalize: bool = True) -> np.ndarray:
        """Fit the preprocessor and transform data in one step.
        
        Args:
            logs_df: DataFrame containing log data
            normalize: Whether to normalize numerical features
            
        Returns:
            numpy array of shape (n_samples, n_features)
        """
        self.fit(logs_df)
        return self.transform(logs_df, normalize=normalize)
    
    def get_feature_names(self) -> List[str]:
        """Get list of all feature names.
        
        Returns:
            List of feature names
        """
        return self.feature_names.copy()
    
    def get_feature_info(self) -> Dict[str, Any]:
        """Get detailed information about all features.
        
        Returns:
            Dictionary containing feature information
        """
        info = {
            'n_features': len(self.feature_names),
            'feature_names': self.feature_names,
            'categorical_features': {},
            'numerical_features': {},
        }
        
        # Categorical feature info
        for cat_name, encoder in self.categorical_encoders.items():
            if encoder:
                info['categorical_features'][cat_name] = {
                    'n_classes': len(encoder),
                    'classes': list(encoder.keys())
                }
        
        # Numerical feature info
        for num_name, stats in self.numerical_stats.items():
            info['numerical_features'][num_name] = stats
        
        return info
    
    def decode_categorical(self, feature_name: str, encoded_value: int) -> str:
        """Decode a categorical feature value.
        
        Args:
            feature_name: Name of the categorical feature
            encoded_value: Encoded integer value
            
        Returns:
            Original categorical value
        """
        if feature_name not in self.categorical_decoders:
            raise ValueError(f"Unknown categorical feature: {feature_name}")
        
        return self.categorical_decoders[feature_name].get(
            encoded_value, f"UNKNOWN_{encoded_value}"
        )
    
    def save(self, filepath: str) -> None:
        """Save preprocessor state to file.
        
        Args:
            filepath: Path to save file (.json)
        """
        state = {
            'categorical_encoders': self.categorical_encoders,
            'categorical_decoders': self.categorical_decoders,
            'numerical_stats': self.numerical_stats,
            'feature_names': self.feature_names,
            'fitted': self.fitted,
        }
        
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)
        
        print(f"Preprocessor saved to {filepath}")
    
    def load(self, filepath: str) -> 'LogPreprocessor':
        """Load preprocessor state from file.
        
        Args:
            filepath: Path to saved file (.json)
            
        Returns:
            self for method chaining
        """
        with open(filepath, 'r') as f:
            state = json.load(f)
        
        self.categorical_encoders = state['categorical_encoders']
        self.categorical_decoders = {
            k: {int(ki): v for ki, v in val.items()}  # Convert string keys back to int
            for k, val in state['categorical_decoders'].items()
        }
        self.numerical_stats = state['numerical_stats']
        self.feature_names = state['feature_names']
        self.fitted = state['fitted']
        
        print(f"Preprocessor loaded from {filepath}")
        return self


def create_sequences(feature_matrix: np.ndarray, 
                     sequence_length: int = 10,
                     stride: Optional[int] = None) -> np.ndarray:
    """Create sequences from feature matrix using sliding window.
    
    Args:
        feature_matrix: Array of shape (n_samples, n_features)
        sequence_length: Length of each sequence
        stride: Stride for sliding window (defaults to sequence_length)
        
    Returns:
        Array of shape (n_sequences, sequence_length, n_features)
    """
    if stride is None:
        stride = sequence_length
    
    n_samples, n_features = feature_matrix.shape
    sequences = []
    
    for i in range(0, n_samples - sequence_length + 1, stride):
        seq = feature_matrix[i:i+sequence_length]
        sequences.append(seq)
    
    return np.array(sequences)
