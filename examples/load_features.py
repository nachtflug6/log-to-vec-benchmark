"""
Example: Load and Use Preprocessed Features

This script demonstrates how to load previously preprocessed
features and use them for analysis or model training.
"""

import numpy as np
import argparse
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.log_to_vec.data.preprocessor import LogPreprocessor


def main():
    parser = argparse.ArgumentParser(
        description="Load and inspect preprocessed features"
    )
    parser.add_argument(
        "--features",
        type=str,
        default="data/processed_features.npz",
        help="Path to preprocessed features file (.npz)"
    )
    parser.add_argument(
        "--preprocessor",
        type=str,
        default="data/preprocessor.json",
        help="Path to preprocessor state file (.json)"
    )
    
    args = parser.parse_args()
    
    # Load preprocessed features
    print("="*60)
    print("Loading Preprocessed Features")
    print("="*60)
    
    data = np.load(args.features)
    
    print(f"\nLoaded from: {args.features}")
    print(f"\nAvailable arrays:")
    for key in data.keys():
        arr = data[key]
        if isinstance(arr, np.ndarray):
            print(f"  {key}: shape={arr.shape}, dtype={arr.dtype}")
        else:
            print(f"  {key}: {arr}")
    
    # Extract features
    feature_matrix = data['features']
    feature_names = data['feature_names'].tolist()
    
    print(f"\n{'='*60}")
    print("Feature Matrix")
    print("="*60)
    print(f"Shape: {feature_matrix.shape}")
    print(f"  Samples: {feature_matrix.shape[0]}")
    print(f"  Features: {feature_matrix.shape[1]}")
    print(f"  Memory: {feature_matrix.nbytes / 1024:.2f} KB")
    
    print(f"\nFeature Names:")
    for i, name in enumerate(feature_names):
        print(f"  {i}: {name}")
    
    # Load preprocessor
    print(f"\n{'='*60}")
    print("Loading Preprocessor")
    print("="*60)
    
    preprocessor = LogPreprocessor()
    preprocessor.load(args.preprocessor)
    
    # Show feature statistics
    print(f"\n{'='*60}")
    print("Feature Statistics")
    print("="*60)
    
    for i, name in enumerate(feature_names):
        values = feature_matrix[:, i]
        
        print(f"\n{name}:")
        print(f"  Min:    {np.min(values):.4f}")
        print(f"  Max:    {np.max(values):.4f}")
        print(f"  Mean:   {np.mean(values):.4f}")
        print(f"  Std:    {np.std(values):.4f}")
        print(f"  Median: {np.median(values):.4f}")
        
        # Show unique values for categorical features
        unique_values = np.unique(values)
        if len(unique_values) <= 20:
            print(f"  Unique: {unique_values.tolist()}")
    
    # If sequences exist, analyze them
    if 'sequences' in data:
        sequences = data['sequences']
        sequence_length = data.get('sequence_length', sequences.shape[1])
        stride = data.get('stride', sequence_length)
        
        print(f"\n{'='*60}")
        print("Sequences")
        print("="*60)
        print(f"Shape: {sequences.shape}")
        print(f"  Number of sequences: {sequences.shape[0]}")
        print(f"  Sequence length: {sequences.shape[1]}")
        print(f"  Features per step: {sequences.shape[2]}")
        print(f"  Stride: {stride}")
        print(f"  Memory: {sequences.nbytes / 1024:.2f} KB")
        
        # Show first sequence
        print(f"\nFirst Sequence (timesteps 0-{sequence_length-1}):")
        print(sequences[0])
    
    # Decode some categorical values as examples
    print(f"\n{'='*60}")
    print("Sample Decoding")
    print("="*60)
    
    print("\nFirst 10 samples with decoded categorical features:")
    
    for i in range(min(10, len(feature_matrix))):
        sample = feature_matrix[i]
        
        # Decode categorical features
        decoded = {}
        if 'event_type_id' in feature_names:
            idx = feature_names.index('event_type_id')
            event_id = int(sample[idx])
            decoded['event_type'] = preprocessor.decode_categorical('event_type', event_id)
        
        if 'component_id' in feature_names:
            idx = feature_names.index('component_id')
            comp_id = int(sample[idx])
            decoded['component'] = preprocessor.decode_categorical('component', comp_id)
        
        if 'severity_id' in feature_names:
            idx = feature_names.index('severity_id')
            sev_id = int(sample[idx])
            decoded['severity'] = preprocessor.decode_categorical('severity', sev_id)
        
        if 'state_id' in feature_names:
            idx = feature_names.index('state_id')
            state_id = int(sample[idx])
            decoded['state'] = preprocessor.decode_categorical('state', state_id)
        
        print(f"\nSample {i}:")
        for key, val in decoded.items():
            print(f"  {key}: {val}")
        
        # Show numerical features
        if 'temperature' in feature_names:
            idx = feature_names.index('temperature')
            print(f"  temperature: {sample[idx]:.4f}")
        if 'pressure' in feature_names:
            idx = feature_names.index('pressure')
            print(f"  pressure: {sample[idx]:.4f}")


if __name__ == "__main__":
    main()
