"""
Example: Preprocess Logs to Numerical Vectors

This script demonstrates how to use the LogPreprocessor to transform
log data into numerical feature vectors suitable for machine learning.
"""

import pandas as pd
import numpy as np
import argparse
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.log_to_vec.data.preprocessor import LogPreprocessor, create_sequences


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess log data to numerical feature vectors"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/toy_logs.csv",
        help="Input log file (CSV)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/processed_features.npz",
        help="Output file for processed features (.npz)"
    )
    parser.add_argument(
        "--save-preprocessor",
        type=str,
        default="data/preprocessor.json",
        help="Save preprocessor state to file"
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        default=True,
        help="Normalize numerical features"
    )
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=10,
        help="Length of sequences (0 to disable sequencing)"
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=None,
        help="Stride for sliding window (default: sequence_length)"
    )
    
    args = parser.parse_args()
    
    # Load log data
    print(f"Loading log data from {args.input}...")
    logs_df = pd.read_csv(args.input)
    print(f"Loaded {len(logs_df)} log entries")
    print(f"\nColumns: {logs_df.columns.tolist()}")
    print(f"\nFirst few rows:")
    print(logs_df.head())
    
    # Initialize and fit preprocessor
    print("\n" + "="*60)
    preprocessor = LogPreprocessor()
    feature_matrix = preprocessor.fit_transform(logs_df, normalize=args.normalize)
    
    print(f"\nFeature matrix shape: {feature_matrix.shape}")
    print(f"Features: {preprocessor.get_feature_names()}")
    
    # Display feature info
    print("\n" + "="*60)
    print("Feature Information:")
    feature_info = preprocessor.get_feature_info()
    
    print(f"\nCategorical Features:")
    for cat_name, cat_info in feature_info['categorical_features'].items():
        print(f"  {cat_name}: {cat_info['n_classes']} classes")
        print(f"    Classes: {cat_info['classes'][:5]}..." if len(cat_info['classes']) > 5 else f"    Classes: {cat_info['classes']}")
    
    print(f"\nNumerical Features:")
    for num_name, num_stats in feature_info['numerical_features'].items():
        print(f"  {num_name}: mean={num_stats['mean']:.3f}, std={num_stats['std']:.3f}")
    
    # Show sample feature vectors
    print("\n" + "="*60)
    print("Sample Feature Vectors:")
    print("\nFirst 5 samples:")
    for i in range(min(5, len(feature_matrix))):
        print(f"  Sample {i}: {feature_matrix[i]}")
    
    # Create sequences if requested
    if args.sequence_length > 0:
        print("\n" + "="*60)
        print(f"Creating sequences with length {args.sequence_length}...")
        sequences = create_sequences(
            feature_matrix, 
            sequence_length=args.sequence_length,
            stride=args.stride
        )
        print(f"Sequence shape: {sequences.shape}")
        print(f"Number of sequences: {len(sequences)}")
        
        # Show first sequence
        print(f"\nFirst sequence (shape {sequences[0].shape}):")
        print(sequences[0])
    else:
        sequences = None
    
    # Save preprocessor state
    if args.save_preprocessor:
        print("\n" + "="*60)
        preprocessor.save(args.save_preprocessor)
    
    # Save processed features
    print(f"\nSaving processed features to {args.output}...")
    save_dict = {
        'features': feature_matrix,
        'feature_names': np.array(preprocessor.get_feature_names()),
    }
    
    if sequences is not None:
        save_dict['sequences'] = sequences
        save_dict['sequence_length'] = args.sequence_length
        save_dict['stride'] = args.stride if args.stride else args.sequence_length
    
    np.savez(args.output, **save_dict)
    print(f"Saved successfully!")
    
    # Statistics
    print("\n" + "="*60)
    print("Statistics:")
    print(f"  Total log entries: {len(logs_df)}")
    print(f"  Feature dimensions: {feature_matrix.shape[1]}")
    print(f"  Feature matrix size: {feature_matrix.nbytes / 1024:.2f} KB")
    if sequences is not None:
        print(f"  Number of sequences: {len(sequences)}")
        print(f"  Sequence dimensions: {sequences.shape[1]} x {sequences.shape[2]}")
        print(f"  Sequence data size: {sequences.nbytes / 1024:.2f} KB")


if __name__ == "__main__":
    main()
