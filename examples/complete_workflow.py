"""
Complete Example: Log Preprocessing Workflow

This example demonstrates the complete workflow of preprocessing
log data into numerical feature vectors.

Run this script to see the full pipeline in action.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.log_to_vec.data.preprocessor import LogPreprocessor, create_sequences


def print_section(title):
    """Print a formatted section header."""
    print("\n" + "="*70)
    print(f" {title}")
    print("="*70 + "\n")


def main():
    print_section("LOG PREPROCESSING COMPLETE EXAMPLE")
    
    # =========================================================================
    # STEP 1: Load Raw Log Data
    # =========================================================================
    print_section("STEP 1: Load Raw Log Data")
    
    logs_df = pd.read_csv("data/toy_logs.csv")
    print(f"Loaded {len(logs_df)} log entries")
    print(f"\nColumns: {list(logs_df.columns)}")
    print(f"\nFirst 3 log entries:")
    print(logs_df.head(3).to_string())
    
    # =========================================================================
    # STEP 2: Initialize Preprocessor
    # =========================================================================
    print_section("STEP 2: Initialize and Fit Preprocessor")
    
    preprocessor = LogPreprocessor()
    print("Fitting preprocessor on training data...")
    preprocessor.fit(logs_df)
    
    print(f"\n✓ Preprocessor fitted successfully")
    print(f"  Total features: {len(preprocessor.get_feature_names())}")
    
    # Show categorical encoders
    print("\nCategorical Encoders:")
    for cat_name, encoder in preprocessor.categorical_encoders.items():
        if encoder:
            print(f"  {cat_name}: {len(encoder)} unique values")
            # Show first 5 mappings
            for i, (key, val) in enumerate(list(encoder.items())[:5]):
                print(f"    {key} → {val}")
            if len(encoder) > 5:
                print(f"    ... and {len(encoder) - 5} more")
    
    # =========================================================================
    # STEP 3: Transform Logs to Feature Vectors
    # =========================================================================
    print_section("STEP 3: Transform Logs to Numerical Features")
    
    feature_matrix = preprocessor.transform(logs_df, normalize=True)
    
    print(f"Feature matrix shape: {feature_matrix.shape}")
    print(f"  Samples: {feature_matrix.shape[0]}")
    print(f"  Features per sample: {feature_matrix.shape[1]}")
    
    print("\nFeature Names:")
    for i, name in enumerate(preprocessor.get_feature_names()):
        print(f"  [{i}] {name}")
    
    print("\nExample Feature Vectors (first 3):")
    for i in range(3):
        print(f"\n  Sample {i}:")
        for j, name in enumerate(preprocessor.get_feature_names()):
            print(f"    {name:20s} = {feature_matrix[i, j]:8.4f}")
    
    # =========================================================================
    # STEP 4: Decode Features (Understand What They Mean)
    # =========================================================================
    print_section("STEP 4: Decode Categorical Features")
    
    print("Let's decode the first 3 samples to understand the values:\n")
    
    for i in range(3):
        print(f"Sample {i}:")
        
        # Get raw vector
        vector = feature_matrix[i]
        feature_names = preprocessor.get_feature_names()
        
        # Decode each categorical feature
        if 'event_type_id' in feature_names:
            idx = feature_names.index('event_type_id')
            event_id = int(vector[idx])
            event_name = preprocessor.decode_categorical('event_type', event_id)
            print(f"  Event: {event_name} (ID={event_id})")
        
        if 'severity_id' in feature_names:
            idx = feature_names.index('severity_id')
            sev_id = int(vector[idx])
            severity = preprocessor.decode_categorical('severity', sev_id)
            print(f"  Severity: {severity} (ID={sev_id})")
        
        if 'state_id' in feature_names:
            idx = feature_names.index('state_id')
            state_id = int(vector[idx])
            state = preprocessor.decode_categorical('state', state_id)
            print(f"  State: {state} (ID={state_id})")
        
        # Show numerical features
        if 'temperature' in feature_names:
            idx = feature_names.index('temperature')
            print(f"  Temperature: {vector[idx]:.4f} (normalized)")
        
        if 'pressure' in feature_names:
            idx = feature_names.index('pressure')
            print(f"  Pressure: {vector[idx]:.4f} (normalized)")
        
        print()
    
    # =========================================================================
    # STEP 5: Create Sequences (for Time Series Models)
    # =========================================================================
    print_section("STEP 5: Create Sequences for Time Series Analysis")
    
    sequence_length = 10
    stride = 5
    
    print(f"Creating sequences with:")
    print(f"  Sequence length: {sequence_length}")
    print(f"  Stride: {stride}")
    print()
    
    sequences = create_sequences(
        feature_matrix,
        sequence_length=sequence_length,
        stride=stride
    )
    
    print(f"✓ Created {len(sequences)} sequences")
    print(f"  Shape: {sequences.shape}")
    print(f"    (n_sequences, sequence_length, n_features)")
    
    print("\nFirst sequence (shape: {})".format(sequences[0].shape))
    print("This represents 10 consecutive log entries:")
    print(sequences[0])
    
    # =========================================================================
    # STEP 6: Save Everything
    # =========================================================================
    print_section("STEP 6: Save Preprocessed Data")
    
    # Save preprocessor
    preprocessor_path = "data/example_preprocessor.json"
    preprocessor.save(preprocessor_path)
    print(f"✓ Saved preprocessor to: {preprocessor_path}")
    
    # Save features
    features_path = "data/example_features.npz"
    np.savez(
        features_path,
        features=feature_matrix,
        sequences=sequences,
        feature_names=np.array(preprocessor.get_feature_names()),
        sequence_length=sequence_length,
        stride=stride
    )
    print(f"✓ Saved features to: {features_path}")
    
    # =========================================================================
    # STEP 7: Show Usage Examples
    # =========================================================================
    print_section("STEP 7: How to Use These Features")
    
    print("The preprocessed features can now be used for:")
    print()
    
    print("1. Machine Learning (Scikit-learn):")
    print("   from sklearn.ensemble import RandomForestClassifier")
    print("   model = RandomForestClassifier()")
    print("   model.fit(feature_matrix, labels)")
    print()
    
    print("2. Deep Learning (PyTorch):")
    print("   import torch")
    print("   tensor_X = torch.FloatTensor(sequences)")
    print("   # Use in your neural network")
    print()
    
    print("3. Anomaly Detection:")
    print("   from sklearn.ensemble import IsolationForest")
    print("   detector = IsolationForest()")
    print("   anomalies = detector.fit_predict(feature_matrix)")
    print()
    
    print("4. Clustering:")
    print("   from sklearn.cluster import KMeans")
    print("   kmeans = KMeans(n_clusters=5)")
    print("   clusters = kmeans.fit_predict(feature_matrix)")
    print()
    
    print("5. Load Later:")
    print("   # Load preprocessor")
    print("   preprocessor = LogPreprocessor()")
    print("   preprocessor.load('data/example_preprocessor.json')")
    print()
    print("   # Transform new logs")
    print("   new_logs = pd.read_csv('new_logs.csv')")
    print("   new_features = preprocessor.transform(new_logs)")
    print()
    
    # =========================================================================
    # Summary Statistics
    # =========================================================================
    print_section("SUMMARY")
    
    print(f"✓ Processed {len(logs_df)} log entries")
    print(f"✓ Generated {feature_matrix.shape[1]} features per entry")
    print(f"✓ Created {len(sequences)} sequences of length {sequence_length}")
    print(f"✓ Memory usage:")
    print(f"    Feature matrix: {feature_matrix.nbytes / 1024:.2f} KB")
    print(f"    Sequences: {sequences.nbytes / 1024:.2f} KB")
    print()
    print("All files saved and ready to use! 🎉")
    print()


if __name__ == "__main__":
    main()
