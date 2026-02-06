"""
Example: Train a Simple Classifier on Preprocessed Log Features

This script demonstrates how to use preprocessed log features
to train a classifier for anomaly detection or event prediction.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import argparse
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.log_to_vec.data.preprocessor import LogPreprocessor


def create_labels_from_events(logs_df, anomaly_events):
    """Create binary labels based on anomalous events.
    
    Args:
        logs_df: DataFrame containing log data
        anomaly_events: List of event types considered anomalous
        
    Returns:
        Binary labels (0=normal, 1=anomaly)
    """
    labels = logs_df['event_type'].apply(
        lambda x: 1 if x in anomaly_events else 0
    ).values
    return labels


def main():
    parser = argparse.ArgumentParser(
        description="Train classifier on preprocessed log features"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/toy_logs.csv",
        help="Input log file (CSV)"
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of data to use for testing"
    )
    
    args = parser.parse_args()
    
    # Load log data
    print(f"Loading log data from {args.input}...")
    logs_df = pd.read_csv(args.input)
    print(f"Loaded {len(logs_df)} log entries")
    
    # Create labels (mark error/alarm events as anomalies)
    anomaly_events = [
        'ALARM_TEMP',
        'ALARM_PRESSURE', 
        'COMMUNICATION_ERROR',
        'SYSTEM_STOP'
    ]
    labels = create_labels_from_events(logs_df, anomaly_events)
    
    print(f"\nLabel distribution:")
    print(f"  Normal (0): {np.sum(labels == 0)} ({100*np.mean(labels == 0):.1f}%)")
    print(f"  Anomaly (1): {np.sum(labels == 1)} ({100*np.mean(labels == 1):.1f}%)")
    
    # Preprocess features
    print("\n" + "="*60)
    print("Preprocessing features...")
    preprocessor = LogPreprocessor()
    X = preprocessor.fit_transform(logs_df)
    
    print(f"Feature matrix shape: {X.shape}")
    print(f"Features: {preprocessor.get_feature_names()}")
    
    # Split data
    print("\n" + "="*60)
    print(f"Splitting data (test_size={args.test_size})...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, labels, 
        test_size=args.test_size, 
        random_state=42,
        stratify=labels
    )
    
    print(f"Train set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Train classifier
    print("\n" + "="*60)
    print("Training Random Forest classifier...")
    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    clf.fit(X_train, y_train)
    
    # Evaluate
    print("\n" + "="*60)
    print("Evaluation Results:")
    
    train_score = clf.score(X_train, y_train)
    test_score = clf.score(X_test, y_test)
    
    print(f"\nAccuracy:")
    print(f"  Train: {train_score:.4f}")
    print(f"  Test:  {test_score:.4f}")
    
    # Predictions
    y_pred = clf.predict(X_test)
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(
        y_test, y_pred,
        target_names=['Normal', 'Anomaly'],
        digits=4
    ))
    
    # Confusion matrix
    print("Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    print("\n[[TN, FP],")
    print(" [FN, TP]]")
    
    # Feature importance
    print("\n" + "="*60)
    print("Top 10 Most Important Features:")
    feature_names = preprocessor.get_feature_names()
    importances = clf.feature_importances_
    
    # Sort by importance
    indices = np.argsort(importances)[::-1]
    
    for i in range(min(10, len(feature_names))):
        idx = indices[i]
        print(f"  {i+1}. {feature_names[idx]}: {importances[idx]:.4f}")
    
    # Example predictions
    print("\n" + "="*60)
    print("Example Predictions (first 10 test samples):")
    
    for i in range(min(10, len(X_test))):
        pred = y_pred[i]
        true = y_test[i]
        prob = clf.predict_proba(X_test[i:i+1])[0]
        
        status = "✓" if pred == true else "✗"
        print(f"  {status} Sample {i}: Pred={pred} (prob={prob[1]:.3f}), True={true}")


if __name__ == "__main__":
    main()
