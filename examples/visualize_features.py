"""
Visualize Preprocessed Log Features

This script visualizes the numerical feature vectors created from logs.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.log_to_vec.data.preprocessor import LogPreprocessor


def plot_feature_distributions(feature_matrix, feature_names, output_dir="data"):
    """Plot distributions of all features."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    n_features = len(feature_names)
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
    axes = axes.flatten() if n_features > 1 else [axes]
    
    for i, (feature_name, ax) in enumerate(zip(feature_names, axes)):
        values = feature_matrix[:, i]
        
        # Remove -1 (unknown) values for visualization
        valid_values = values[values >= 0]
        
        if len(valid_values) > 0:
            ax.hist(valid_values, bins=50, edgecolor='black', alpha=0.7)
            ax.set_title(f"{feature_name}")
            ax.set_xlabel("Value")
            ax.set_ylabel("Frequency")
            ax.grid(True, alpha=0.3)
    
    # Remove unused subplots
    for i in range(n_features, len(axes)):
        fig.delaxes(axes[i])
    
    plt.tight_layout()
    output_file = output_dir / "feature_distributions.png"
    plt.savefig(output_file, dpi=150)
    print(f"Saved feature distributions to {output_file}")
    plt.close()


def plot_feature_correlations(feature_matrix, feature_names, output_dir="data"):
    """Plot correlation matrix of features."""
    output_dir = Path(output_dir)
    
    # Compute correlation matrix
    corr_matrix = np.corrcoef(feature_matrix.T)
    
    # Create heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        corr_matrix, 
        annot=True, 
        fmt='.2f',
        xticklabels=feature_names,
        yticklabels=feature_names,
        cmap='coolwarm',
        center=0,
        vmin=-1, vmax=1,
        square=True
    )
    plt.title("Feature Correlation Matrix")
    plt.tight_layout()
    
    output_file = output_dir / "feature_correlations.png"
    plt.savefig(output_file, dpi=150)
    print(f"Saved feature correlations to {output_file}")
    plt.close()


def plot_sequence_visualization(sequences, feature_names, output_dir="data", n_sequences=5):
    """Visualize feature sequences as heatmaps."""
    output_dir = Path(output_dir)
    
    n_show = min(n_sequences, len(sequences))
    fig, axes = plt.subplots(n_show, 1, figsize=(12, 3*n_show))
    
    if n_show == 1:
        axes = [axes]
    
    for i, ax in enumerate(axes):
        seq = sequences[i]  # Shape: (sequence_length, n_features)
        
        sns.heatmap(
            seq.T,  # Transpose so features are rows
            ax=ax,
            cmap='viridis',
            yticklabels=feature_names,
            xticklabels=range(len(seq)),
            cbar_kws={'label': 'Normalized Value'}
        )
        ax.set_title(f"Sequence {i}")
        ax.set_xlabel("Time Step")
        ax.set_ylabel("Feature")
    
    plt.tight_layout()
    output_file = output_dir / "sequence_visualization.png"
    plt.savefig(output_file, dpi=150)
    print(f"Saved sequence visualization to {output_file}")
    plt.close()


def plot_time_series_samples(feature_matrix, feature_names, output_dir="data", n_samples=1000):
    """Plot time series of selected features."""
    output_dir = Path(output_dir)
    
    n_samples = min(n_samples, len(feature_matrix))
    sample_data = feature_matrix[:n_samples]
    
    # Select a few interesting features to plot
    plot_features = []
    for name in ['event_type_id', 'severity_id', 'temperature', 'pressure', 'position']:
        if name in feature_names:
            plot_features.append(feature_names.index(name))
    
    if not plot_features:
        plot_features = list(range(min(5, len(feature_names))))
    
    fig, axes = plt.subplots(len(plot_features), 1, figsize=(15, 3*len(plot_features)))
    
    if len(plot_features) == 1:
        axes = [axes]
    
    for ax, feat_idx in zip(axes, plot_features):
        values = sample_data[:, feat_idx]
        ax.plot(values, linewidth=0.5)
        ax.set_title(f"{feature_names[feat_idx]} over time")
        ax.set_xlabel("Log Entry Index")
        ax.set_ylabel("Value")
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_file = output_dir / "time_series_features.png"
    plt.savefig(output_file, dpi=150)
    print(f"Saved time series plot to {output_file}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Visualize preprocessed log features"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/processed_features.npz",
        help="Input processed features file (.npz)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data",
        help="Output directory for plots"
    )
    
    args = parser.parse_args()
    
    # Load processed features
    print(f"Loading processed features from {args.input}...")
    data = np.load(args.input)
    
    feature_matrix = data['features']
    feature_names = data['feature_names'].tolist()
    
    print(f"Feature matrix shape: {feature_matrix.shape}")
    print(f"Features: {feature_names}")
    
    # Create visualizations
    print("\nGenerating visualizations...")
    
    print("1. Plotting feature distributions...")
    plot_feature_distributions(feature_matrix, feature_names, args.output_dir)
    
    print("2. Plotting feature correlations...")
    plot_feature_correlations(feature_matrix, feature_names, args.output_dir)
    
    print("3. Plotting time series...")
    plot_time_series_samples(feature_matrix, feature_names, args.output_dir)
    
    # If sequences exist, visualize them
    if 'sequences' in data:
        sequences = data['sequences']
        print(f"4. Plotting sequences (shape: {sequences.shape})...")
        plot_sequence_visualization(sequences, feature_names, args.output_dir)
    
    print(f"\nAll visualizations saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
