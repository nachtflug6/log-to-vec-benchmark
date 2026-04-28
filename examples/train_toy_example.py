"""
Training Script for Toy Example

Demonstrates how to train an embedding model on the toy log dataset.
"""

import argparse
import yaml
from pathlib import Path
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from tqdm import tqdm

# Add parent directory to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from log_to_vec.data.log_parser import LogParser
from log_to_vec.data.dataset import create_dataloaders
from log_to_vec.models.autoencoder import LSTMAutoencoder, TransformerAutoencoder
from log_to_vec.evaluation.metrics import reconstruction_accuracy, EvaluationSuite
from log_to_vec.mode_change import compute_change_scores, detect_change_points, cluster_segments
from log_to_vec.evaluation.mode_change_metrics import mode_change_metrics


def evaluate_mode_change(embeddings, config):
    """Run optional baseline mode-change evaluation on embeddings."""
    mode_cfg = config.get("mode_change", {})
    if not mode_cfg.get("enabled", False):
        return {}

    scores = compute_change_scores(
        embeddings,
        window_size=mode_cfg.get("window_size", 5),
    )
    change_points = detect_change_points(
        scores,
        threshold_scale=mode_cfg.get("threshold_scale", 2.5),
        min_distance=mode_cfg.get("min_distance", 3),
    )
    clustered = cluster_segments(
        embeddings,
        change_points,
        num_clusters=mode_cfg.get("num_clusters", 3),
        random_state=config["training"].get("seed", 42),
    )

    metrics = mode_change_metrics(
        scores=scores,
        change_points=change_points,
        segment_labels=clustered["segment_labels"],
    )
    return metrics


def train_epoch(model, dataloader, optimizer, criterion, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    total_accuracy = 0
    num_batches = 0
    
    for batch in tqdm(dataloader, desc="Training"):
        # Move batch to device
        batch = {k: v.to(device) for k, v in batch.items()}
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(batch)
        
        # Compute loss
        logits = outputs["logits"]
        targets = batch["events"]
        
        # Reshape for loss computation
        loss = criterion(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1)
        )
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Compute accuracy
        accuracy = reconstruction_accuracy(logits, targets)
        
        total_loss += loss.item()
        total_accuracy += accuracy
        num_batches += 1
    
    avg_loss = total_loss / num_batches
    avg_accuracy = total_accuracy / num_batches
    
    return avg_loss, avg_accuracy


def validate(model, dataloader, criterion, device):
    """Validate model."""
    model.eval()
    total_loss = 0
    total_accuracy = 0
    num_batches = 0
    
    all_embeddings = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation"):
            batch = {k: v.to(device) for k, v in batch.items()}
            
            outputs = model(batch)
            logits = outputs["logits"]
            targets = batch["events"]
            embeddings = outputs["embeddings"]
            
            # Compute loss
            loss = criterion(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1)
            )
            
            # Compute accuracy
            accuracy = reconstruction_accuracy(logits, targets)
            
            total_loss += loss.item()
            total_accuracy += accuracy
            num_batches += 1
            
            # Collect embeddings
            all_embeddings.append(embeddings.cpu().numpy())
    
    avg_loss = total_loss / num_batches
    avg_accuracy = total_accuracy / num_batches
    
    # Concatenate all embeddings
    all_embeddings = np.concatenate(all_embeddings, axis=0)
    
    return avg_loss, avg_accuracy, all_embeddings


def main():
    parser = argparse.ArgumentParser(description="Train log embedding model on toy example")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/toy_example.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--data-file",
        type=str,
        default=None,
        help="Path to log data file (overrides config)"
    )
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    
    # Override data file if specified
    if args.data_file:
        config["data"]["log_file"] = args.data_file
    
    # Set random seed
    torch.manual_seed(config["training"]["seed"])
    np.random.seed(config["training"]["seed"])
    
    # Set device
    device = torch.device(config["training"]["device"] if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load data
    print(f"\nLoading data from {config['data']['log_file']}...")
    logs_df = pd.read_csv(config["data"]["log_file"])
    print(f"Loaded {len(logs_df)} log events")
    
    # Parse logs
    print("\nParsing logs...")
    parser = LogParser(vocab_size=None)
    parser.build_vocabulary(logs_df)
    features = parser.extract_features(logs_df)
    
    # Create dataloaders
    print("\nCreating dataloaders...")
    train_loader, val_loader, test_loader = create_dataloaders(features, config, parser)
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}, Test batches: {len(test_loader)}")
    
    # Create model
    print(f"\nCreating {config['model']['type']} model...")
    vocab_size = len(parser.token2idx)
    
    if config["model"]["type"] == "autoencoder":
        model = LSTMAutoencoder(
            vocab_size=vocab_size,
            embedding_dim=config["model"]["embedding_dim"],
            hidden_dim=config["model"]["hidden_dim"],
            num_layers=config["model"]["num_layers"],
            dropout=config["model"]["dropout"],
        )
    elif config["model"]["type"] == "transformer":
        model = TransformerAutoencoder(
            vocab_size=vocab_size,
            embedding_dim=config["model"]["embedding_dim"],
            num_heads=8,
            num_encoder_layers=config["model"]["num_layers"],
            num_decoder_layers=config["model"]["num_layers"],
            dim_feedforward=config["model"]["hidden_dim"],
            dropout=config["model"]["dropout"],
        )
    else:
        raise ValueError(f"Unknown model type: {config['model']['type']}")
    
    model = model.to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Setup training
    criterion = nn.CrossEntropyLoss(ignore_index=parser.token2idx["<PAD>"])
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"]
    )
    
    # Create checkpoint directory
    checkpoint_dir = Path(config["logging"]["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Training loop
    print("\nStarting training...")
    best_val_loss = float("inf")
    
    for epoch in range(config["training"]["num_epochs"]):
        print(f"\nEpoch {epoch + 1}/{config['training']['num_epochs']}")
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}")
        
        # Validate
        val_loss, val_acc, val_embeddings = validate(model, val_loader, criterion, device)
        print(f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.4f}")
        
        # Evaluate embeddings every few epochs
        if (epoch + 1) % 10 == 0:
            print("\nEvaluating embeddings...")
            evaluator = EvaluationSuite(
                num_neighbors=config["evaluation"]["num_neighbors"],
                num_clusters=config["evaluation"]["num_clusters"]
            )
            eval_metrics = evaluator.evaluate(val_embeddings)
            for metric_name, value in eval_metrics.items():
                print(f"  {metric_name}: {value:.4f}")

            mode_metrics = evaluate_mode_change(val_embeddings, config)
            if mode_metrics:
                print("\nMode-change metrics:")
                for metric_name, value in mode_metrics.items():
                    print(f"  {metric_name}: {value:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = checkpoint_dir / "best_model.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "val_accuracy": val_acc,
                "config": config,
            }, checkpoint_path)
            print(f"Saved best model to {checkpoint_path}")
    
    # Final evaluation on test set
    print("\n" + "="*50)
    print("Final evaluation on test set...")
    test_loss, test_acc, test_embeddings = validate(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")
    
    evaluator = EvaluationSuite(
        num_neighbors=config["evaluation"]["num_neighbors"],
        num_clusters=config["evaluation"]["num_clusters"]
    )
    eval_metrics = evaluator.evaluate(test_embeddings)
    print("\nTest Embedding Metrics:")
    for metric_name, value in eval_metrics.items():
        print(f"  {metric_name}: {value:.4f}")

    mode_metrics = evaluate_mode_change(test_embeddings, config)
    if mode_metrics:
        print("\nTest Mode-Change Metrics:")
        for metric_name, value in mode_metrics.items():
            print(f"  {metric_name}: {value:.4f}")
    
    # Save embeddings
    embeddings_path = checkpoint_dir / "test_embeddings.npy"
    np.save(embeddings_path, test_embeddings)
    print(f"\nSaved test embeddings to {embeddings_path}")
    
    print("\nTraining complete!")


if __name__ == "__main__":
    main()
