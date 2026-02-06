"""
Evaluation Metrics

Metrics for assessing embedding quality.
"""

import torch
import numpy as np
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from typing import Dict, List, Tuple
import warnings


def reconstruction_accuracy(logits: torch.Tensor, 
                           targets: torch.Tensor,
                           padding_idx: int = 0) -> float:
    """Calculate reconstruction accuracy.
    
    Args:
        logits: Model predictions [batch_size, seq_len, vocab_size]
        targets: Target sequences [batch_size, seq_len]
        padding_idx: Padding token index to ignore
        
    Returns:
        Accuracy (0-1)
    """
    predictions = logits.argmax(dim=-1)
    mask = (targets != padding_idx)
    correct = (predictions == targets) & mask
    accuracy = correct.sum().item() / mask.sum().item()
    return accuracy


def embedding_similarity(embeddings: np.ndarray,
                        num_neighbors: int = 5) -> Dict[str, float]:
    """Calculate embedding similarity metrics.
    
    Args:
        embeddings: Embedding matrix [num_samples, embedding_dim]
        num_neighbors: Number of nearest neighbors to consider
        
    Returns:
        Dictionary of similarity metrics
    """
    # Fit nearest neighbors
    nbrs = NearestNeighbors(n_neighbors=num_neighbors + 1, metric='cosine')
    nbrs.fit(embeddings)
    distances, indices = nbrs.kneighbors(embeddings)
    
    # Average distance to k nearest neighbors (excluding self)
    avg_neighbor_distance = distances[:, 1:].mean()
    
    # Compute pairwise cosine similarity statistics
    from sklearn.metrics.pairwise import cosine_similarity
    sim_matrix = cosine_similarity(embeddings)
    
    # Remove diagonal
    np.fill_diagonal(sim_matrix, np.nan)
    
    metrics = {
        "avg_neighbor_distance": float(avg_neighbor_distance),
        "mean_similarity": float(np.nanmean(sim_matrix)),
        "std_similarity": float(np.nanstd(sim_matrix)),
        "min_similarity": float(np.nanmin(sim_matrix)),
        "max_similarity": float(np.nanmax(sim_matrix)),
    }
    
    return metrics


def clustering_metrics(embeddings: np.ndarray,
                      num_clusters: int = 3,
                      labels: np.ndarray = None) -> Dict[str, float]:
    """Calculate clustering quality metrics.
    
    Args:
        embeddings: Embedding matrix [num_samples, embedding_dim]
        num_clusters: Number of clusters for KMeans
        labels: Ground truth labels (optional, for supervised metrics)
        
    Returns:
        Dictionary of clustering metrics
    """
    if len(embeddings) < num_clusters:
        warnings.warn(f"Not enough samples ({len(embeddings)}) for {num_clusters} clusters")
        return {}
    
    # Perform clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(embeddings)
    
    metrics = {}
    
    # Silhouette score (higher is better, range [-1, 1])
    if len(np.unique(cluster_labels)) > 1:
        metrics["silhouette_score"] = float(silhouette_score(embeddings, cluster_labels))
    
    # Davies-Bouldin index (lower is better)
    if len(np.unique(cluster_labels)) > 1:
        metrics["davies_bouldin_score"] = float(davies_bouldin_score(embeddings, cluster_labels))
    
    # Inertia (within-cluster sum of squares)
    metrics["inertia"] = float(kmeans.inertia_)
    
    # If ground truth labels are provided
    if labels is not None:
        from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
        metrics["adjusted_rand_score"] = float(adjusted_rand_score(labels, cluster_labels))
        metrics["normalized_mutual_info"] = float(normalized_mutual_info_score(labels, cluster_labels))
    
    return metrics


def retrieval_metrics(query_embeddings: np.ndarray,
                     corpus_embeddings: np.ndarray,
                     ground_truth_indices: np.ndarray = None,
                     k: int = 5) -> Dict[str, float]:
    """Calculate retrieval metrics (similarity search).
    
    Args:
        query_embeddings: Query embedding matrix [num_queries, embedding_dim]
        corpus_embeddings: Corpus embedding matrix [num_corpus, embedding_dim]
        ground_truth_indices: Ground truth relevant indices for each query [num_queries, num_relevant]
        k: Number of top results to retrieve
        
    Returns:
        Dictionary of retrieval metrics
    """
    # Find k nearest neighbors for each query
    nbrs = NearestNeighbors(n_neighbors=k, metric='cosine')
    nbrs.fit(corpus_embeddings)
    distances, indices = nbrs.kneighbors(query_embeddings)
    
    metrics = {
        "mean_distance_at_k": float(distances.mean()),
        "std_distance_at_k": float(distances.std()),
    }
    
    # If ground truth is provided, calculate precision/recall
    if ground_truth_indices is not None:
        precisions = []
        recalls = []
        
        for i, retrieved in enumerate(indices):
            if i < len(ground_truth_indices):
                relevant = set(ground_truth_indices[i])
                retrieved_set = set(retrieved)
                
                true_positives = len(relevant & retrieved_set)
                precision = true_positives / k if k > 0 else 0
                recall = true_positives / len(relevant) if len(relevant) > 0 else 0
                
                precisions.append(precision)
                recalls.append(recall)
        
        if precisions:
            metrics["precision_at_k"] = float(np.mean(precisions))
            metrics["recall_at_k"] = float(np.mean(recalls))
    
    return metrics


class EvaluationSuite:
    """Comprehensive evaluation suite for embeddings."""
    
    def __init__(self, 
                 num_neighbors: int = 5,
                 num_clusters: int = 3):
        """Initialize evaluation suite.
        
        Args:
            num_neighbors: Number of neighbors for similarity metrics
            num_clusters: Number of clusters for clustering metrics
        """
        self.num_neighbors = num_neighbors
        self.num_clusters = num_clusters
    
    def evaluate(self,
                embeddings: np.ndarray,
                labels: np.ndarray = None) -> Dict[str, float]:
        """Run comprehensive evaluation.
        
        Args:
            embeddings: Embedding matrix [num_samples, embedding_dim]
            labels: Optional ground truth labels
            
        Returns:
            Dictionary of all metrics
        """
        all_metrics = {}
        
        # Similarity metrics
        sim_metrics = embedding_similarity(embeddings, self.num_neighbors)
        all_metrics.update({f"similarity/{k}": v for k, v in sim_metrics.items()})
        
        # Clustering metrics
        cluster_metrics = clustering_metrics(embeddings, self.num_clusters, labels)
        all_metrics.update({f"clustering/{k}": v for k, v in cluster_metrics.items()})
        
        return all_metrics
