"""
Evaluation metrics for the music recommendation system.
"""
import numpy as np
from typing import List, Set


def recall_at_k(recommended: List[str], relevant: Set[str], k: int) -> float:
    """
    Calculate Recall@K.
    
    Args:
        recommended: List of recommended item IDs
        relevant: Set of relevant (ground truth) item IDs
        k: Number of top recommendations to consider
        
    Returns:
        Recall@K score
    """
    if len(relevant) == 0:
        return 0.0
    
    recommended_k = set(recommended[:k])
    relevant_recommended = recommended_k.intersection(relevant)
    
    return len(relevant_recommended) / len(relevant)


def precision_at_k(recommended: List[str], relevant: Set[str], k: int) -> float:
    """
    Calculate Precision@K.
    
    Args:
        recommended: List of recommended item IDs
        relevant: Set of relevant (ground truth) item IDs
        k: Number of top recommendations to consider
        
    Returns:
        Precision@K score
    """
    if k == 0:
        return 0.0
    
    recommended_k = set(recommended[:k])
    relevant_recommended = recommended_k.intersection(relevant)
    
    return len(relevant_recommended) / k


def ndcg_at_k(recommended: List[str], relevant: Set[str], k: int) -> float:
    """
    Calculate Normalized Discounted Cumulative Gain (NDCG)@K.
    
    Args:
        recommended: List of recommended item IDs
        relevant: Set of relevant (ground truth) item IDs
        k: Number of top recommendations to consider
        
    Returns:
        NDCG@K score
    """
    if len(relevant) == 0:
        return 0.0
    
    # Calculate DCG
    dcg = 0.0
    for i, item in enumerate(recommended[:k]):
        if item in relevant:
            # Relevance is 1 if item is relevant, 0 otherwise
            relevance = 1.0
            dcg += relevance / np.log2(i + 2)  # i+2 because log2(1) = 0
    
    # Calculate IDCG (Ideal DCG)
    idcg = 0.0
    n_relevant = min(len(relevant), k)
    for i in range(n_relevant):
        idcg += 1.0 / np.log2(i + 2)
    
    if idcg == 0:
        return 0.0
    
    return dcg / idcg


def mean_reciprocal_rank(recommended: List[str], relevant: Set[str]) -> float:
    """
    Calculate Mean Reciprocal Rank (MRR).
    
    Args:
        recommended: List of recommended item IDs
        relevant: Set of relevant (ground truth) item IDs
        
    Returns:
        MRR score
    """
    if len(relevant) == 0:
        return 0.0
    
    for i, item in enumerate(recommended):
        if item in relevant:
            return 1.0 / (i + 1)
    
    return 0.0


def diversity(recommended: List[str], track_features: dict, feature_cols: List[str]) -> float:
    """
    Calculate diversity of recommendations based on feature variance.
    
    Args:
        recommended: List of recommended track IDs
        track_features: Dictionary mapping track ID to feature vector
        feature_cols: List of feature column names
        
    Returns:
        Diversity score (average pairwise distance)
    """
    if len(recommended) < 2:
        return 0.0
    
    from sklearn.metrics.pairwise import cosine_similarity
    
    # Get feature vectors for recommended tracks
    feature_vectors = []
    for track_id in recommended:
        if track_id in track_features:
            features = track_features[track_id]
            feature_vectors.append(features)
    
    if len(feature_vectors) < 2:
        return 0.0
    
    feature_matrix = np.array(feature_vectors)
    
    # Calculate pairwise distances (1 - similarity)
    similarities = cosine_similarity(feature_matrix)
    distances = 1 - similarities
    
    # Return mean pairwise distance (excluding diagonal)
    mask = ~np.eye(len(distances), dtype=bool)
    return np.mean(distances[mask])


def novelty(recommended: List[str], track_popularity: dict) -> float:
    """
    Calculate novelty of recommendations based on inverse popularity.
    
    Args:
        recommended: List of recommended track IDs
        track_popularity: Dictionary mapping track ID to popularity score
        
    Returns:
        Novelty score (average inverse popularity)
    """
    if len(recommended) == 0:
        return 0.0
    
    # Normalize popularity to [0, 1] if needed
    max_popularity = max(track_popularity.values()) if track_popularity.values() else 1.0
    min_popularity = min(track_popularity.values()) if track_popularity.values() else 0.0
    pop_range = max_popularity - min_popularity if max_popularity != min_popularity else 1.0
    
    novelty_scores = []
    for track_id in recommended:
        if track_id in track_popularity:
            # Normalize popularity
            normalized_pop = (track_popularity[track_id] - min_popularity) / pop_range
            # Novelty is inverse of popularity
            novelty = 1.0 - normalized_pop
            novelty_scores.append(novelty)
    
    return np.mean(novelty_scores) if novelty_scores else 0.0


def evaluate_recommendations(recommended: List[str], relevant: Set[str], 
                            k_values: List[int] = [5, 10, 20]) -> dict:
    """
    Comprehensive evaluation of recommendations.
    
    Args:
        recommended: List of recommended item IDs
        relevant: Set of relevant (ground truth) item IDs
        k_values: List of k values to evaluate at
        
    Returns:
        Dictionary with all evaluation metrics
    """
    results = {}
    
    for k in k_values:
        results[f'recall@{k}'] = recall_at_k(recommended, relevant, k)
        results[f'precision@{k}'] = precision_at_k(recommended, relevant, k)
        results[f'ndcg@{k}'] = ndcg_at_k(recommended, relevant, k)
    
    results['mrr'] = mean_reciprocal_rank(recommended, relevant)
    
    return results

