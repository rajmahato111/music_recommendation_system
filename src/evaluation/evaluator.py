"""
Evaluation pipeline for the recommendation system.
"""
import numpy as np
from typing import List, Dict, Set, Callable
from .metrics import (
    recall_at_k, precision_at_k, ndcg_at_k, 
    mean_reciprocal_rank, diversity, novelty, evaluate_recommendations
)
from ..data_loader import DataLoader


class RecommenderEvaluator:
    """Evaluates recommendation system performance."""
    
    def __init__(self, data_loader: DataLoader):
        """
        Initialize the evaluator.
        
        Args:
            data_loader: DataLoader instance
        """
        self.data_loader = data_loader
        self.data = data_loader.get_data()
    
    def evaluate_on_playlists(self, recommender, test_playlists: List[List[str]], 
                             k_values: List[int] = [5, 10, 20], 
                             n_recommendations: int = 20) -> Dict:
        """
        Evaluate recommender on test playlists.
        
        Args:
            recommender: Recommender instance with recommend_songs method
            test_playlists: List of test playlists (each playlist is a list of track IDs)
            k_values: List of k values for evaluation
            n_recommendations: Number of recommendations to generate
            
        Returns:
            Dictionary with aggregated evaluation metrics
        """
        all_results = []
        
        for playlist in test_playlists:
            if len(playlist) < 2:
                continue
            
            # Use first track as seed, rest as ground truth
            seed_track_id = playlist[0]
            ground_truth = set(playlist[1:])
            
            # Get seed track name
            seed_track = self.data_loader.get_song_by_id(seed_track_id)
            if seed_track is None:
                continue
            
            seed_songs = [{'name': seed_track['name']}]
            
            # Get recommendations
            try:
                recommended = recommender.recommend_songs(
                    seed_songs, 
                    n_recommendations=n_recommendations
                )
                
                # Convert recommendations to track IDs
                recommended_ids = []
                for rec in recommended:
                    # Find track ID by name
                    track = self.data_loader.get_song_data(rec['name'])
                    if track is not None:
                        recommended_ids.append(track['id'])
                
                # Evaluate
                results = evaluate_recommendations(recommended_ids, ground_truth, k_values)
                all_results.append(results)
                
            except Exception as e:
                print(f"Error evaluating playlist: {e}")
                continue
        
        # Aggregate results
        if not all_results:
            return {}
        
        aggregated = {}
        for key in all_results[0].keys():
            values = [r[key] for r in all_results if key in r]
            aggregated[key] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'median': np.median(values)
            }
        
        return aggregated
    
    def evaluate_diversity_novelty(self, recommender, seed_songs: List[Dict], 
                                   n_recommendations: int = 20) -> Dict:
        """
        Evaluate diversity and novelty of recommendations.
        
        Args:
            recommender: Recommender instance
            seed_songs: List of seed songs
            n_recommendations: Number of recommendations
            
        Returns:
            Dictionary with diversity and novelty scores
        """
        # Get recommendations
        recommended = recommender.recommend_songs(
            seed_songs, 
            n_recommendations=n_recommendations
        )
        
        if not recommended:
            return {'diversity': 0.0, 'novelty': 0.0}
        
        # Get track IDs and features
        recommended_ids = []
        track_features = {}
        track_popularity = {}
        
        for rec in recommended:
            track = self.data_loader.get_song_data(rec['name'])
            if track is not None:
                track_id = track['id']
                recommended_ids.append(track_id)
                
                # Get features
                features = self.data_loader.get_numerical_features(track)
                track_features[track_id] = features
                
                # Get popularity
                if 'popularity' in track:
                    track_popularity[track_id] = track['popularity']
        
        # Calculate diversity
        div_score = diversity(
            recommended_ids, 
            track_features, 
            self.data_loader.NUMERICAL_COLS
        )
        
        # Calculate novelty
        nov_score = novelty(recommended_ids, track_popularity)
        
        return {
            'diversity': div_score,
            'novelty': nov_score
        }

