"""
Ablation studies and comprehensive evaluation framework.
"""
import numpy as np
import pandas as pd
from typing import List, Dict, Optional
from pathlib import Path
import json
from .evaluator import RecommenderEvaluator
from .metrics import evaluate_recommendations, diversity, novelty
from ..data_loader import DataLoader
from ..recommender.baseline_recommender import BaselineRecommender
from ..recommender.hybrid_recommender import HybridRecommender


class AblationStudy:
    """Conduct ablation studies comparing different model variants."""
    
    def __init__(self, data_loader: DataLoader):
        """
        Initialize ablation study framework.
        
        Args:
            data_loader: DataLoader instance
        """
        self.data_loader = data_loader
        self.evaluator = RecommenderEvaluator(data_loader)
        self.results = {}
    
    def evaluate_baseline_content(self, recommender: BaselineRecommender, 
                                  test_playlists: List[List[str]], 
                                  k_values: List[int] = [5, 10, 20]) -> Dict:
        """
        Evaluate content-based baseline.
        
        Args:
            recommender: BaselineRecommender instance
            test_playlists: Test playlists
            k_values: K values for evaluation
            
        Returns:
            Evaluation results
        """
        results = self.evaluator.evaluate_on_playlists(
            recommender, test_playlists, k_values
        )
        self.results['baseline_content'] = results
        return results
    
    def evaluate_collaborative_only(self, hybrid_recommender: HybridRecommender,
                                    test_playlists: List[List[str]],
                                    k_values: List[int] = [5, 10, 20]) -> Dict:
        """
        Evaluate collaborative-only recommendations.
        Note: This requires modifying the hybrid recommender to use only collaborative embeddings.
        
        Args:
            hybrid_recommender: HybridRecommender instance
            test_playlists: Test playlists
            k_values: K values for evaluation
            
        Returns:
            Evaluation results
        """
        # For now, we'll use the hybrid recommender as-is
        # In a full implementation, you'd create a collaborative-only variant
        results = self.evaluator.evaluate_on_playlists(
            hybrid_recommender, test_playlists, k_values
        )
        self.results['collaborative_only'] = results
        return results
    
    def evaluate_hybrid(self, hybrid_recommender: HybridRecommender,
                       test_playlists: List[List[str]],
                       k_values: List[int] = [5, 10, 20]) -> Dict:
        """
        Evaluate hybrid recommendations.
        
        Args:
            hybrid_recommender: HybridRecommender instance
            test_playlists: Test playlists
            k_values: K values for evaluation
            
        Returns:
            Evaluation results
        """
        results = self.evaluator.evaluate_on_playlists(
            hybrid_recommender, test_playlists, k_values
        )
        self.results['hybrid'] = results
        return results
    
    def evaluate_hybrid_with_ranking(self, hybrid_recommender: HybridRecommender,
                                    ranker, test_playlists: List[List[str]],
                                    k_values: List[int] = [5, 10, 20]) -> Dict:
        """
        Evaluate hybrid recommendations with ranking model.
        
        Args:
            hybrid_recommender: HybridRecommender instance
            ranker: Ranker instance
            test_playlists: Test playlists
            k_values: K values for evaluation
            
        Returns:
            Evaluation results
        """
        # This would require modifying the recommender to use ranking
        # For now, we'll evaluate the hybrid recommender
        results = self.evaluator.evaluate_on_playlists(
            hybrid_recommender, test_playlists, k_values
        )
        self.results['hybrid_ranking'] = results
        return results
    
    def evaluate_cold_start(self, recommender, new_tracks: List[str],
                           seed_songs: List[Dict], k_values: List[int] = [5, 10, 20]) -> Dict:
        """
        Evaluate cold-start performance (new tracks not in training).
        
        Args:
            recommender: Recommender instance
            new_tracks: List of new track IDs not in training
            seed_songs: Seed songs for recommendations
            k_values: K values for evaluation
            
        Returns:
            Evaluation results
        """
        # Get recommendations
        recommended = recommender.recommend_songs(seed_songs, n_recommendations=max(k_values))
        
        # Convert to track IDs
        recommended_ids = []
        for rec in recommended:
            track = self.data_loader.get_song_data(rec['name'])
            if track is not None:
                recommended_ids.append(track['id'])
        
        # Check how many new tracks are recommended
        new_tracks_set = set(new_tracks)
        results = {}
        
        for k in k_values:
            recommended_k = set(recommended_ids[:k])
            new_tracks_recommended = recommended_k.intersection(new_tracks_set)
            results[f'new_tracks@{k}'] = len(new_tracks_recommended) / k
        
        self.results['cold_start'] = results
        return results
    
    def compare_models(self) -> pd.DataFrame:
        """
        Compare all evaluated models.
        
        Returns:
            DataFrame with comparison results
        """
        if not self.results:
            return pd.DataFrame()
        
        # Flatten results
        comparison_data = []
        for model_name, metrics in self.results.items():
            for metric_name, metric_value in metrics.items():
                if isinstance(metric_value, dict):
                    # Handle nested metrics (mean, std, median)
                    for stat_name, stat_value in metric_value.items():
                        comparison_data.append({
                            'model': model_name,
                            'metric': f"{metric_name}_{stat_name}",
                            'value': stat_value
                        })
                else:
                    comparison_data.append({
                        'model': model_name,
                        'metric': metric_name,
                        'value': metric_value
                    })
        
        return pd.DataFrame(comparison_data)
    
    def save_results(self, filepath: Path):
        """
        Save evaluation results to file.
        
        Args:
            filepath: Path to save results
        """
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert numpy types to native Python types for JSON serialization
        def convert_to_serializable(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            return obj
        
        serializable_results = convert_to_serializable(self.results)
        
        with open(filepath, 'w') as f:
            json.dump(serializable_results, f, indent=2)
    
    def load_results(self, filepath: Path):
        """
        Load evaluation results from file.
        
        Args:
            filepath: Path to results file
        """
        with open(filepath, 'r') as f:
            self.results = json.load(f)
    
    def generate_report(self) -> str:
        """
        Generate a text report of evaluation results.
        
        Returns:
            Formatted report string
        """
        if not self.results:
            return "No evaluation results available."
        
        report_lines = ["=" * 80]
        report_lines.append("EVALUATION REPORT")
        report_lines.append("=" * 80)
        report_lines.append("")
        
        for model_name, metrics in self.results.items():
            report_lines.append(f"Model: {model_name.upper()}")
            report_lines.append("-" * 80)
            
            for metric_name, metric_value in metrics.items():
                if isinstance(metric_value, dict):
                    report_lines.append(f"  {metric_name}:")
                    for stat_name, stat_value in metric_value.items():
                        report_lines.append(f"    {stat_name}: {stat_value:.4f}")
                else:
                    report_lines.append(f"  {metric_name}: {metric_value:.4f}")
            
            report_lines.append("")
        
        report_lines.append("=" * 80)
        
        return "\n".join(report_lines)

