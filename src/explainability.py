"""
Explainability features for recommendations.
"""
from typing import List, Dict, Optional
from .data_loader import DataLoader
from .recommender.baseline_recommender import BaselineRecommender

# Optional import for hybrid recommender
try:
    from .recommender.hybrid_recommender import HybridRecommender
except ImportError:
    HybridRecommender = None


class RecommendationExplainer:
    """Generate explanations for recommendations."""
    
    def __init__(self, data_loader: DataLoader):
        """
        Initialize explainer.
        
        Args:
            data_loader: DataLoader instance
        """
        self.data_loader = data_loader
    
    def explain_baseline_recommendation(self, recommender: BaselineRecommender,
                                       seed_songs: List[Dict], 
                                       recommended_song: Dict) -> Dict:
        """
        Explain why a song was recommended by the baseline recommender.
        
        Args:
            recommender: BaselineRecommender instance
            seed_songs: List of seed songs
            recommended_song: Recommended song dict
            
        Returns:
            Dictionary with explanation
        """
        explanation = {
            'reason': 'Because you listened to:',
            'seed_songs': [song['name'] for song in seed_songs],
            'similarity_reason': 'This song has similar audio features to your seed songs.'
        }
        
        # Get feature similarities
        seed_track = self.data_loader.get_song_data(seed_songs[0]['name'])
        rec_track = self.data_loader.get_song_data(recommended_song['name'])
        
        if seed_track is not None and rec_track is not None:
            seed_features = self.data_loader.get_numerical_features(seed_track)
            rec_features = self.data_loader.get_numerical_features(rec_track)
            
            # Find most similar features
            feature_cols = self.data_loader.NUMERICAL_COLS
            feature_diffs = {}
            for i, col in enumerate(feature_cols):
                diff = abs(seed_features[i] - rec_features[i])
                feature_diffs[col] = diff
            
            # Get top 3 most similar features
            sorted_features = sorted(feature_diffs.items(), key=lambda x: x[1])[:3]
            explanation['similar_features'] = [feat[0] for feat in sorted_features]
        
        return explanation
    
    def explain_hybrid_recommendation(self, hybrid_recommender,
                                     seed_songs: List[Dict],
                                     recommended_song: Dict) -> Dict:
        """
        Explain why a song was recommended by the hybrid recommender.
        
        Args:
            hybrid_recommender: HybridRecommender instance
            seed_songs: List of seed songs
            recommended_song: Recommended song dict
            
        Returns:
            Dictionary with explanation
        """
        explanation = {
            'reason': 'Because you listened to:',
            'seed_songs': [song['name'] for song in seed_songs],
            'hybrid_reason': 'This song was recommended based on both collaborative filtering (similar playlists) and audio content similarity.'
        }
        
        # Get similarity score if available
        if 'similarity' in recommended_song:
            explanation['similarity_score'] = recommended_song['similarity']
            explanation['confidence'] = 'high' if recommended_song['similarity'] > 0.8 else 'medium' if recommended_song['similarity'] > 0.6 else 'low'
        
        return explanation
    
    def format_explanation(self, explanation: Dict) -> str:
        """
        Format explanation as a readable string.
        
        Args:
            explanation: Explanation dictionary
            
        Returns:
            Formatted explanation string
        """
        lines = []
        
        if 'reason' in explanation:
            lines.append(explanation['reason'])
        
        if 'seed_songs' in explanation:
            seed_list = ', '.join(explanation['seed_songs'])
            lines.append(f"  • {seed_list}")
        
        lines.append("")
        
        if 'similarity_reason' in explanation:
            lines.append(explanation['similarity_reason'])
        
        if 'hybrid_reason' in explanation:
            lines.append(explanation['hybrid_reason'])
        
        if 'similar_features' in explanation:
            features = ', '.join(explanation['similar_features'])
            lines.append(f"Similar features: {features}")
        
        if 'confidence' in explanation:
            lines.append(f"Confidence: {explanation['confidence']}")
        
        return "\n".join(lines)

