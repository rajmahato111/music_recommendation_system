"""
Hybrid recommender combining collaborative and audio embeddings with ANN indexing.
"""
import numpy as np
from typing import List, Dict, Optional, Union
from ..embeddings.hybrid_embeddings import HybridEmbeddings
from ..retrieval.ann_index import ANNIndex
from ..data_loader import DataLoader


class HybridRecommender:
    """Hybrid recommender using fused embeddings and ANN search."""
    
    def __init__(self, hybrid_embeddings: HybridEmbeddings, 
                 ann_index: Union[ANNIndex, 'BruteForceIndex'],
                 data_loader: DataLoader):
        """
        Initialize hybrid recommender.
        
        Args:
            hybrid_embeddings: HybridEmbeddings instance
            ann_index: ANNIndex instance
            data_loader: DataLoader instance
        """
        self.hybrid_embeddings = hybrid_embeddings
        self.ann_index = ann_index
        self.data_loader = data_loader
        self.data = data_loader.get_data()
    
    def recommend_songs(self, seed_songs: List[Dict], n_recommendations: int = 10) -> List[Dict]:
        """
        Recommend songs based on seed songs using hybrid embeddings.
        
        Args:
            seed_songs: List of dicts with 'name' key
            n_recommendations: Number of recommendations to return
            
        Returns:
            List of dicts with recommended songs (name, artists, year)
        """
        # Get embeddings for seed songs
        seed_embeddings = []
        seed_track_ids = set()
        
        for song in seed_songs:
            track = self.data_loader.get_song_data(song['name'])
            if track is None:
                continue
            
            track_id = str(track['id'])
            seed_track_ids.add(track_id)
            
            embedding = self.hybrid_embeddings.get_embedding(track_id)
            if embedding is not None:
                seed_embeddings.append(embedding)
        
        if not seed_embeddings:
            return []
        
        # Average seed embeddings
        query_embedding = np.mean(seed_embeddings, axis=0)
        
        # Search for similar tracks
        search_k = n_recommendations + len(seed_track_ids)  # Get extra to filter seeds
        similar_tracks = self.ann_index.search(query_embedding, k=search_k)
        
        # Filter out seed songs and get top recommendations
        recommendations = []
        seen_names = set()
        
        for track_id, similarity in similar_tracks:
            if track_id in seed_track_ids:
                continue
            
            track = self.data_loader.get_song_by_id(track_id)
            if track is None:
                continue
            
            song_name = track['name']
            if song_name.lower() in seen_names:
                continue
            
            recommendations.append({
                'name': song_name,
                'artists': track.get('artists', ''),
                'year': track.get('year', ''),
                'similarity': similarity
            })
            seen_names.add(song_name.lower())
            
            if len(recommendations) >= n_recommendations:
                break
        
        return recommendations

