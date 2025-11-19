"""
Feature engineering for ranking model.
"""
import numpy as np
import pandas as pd
from typing import List, Dict, Optional
from ..data_loader import DataLoader


class FeatureEngineer:
    """Engineer features for ranking model."""
    
    def __init__(self, data_loader: DataLoader):
        """
        Initialize feature engineer.
        
        Args:
            data_loader: DataLoader instance
        """
        self.data_loader = data_loader
        self.data = data_loader.get_data()
    
    def extract_features(self, seed_songs: List[Dict], candidate_track_id: str, 
                        similarity_score: float) -> Dict:
        """
        Extract features for a candidate track given seed songs.
        
        Args:
            seed_songs: List of seed song dicts with 'name' key
            candidate_track_id: Candidate track ID
            similarity_score: Similarity score from embedding search
            
        Returns:
            Dictionary of features
        """
        candidate_track = self.data_loader.get_song_by_id(candidate_track_id)
        if candidate_track is None:
            return {}
        
        features = {}
        
        # Similarity score
        features['similarity_score'] = similarity_score
        
        # Track popularity
        features['popularity'] = candidate_track.get('popularity', 0)
        
        # Audio feature statistics from seed songs
        seed_features = []
        for song in seed_songs:
            track = self.data_loader.get_song_data(song['name'])
            if track is not None:
                audio_features = self.data_loader.get_numerical_features(track)
                seed_features.append(audio_features)
        
        if seed_features:
            seed_features = np.array(seed_features)
            seed_mean = np.mean(seed_features, axis=0)
            seed_std = np.std(seed_features, axis=0)
            
            candidate_features = self.data_loader.get_numerical_features(candidate_track)
            
            # Feature differences
            feature_cols = self.data_loader.NUMERICAL_COLS
            for i, col in enumerate(feature_cols):
                if col != 'popularity':  # Already included
                    features[f'mean_diff_{col}'] = abs(candidate_features[i] - seed_mean[i])
                    if seed_std[i] > 0:
                        features[f'std_normalized_diff_{col}'] = abs(
                            (candidate_features[i] - seed_mean[i]) / seed_std[i]
                        )
                    else:
                        features[f'std_normalized_diff_{col}'] = 0.0
        
        # Artist match (check if candidate artist matches any seed artist)
        candidate_artists = str(candidate_track.get('artists', '')).lower()
        artist_match = 0
        for song in seed_songs:
            track = self.data_loader.get_song_data(song['name'])
            if track is not None:
                seed_artists = str(track.get('artists', '')).lower()
                if seed_artists in candidate_artists or candidate_artists in seed_artists:
                    artist_match = 1
                    break
        features['artist_match'] = artist_match
        
        # Year difference
        candidate_year = candidate_track.get('year', 0)
        if seed_features:
            seed_years = [self.data_loader.get_song_data(s['name']).get('year', 0) 
                         for s in seed_songs 
                         if self.data_loader.get_song_data(s['name']) is not None]
            if seed_years:
                mean_seed_year = np.mean(seed_years)
                features['year_diff'] = abs(candidate_year - mean_seed_year)
                features['year_recency'] = candidate_year - 1920  # Years since earliest in dataset
            else:
                features['year_diff'] = 0
                features['year_recency'] = 0
        else:
            features['year_diff'] = 0
            features['year_recency'] = 0
        
        # Track duration features
        features['duration_ms'] = candidate_track.get('duration_ms', 0)
        
        # Explicit content
        features['explicit'] = candidate_track.get('explicit', 0)
        
        return features
    
    def extract_batch_features(self, seed_songs: List[Dict], 
                              candidate_tracks: List[tuple]) -> pd.DataFrame:
        """
        Extract features for multiple candidate tracks.
        
        Args:
            seed_songs: List of seed song dicts
            candidate_tracks: List of (track_id, similarity_score) tuples
            
        Returns:
            DataFrame with features for each candidate
        """
        feature_dicts = []
        track_ids = []
        
        for track_id, similarity_score in candidate_tracks:
            features = self.extract_features(seed_songs, track_id, similarity_score)
            if features:
                feature_dicts.append(features)
                track_ids.append(track_id)
        
        if not feature_dicts:
            return pd.DataFrame()
        
        df = pd.DataFrame(feature_dicts)
        df['track_id'] = track_ids
        
        return df

