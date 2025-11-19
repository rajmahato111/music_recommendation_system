"""
Data preprocessing utilities for the music recommendation system.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import json
from typing import List, Dict, Tuple
from .data_loader import DataLoader


class DataPreprocessor:
    """Handles data preprocessing, cleaning, and splitting."""
    
    def __init__(self, data_loader: DataLoader):
        """
        Initialize the data preprocessor.
        
        Args:
            data_loader: DataLoader instance
        """
        self.data_loader = data_loader
        self.data = data_loader.get_data()
    
    def clean_data(self):
        """
        Clean the dataset by removing duplicates and handling missing values.
        
        Returns:
            Cleaned DataFrame
        """
        data = self.data.copy()
        
        # Remove duplicates based on track ID
        if 'id' in data.columns:
            data = data.drop_duplicates(subset=['id'], keep='first')
        
        # Handle missing values in numerical columns
        numerical_cols = self.data_loader.NUMERICAL_COLS
        for col in numerical_cols:
            if col in data.columns:
                # Fill missing values with median
                data[col] = data[col].fillna(data[col].median())
        
        # Remove rows with missing song names
        data = data.dropna(subset=['name'])
        
        return data
    
    def create_synthetic_playlists(self, n_playlists=10000, min_tracks=5, max_tracks=50, 
                                   similarity_threshold=0.7, random_seed=42):
        """
        Create synthetic playlists based on audio feature similarity.
        This simulates playlist co-occurrence patterns.
        
        Args:
            n_playlists: Number of playlists to generate
            min_tracks: Minimum tracks per playlist
            max_tracks: Maximum tracks per playlist
            similarity_threshold: Similarity threshold for grouping tracks
            random_seed: Random seed for reproducibility
            
        Returns:
            List of playlists, where each playlist is a list of track IDs
        """
        np.random.seed(random_seed)
        data = self.data.copy()
        
        # Get numerical features and normalize
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics.pairwise import cosine_similarity
        
        numerical_data = data[self.data_loader.NUMERICAL_COLS].values
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(numerical_data)
        
        # Calculate similarity matrix (sample for efficiency)
        # For large datasets, we'll use a sampling approach
        n_samples = min(5000, len(data))
        sample_indices = np.random.choice(len(data), n_samples, replace=False)
        sample_features = scaled_features[sample_indices]
        
        # Calculate pairwise similarities
        similarity_matrix = cosine_similarity(sample_features)
        
        playlists = []
        used_tracks = set()
        
        for _ in range(n_playlists):
            # Select a random seed track
            seed_idx = np.random.choice(len(sample_indices))
            seed_track_id = data.iloc[sample_indices[seed_idx]]['id']
            
            # Find similar tracks
            similarities = similarity_matrix[seed_idx]
            similar_indices = np.where(similarities >= similarity_threshold)[0]
            
            # Create playlist with similar tracks
            playlist_size = np.random.randint(min_tracks, max_tracks + 1)
            playlist_size = min(playlist_size, len(similar_indices))
            
            if playlist_size > 0:
                selected_indices = np.random.choice(
                    similar_indices, 
                    size=min(playlist_size, len(similar_indices)), 
                    replace=False
                )
                playlist_tracks = [
                    data.iloc[sample_indices[idx]]['id'] 
                    for idx in selected_indices
                ]
                playlists.append(playlist_tracks)
                used_tracks.update(playlist_tracks)
        
        # Add some random playlists for diversity
        n_random_playlists = n_playlists // 10
        for _ in range(n_random_playlists):
            playlist_size = np.random.randint(min_tracks, max_tracks + 1)
            random_tracks = np.random.choice(
                data['id'].values, 
                size=min(playlist_size, len(data)), 
                replace=False
            ).tolist()
            playlists.append(random_tracks)
        
        return playlists
    
    def split_data_temporal(self, playlists: List[List[str]], train_ratio=0.7, 
                           val_ratio=0.15, test_ratio=0.15):
        """
        Split playlists into train/validation/test sets preserving temporal order.
        
        Args:
            playlists: List of playlists (each playlist is a list of track IDs)
            train_ratio: Ratio of training data
            val_ratio: Ratio of validation data
            test_ratio: Ratio of test data
            
        Returns:
            Tuple of (train_playlists, val_playlists, test_playlists)
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
            "Ratios must sum to 1.0"
        
        n_playlists = len(playlists)
        n_train = int(n_playlists * train_ratio)
        n_val = int(n_playlists * val_ratio)
        
        train_playlists = playlists[:n_train]
        val_playlists = playlists[n_train:n_train + n_val]
        test_playlists = playlists[n_train + n_val:]
        
        return train_playlists, val_playlists, test_playlists
    
    def save_playlists(self, playlists: List[List[str]], filepath: Path):
        """
        Save playlists to a JSON file.
        
        Args:
            playlists: List of playlists
            filepath: Path to save the playlists
        """
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(playlists, f, indent=2)
    
    def load_playlists(self, filepath: Path) -> List[List[str]]:
        """
        Load playlists from a JSON file.
        
        Args:
            filepath: Path to the playlists file
            
        Returns:
            List of playlists
        """
        with open(filepath, 'r') as f:
            playlists = json.load(f)
        return playlists
    
    def create_track_sequences(self, playlists: List[List[str]]) -> List[List[str]]:
        """
        Convert playlists to track sequences for Item2Vec training.
        
        Args:
            playlists: List of playlists
            
        Returns:
            List of track sequences (same as playlists for Item2Vec)
        """
        return playlists
    
    def get_statistics(self, playlists: List[List[str]]) -> Dict:
        """
        Get statistics about the playlist dataset.
        
        Args:
            playlists: List of playlists
            
        Returns:
            Dictionary with statistics
        """
        playlist_lengths = [len(playlist) for playlist in playlists]
        all_tracks = set()
        for playlist in playlists:
            all_tracks.update(playlist)
        
        return {
            'n_playlists': len(playlists),
            'n_unique_tracks': len(all_tracks),
            'avg_playlist_length': np.mean(playlist_lengths),
            'min_playlist_length': np.min(playlist_lengths),
            'max_playlist_length': np.max(playlist_lengths),
            'median_playlist_length': np.median(playlist_lengths)
        }

