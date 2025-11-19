"""
Baseline content-based recommender using Euclidean distance.
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from scipy.spatial.distance import cdist
from ..data_loader import DataLoader


class BaselineRecommender:
    """Baseline content-based recommender using feature similarity."""
    
    def __init__(self, data_loader):
        """
        Initialize the baseline recommender.
        
        Args:
            data_loader: DataLoader instance
        """
        self.data_loader = data_loader
        self.data = data_loader.get_data()
        self.number_cols = data_loader.NUMERICAL_COLS
        
        # Initialize scalers
        self.min_max_scaler = MinMaxScaler()
        self.standard_scaler = StandardScaler()
        
        # Preprocess and scale the data
        self._prepare_data()
    
    def _prepare_data(self):
        """Preprocess and scale the numerical data."""
        # Get numerical features
        numerical_data = self.data_loader.get_all_numerical_features()
        
        # Convert to numpy array to avoid feature name warnings
        if isinstance(numerical_data, pd.DataFrame):
            numerical_array = numerical_data.values
        else:
            numerical_array = numerical_data
        
        # Normalize using Min-Max Scaler
        self.normalized_data = self.min_max_scaler.fit_transform(numerical_array)
        
        # Standardize the normalized data
        self.scaled_normalized_data = self.standard_scaler.fit_transform(self.normalized_data)
    
    def get_mean_vector(self, song_list):
        """
        Calculate the mean vector of a list of songs.
        
        Args:
            song_list: List of dicts with 'name' key
            
        Returns:
            numpy array of mean feature vector, or None if songs not found
        """
        song_vectors = []
        for song in song_list:
            song_data = self.data_loader.get_song_data(song['name'])
            if song_data is None:
                print(f"Warning: {song['name']} does not exist in the dataset")
                return None
            song_vector = self.data_loader.get_numerical_features(song_data)
            song_vectors.append(song_vector)
        
        if not song_vectors:
            return None
        
        song_matrix = np.array(song_vectors)
        return np.mean(song_matrix, axis=0)
    
    def recommend_songs(self, seed_songs, n_recommendations=10, distance_metric='euclidean'):
        """
        Recommend songs based on a list of seed songs.
        
        Args:
            seed_songs: List of dicts with 'name' key
            n_recommendations: Number of recommendations to return
            distance_metric: Distance metric to use ('euclidean' or 'cosine')
            
        Returns:
            List of dicts with recommended songs (name, artists, year)
        """
        metadata_cols = self.data_loader.METADATA_COLS
        song_center = self.get_mean_vector(seed_songs)
        
        # Return empty list if song_center is missing
        if song_center is None:
            return []
        
        # Normalize the song center
        normalized_song_center = self.min_max_scaler.transform([song_center])
        
        # Standardize the normalized song center
        scaled_normalized_song_center = self.standard_scaler.transform(normalized_song_center)
        
        # Calculate distances
        if distance_metric == 'euclidean':
            distances = cdist(scaled_normalized_song_center, self.scaled_normalized_data, 'euclidean')
        elif distance_metric == 'cosine':
            distances = cdist(scaled_normalized_song_center, self.scaled_normalized_data, 'cosine')
        else:
            raise ValueError(f"Unsupported distance metric: {distance_metric}")
        
        index = np.argsort(distances)[0]
        
        # Filter out seed songs and duplicates, then get top n_recommendations
        seed_song_names = {song['name'].lower() for song in seed_songs}
        rec_songs = []
        seen_names = set()
        
        for i in index:
            song_name = self.data.iloc[i]['name']
            song_name_lower = song_name.lower()
            
            if song_name_lower not in seed_song_names and song_name_lower not in seen_names:
                rec_songs.append(self.data.iloc[i])
                seen_names.add(song_name_lower)
                if len(rec_songs) == n_recommendations:
                    break
        
        # Convert to list of dicts
        return pd.DataFrame(rec_songs)[metadata_cols].to_dict(orient='records')

