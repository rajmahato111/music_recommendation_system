"""
Audio embeddings generation from Spotify audio features.
"""
import numpy as np
from typing import Dict, Optional
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from pathlib import Path
import pickle
from ..data_loader import DataLoader


class AudioEmbeddings:
    """Generate and manage audio embeddings from Spotify features."""
    
    def __init__(self, data_loader: DataLoader, use_pca=False, n_components=None):
        """
        Initialize audio embeddings generator.
        
        Args:
            data_loader: DataLoader instance
            use_pca: Whether to use PCA for dimensionality reduction
            n_components: Number of PCA components (if None, keeps all variance)
        """
        self.data_loader = data_loader
        self.use_pca = use_pca
        self.n_components = n_components
        self.scaler = StandardScaler()
        self.pca = None
        self.embeddings = {}
        self.feature_cols = data_loader.NUMERICAL_COLS
    
    def generate_embeddings(self):
        """
        Generate audio embeddings for all tracks.
        """
        # Get all numerical features
        numerical_data = self.data_loader.get_all_numerical_features()
        
        # Standardize features
        scaled_features = self.scaler.fit_transform(numerical_data)
        
        # Apply PCA if requested
        if self.use_pca:
            if self.n_components is None:
                # Keep 95% of variance
                self.pca = PCA(n_components=0.95)
            else:
                self.pca = PCA(n_components=self.n_components)
            
            scaled_features = self.pca.fit_transform(scaled_features)
        
        # Store embeddings indexed by track ID
        data = self.data_loader.get_data()
        for idx, row in data.iterrows():
            track_id = row['id']
            self.embeddings[str(track_id)] = scaled_features[idx]
    
    def get_embedding(self, track_id: str) -> Optional[np.ndarray]:
        """
        Get embedding for a track.
        
        Args:
            track_id: Track ID
            
        Returns:
            Embedding vector or None
        """
        return self.embeddings.get(str(track_id))
    
    def get_embedding_by_name(self, track_name: str) -> Optional[np.ndarray]:
        """
        Get embedding for a track by name.
        
        Args:
            track_name: Track name
            
        Returns:
            Embedding vector or None
        """
        track = self.data_loader.get_song_data(track_name)
        if track is None:
            return None
        
        track_id = track['id']
        return self.get_embedding(str(track_id))
    
    def get_all_embeddings(self) -> Dict[str, np.ndarray]:
        """
        Get all embeddings.
        
        Returns:
            Dictionary mapping track ID to embedding vector
        """
        return self.embeddings
    
    def get_embedding_dim(self) -> int:
        """
        Get the dimensionality of embeddings.
        
        Returns:
            Embedding dimensionality
        """
        if not self.embeddings:
            return len(self.feature_cols)
        
        # Get dimension from first embedding
        first_embedding = next(iter(self.embeddings.values()))
        return len(first_embedding)
    
    def save(self, filepath: Path):
        """
        Save the embeddings and preprocessing objects.
        
        Args:
            filepath: Path to save the embeddings
        """
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Save embeddings
        embeddings_path = filepath.with_suffix('.embeddings.pkl')
        with open(embeddings_path, 'wb') as f:
            pickle.dump(self.embeddings, f)
        
        # Save preprocessing objects
        preprocess_path = filepath.with_suffix('.preprocess.pkl')
        preprocess_data = {
            'scaler': self.scaler,
            'pca': self.pca,
            'use_pca': self.use_pca,
            'n_components': self.n_components,
            'feature_cols': self.feature_cols
        }
        with open(preprocess_path, 'wb') as f:
            pickle.dump(preprocess_data, f)
    
    def load(self, filepath: Path):
        """
        Load embeddings and preprocessing objects.
        
        Args:
            filepath: Path to the embeddings file
        """
        # Load embeddings
        embeddings_path = filepath.with_suffix('.embeddings.pkl')
        with open(embeddings_path, 'rb') as f:
            self.embeddings = pickle.load(f)
        
        # Load preprocessing objects
        preprocess_path = filepath.with_suffix('.preprocess.pkl')
        with open(preprocess_path, 'rb') as f:
            preprocess_data = pickle.load(f)
        
        self.scaler = preprocess_data['scaler']
        self.pca = preprocess_data['pca']
        self.use_pca = preprocess_data['use_pca']
        self.n_components = preprocess_data['n_components']
        self.feature_cols = preprocess_data['feature_cols']

