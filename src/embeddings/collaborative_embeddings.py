"""
Collaborative embeddings generation using Item2Vec.
"""
import numpy as np
from typing import Dict, List, Optional
from pathlib import Path
from ..models.item2vec import Item2Vec
from ..data_loader import DataLoader


class CollaborativeEmbeddings:
    """Generate and manage collaborative embeddings from Item2Vec."""
    
    def __init__(self, data_loader: DataLoader, vector_size=128):
        """
        Initialize collaborative embeddings generator.
        
        Args:
            data_loader: DataLoader instance
            vector_size: Dimensionality of embeddings
        """
        self.data_loader = data_loader
        self.vector_size = vector_size
        self.item2vec = None
        self.embeddings = {}
    
    def train(self, playlists: List[List[str]], **item2vec_kwargs):
        """
        Train Item2Vec model on playlists.
        
        Args:
            playlists: List of playlists (each playlist is a list of track IDs)
            **item2vec_kwargs: Additional arguments for Item2Vec initialization
        """
        # Initialize Item2Vec
        self.item2vec = Item2Vec(vector_size=self.vector_size, **item2vec_kwargs)
        
        # Train the model
        self.item2vec.train(playlists)
        
        # Get all embeddings
        self.embeddings = self.item2vec.get_all_embeddings()
    
    def get_embedding(self, track_id: str) -> Optional[np.ndarray]:
        """
        Get embedding for a track.
        
        Args:
            track_id: Track ID
            
        Returns:
            Embedding vector or None
        """
        if self.item2vec is None:
            raise ValueError("Model not trained. Call train() first.")
        
        return self.item2vec.get_embedding(track_id)
    
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
    
    def save(self, filepath: Path):
        """
        Save the trained model and embeddings.
        
        Args:
            filepath: Path to save the model
        """
        if self.item2vec is None:
            raise ValueError("Model not trained. Cannot save.")
        
        self.item2vec.save(filepath)
    
    def load(self, filepath: Path):
        """
        Load a trained model.
        
        Args:
            filepath: Path to the model file
        """
        self.item2vec = Item2Vec()
        self.item2vec.load(filepath)
        self.embeddings = self.item2vec.get_all_embeddings()
        self.vector_size = self.item2vec.vector_size

