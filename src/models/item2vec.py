"""
Item2Vec model for collaborative filtering based on playlist co-occurrences.
"""
import numpy as np
from gensim.models import Word2Vec
from typing import List, Dict, Optional
from pathlib import Path
import pickle


class Item2Vec:
    """Item2Vec model for learning track embeddings from playlist sequences."""
    
    def __init__(self, vector_size=128, window=5, min_count=2, workers=4, 
                 sg=1, epochs=10, random_seed=42):
        """
        Initialize Item2Vec model.
        
        Args:
            vector_size: Dimensionality of the feature vectors
            window: Maximum distance between the current and predicted word
            min_count: Ignores all words with total frequency lower than this
            workers: Number of worker threads
            sg: Training algorithm: 1 for skip-gram, 0 for CBOW
            epochs: Number of iterations over the corpus
            random_seed: Random seed for reproducibility
        """
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.workers = workers
        self.sg = sg
        self.epochs = epochs
        self.random_seed = random_seed
        self.model = None
        self.track_to_id = {}
        self.id_to_track = {}
    
    def train(self, playlists: List[List[str]], track_id_mapping: Optional[Dict] = None):
        """
        Train Item2Vec model on playlist sequences.
        
        Args:
            playlists: List of playlists, where each playlist is a list of track IDs
            track_id_mapping: Optional mapping from track ID to string identifier
        """
        # Convert track IDs to strings if needed
        if track_id_mapping is None:
            # Use track IDs directly as strings
            sequences = [[str(track_id) for track_id in playlist] for playlist in playlists]
        else:
            # Map track IDs to string identifiers
            sequences = []
            for playlist in playlists:
                seq = [str(track_id_mapping.get(track_id, track_id)) for track_id in playlist]
                sequences.append(seq)
        
        # Train Word2Vec model
        self.model = Word2Vec(
            sentences=sequences,
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            workers=self.workers,
            sg=self.sg,
            epochs=self.epochs,
            seed=self.random_seed
        )
        
        # Build track ID mappings
        if track_id_mapping:
            self.track_to_id = {v: k for k, v in track_id_mapping.items()}
            self.id_to_track = track_id_mapping
        else:
            # Use track IDs as strings
            vocab = list(self.model.wv.key_to_index.keys())
            self.track_to_id = {track_id: track_id for track_id in vocab}
            self.id_to_track = {track_id: track_id for track_id in vocab}
    
    def get_embedding(self, track_id: str) -> Optional[np.ndarray]:
        """
        Get embedding vector for a track.
        
        Args:
            track_id: Track ID
            
        Returns:
            Embedding vector or None if track not in vocabulary
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        track_id_str = str(track_id)
        if track_id_str in self.model.wv:
            return self.model.wv[track_id_str]
        return None
    
    def get_all_embeddings(self) -> Dict[str, np.ndarray]:
        """
        Get embeddings for all tracks in the vocabulary.
        
        Returns:
            Dictionary mapping track ID to embedding vector
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        embeddings = {}
        for track_id in self.model.wv.key_to_index:
            embeddings[track_id] = self.model.wv[track_id]
        
        return embeddings
    
    def get_similar_tracks(self, track_id: str, topn=10) -> List[tuple]:
        """
        Get most similar tracks to a given track.
        
        Args:
            track_id: Track ID
            topn: Number of similar tracks to return
            
        Returns:
            List of (track_id, similarity_score) tuples
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        track_id_str = str(track_id)
        if track_id_str not in self.model.wv:
            return []
        
        similar = self.model.wv.most_similar(track_id_str, topn=topn)
        return similar
    
    def save(self, filepath: Path):
        """
        Save the trained model.
        
        Args:
            filepath: Path to save the model
        """
        if self.model is None:
            raise ValueError("Model not trained. Cannot save.")
        
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Save Word2Vec model
        model_path = filepath.with_suffix('.model')
        self.model.save(str(model_path))
        
        # Save metadata
        metadata = {
            'vector_size': self.vector_size,
            'window': self.window,
            'min_count': self.min_count,
            'track_to_id': self.track_to_id,
            'id_to_track': self.id_to_track
        }
        
        metadata_path = filepath.with_suffix('.metadata.pkl')
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)
    
    def load(self, filepath: Path):
        """
        Load a trained model.
        
        Args:
            filepath: Path to the model file
        """
        # Load Word2Vec model
        model_path = filepath.with_suffix('.model')
        self.model = Word2Vec.load(str(model_path))
        
        # Load metadata
        metadata_path = filepath.with_suffix('.metadata.pkl')
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)
        
        self.vector_size = metadata['vector_size']
        self.window = metadata['window']
        self.min_count = metadata['min_count']
        self.track_to_id = metadata['track_to_id']
        self.id_to_track = metadata['id_to_track']

