"""
Brute force similarity search index (fallback when FAISS/HNSW not available).
"""
import numpy as np
from typing import List, Optional
from pathlib import Path
import pickle
from scipy.spatial.distance import cdist


class BruteForceIndex:
    """Brute force similarity search index (slower but no dependencies)."""
    
    def __init__(self, dimension: int = 128, metric: str = 'cosine'):
        """
        Initialize brute force index.
        
        Args:
            dimension: Dimensionality of vectors
            metric: Distance metric ('cosine', 'euclidean')
        """
        self.dimension = dimension
        self.metric = metric
        self.embedding_matrix = None
        self.track_ids = []
        self.id_to_index = {}
        self.index_to_id = {}
    
    def build_index(self, embeddings: dict):
        """
        Build the index from embeddings.
        
        Args:
            embeddings: Dictionary mapping track ID to embedding vector
        """
        if not embeddings:
            raise ValueError("Empty embeddings dictionary")
        
        # Get all embeddings and track IDs
        self.track_ids = list(embeddings.keys())
        embedding_vectors = []
        
        for idx, track_id in enumerate(self.track_ids):
            embedding = embeddings[track_id]
            embedding_vectors.append(embedding)
            self.id_to_index[track_id] = idx
            self.index_to_id[idx] = track_id
        
        self.embedding_matrix = np.array(embedding_vectors).astype('float32')
        
        # Normalize for cosine similarity
        if self.metric == 'cosine':
            norms = np.linalg.norm(self.embedding_matrix, axis=1, keepdims=True)
            norms[norms == 0] = 1  # Avoid division by zero
            self.embedding_matrix = self.embedding_matrix / norms
    
    def search(self, query_vector: np.ndarray, k: int = 10) -> List[tuple]:
        """
        Search for k nearest neighbors using brute force.
        
        Args:
            query_vector: Query embedding vector
            k: Number of neighbors to retrieve
            
        Returns:
            List of (track_id, similarity) tuples
        """
        if self.embedding_matrix is None:
            raise ValueError("Index not built. Call build_index() first.")
        
        query_vector = query_vector.astype('float32').reshape(1, -1)
        
        # Normalize for cosine similarity
        if self.metric == 'cosine':
            norm = np.linalg.norm(query_vector)
            if norm > 0:
                query_vector = query_vector / norm
        
        # Calculate distances
        if self.metric == 'cosine':
            # Use dot product for cosine similarity (after normalization)
            similarities = np.dot(self.embedding_matrix, query_vector.T).flatten()
            # Get top k (highest similarities)
            top_indices = np.argsort(similarities)[::-1][:k]
            results = [(self.index_to_id[idx], float(similarities[idx])) for idx in top_indices]
        else:
            # Use euclidean distance
            distances = cdist(query_vector, self.embedding_matrix, metric='euclidean').flatten()
            top_indices = np.argsort(distances)[:k]
            results = [(self.index_to_id[idx], 1.0 / (1.0 + float(distances[idx]))) 
                      for idx in top_indices]
        
        return results
    
    def save(self, filepath: Path):
        """Save the index to disk."""
        if self.embedding_matrix is None:
            raise ValueError("Index not built. Cannot save.")
        
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Save as pickle
        with open(filepath, 'wb') as f:
            pickle.dump({
                'embedding_matrix': self.embedding_matrix,
                'track_ids': self.track_ids,
                'id_to_index': self.id_to_index,
                'index_to_id': self.index_to_id,
                'dimension': self.dimension,
                'metric': self.metric
            }, f)
    
    def load(self, filepath: Path):
        """Load the index from disk."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        self.embedding_matrix = data['embedding_matrix']
        self.track_ids = data['track_ids']
        self.id_to_index = data['id_to_index']
        self.index_to_id = data['index_to_id']
        self.dimension = data['dimension']
        self.metric = data['metric']

