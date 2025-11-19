"""
Approximate Nearest Neighbor (ANN) indexing for fast similarity search.
"""
import numpy as np
from typing import List, Optional, Literal
from pathlib import Path
import pickle

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

try:
    import hnswlib
    HNSWLIB_AVAILABLE = True
except ImportError:
    HNSWLIB_AVAILABLE = False


class ANNIndex:
    """Approximate Nearest Neighbor index for fast similarity search."""
    
    def __init__(self, method: Literal['faiss', 'hnsw'] = 'faiss', 
                 dimension: int = 128, metric: str = 'cosine'):
        """
        Initialize ANN index.
        
        Args:
            method: Indexing method ('faiss' or 'hnsw')
            dimension: Dimensionality of vectors
            metric: Distance metric ('cosine', 'l2', 'ip')
        """
        self.method = method
        self.dimension = dimension
        self.metric = metric
        self.index = None
        self.track_ids = []
        self.id_to_index = {}
        self.index_to_id = {}
        
        if method == 'faiss' and not FAISS_AVAILABLE:
            raise ImportError("faiss-cpu not installed. Install with: pip install faiss-cpu")
        if method == 'hnsw' and not HNSWLIB_AVAILABLE:
            raise ImportError("hnswlib not installed. Install with: pip install hnswlib")
    
    def build_index(self, embeddings: dict, ef_construction: int = 200, 
                   M: int = 16, ef_search: int = 50):
        """
        Build the ANN index from embeddings.
        
        Args:
            embeddings: Dictionary mapping track ID to embedding vector
            ef_construction: HNSW parameter for construction (only for hnsw)
            M: HNSW parameter for number of connections (only for hnsw)
            ef_search: HNSW parameter for search (only for hnsw)
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
        
        embedding_matrix = np.array(embedding_vectors).astype('float32')
        
        # Normalize for cosine similarity
        if self.metric == 'cosine':
            faiss.normalize_L2(embedding_matrix)
        
        # Build index based on method
        if self.method == 'faiss':
            self._build_faiss_index(embedding_matrix)
        elif self.method == 'hnsw':
            self._build_hnsw_index(embedding_matrix, ef_construction, M, ef_search)
    
    def _build_faiss_index(self, embedding_matrix: np.ndarray):
        """Build FAISS index."""
        dimension = embedding_matrix.shape[1]
        
        if self.metric == 'cosine':
            # Use Inner Product for cosine similarity (after normalization)
            index = faiss.IndexFlatIP(dimension)
        elif self.metric == 'l2':
            index = faiss.IndexFlatL2(dimension)
        else:
            raise ValueError(f"Unsupported metric for FAISS: {self.metric}")
        
        index.add(embedding_matrix)
        self.index = index
        self.dimension = dimension
    
    def _build_hnsw_index(self, embedding_matrix: np.ndarray, ef_construction: int, 
                         M: int, ef_search: int):
        """Build HNSW index."""
        dimension = embedding_matrix.shape[1]
        
        # Create HNSW index
        index = hnswlib.Index(space='cosine' if self.metric == 'cosine' else 'l2', 
                             dim=dimension)
        index.init_index(max_elements=len(embedding_matrix), ef_construction=ef_construction, M=M)
        index.set_ef(ef_search)
        
        # Add vectors
        index.add_items(embedding_matrix, list(range(len(embedding_matrix))))
        
        self.index = index
        self.dimension = dimension
    
    def search(self, query_vector: np.ndarray, k: int = 10) -> List[tuple]:
        """
        Search for k nearest neighbors.
        
        Args:
            query_vector: Query embedding vector
            k: Number of neighbors to retrieve
            
        Returns:
            List of (track_id, distance) tuples
        """
        if self.index is None:
            raise ValueError("Index not built. Call build_index() first.")
        
        query_vector = query_vector.astype('float32').reshape(1, -1)
        
        # Normalize for cosine similarity
        if self.metric == 'cosine':
            faiss.normalize_L2(query_vector)
        
        if self.method == 'faiss':
            distances, indices = self.index.search(query_vector, k)
            results = []
            for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
                if idx < len(self.track_ids):
                    track_id = self.index_to_id[idx]
                    # For cosine similarity, higher is better (convert distance to similarity)
                    if self.metric == 'cosine':
                        similarity = float(dist)
                    else:
                        similarity = 1.0 / (1.0 + float(dist))
                    results.append((track_id, similarity))
            return results
        
        elif self.method == 'hnsw':
            labels, distances = self.index.knn_query(query_vector[0], k=k)
            results = []
            for label, dist in zip(labels[0], distances[0]):
                track_id = self.index_to_id[label]
                if self.metric == 'cosine':
                    similarity = 1.0 - float(dist)  # HNSW returns distance
                else:
                    similarity = 1.0 / (1.0 + float(dist))
                results.append((track_id, similarity))
            return results
    
    def save(self, filepath: Path):
        """
        Save the index to disk.
        
        Args:
            filepath: Path to save the index
        """
        if self.index is None:
            raise ValueError("Index not built. Cannot save.")
        
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        if self.method == 'faiss':
            faiss.write_index(self.index, str(filepath))
        elif self.method == 'hnsw':
            self.index.save_index(str(filepath))
        
        # Save metadata
        metadata_path = filepath.with_suffix('.metadata.pkl')
        with open(metadata_path, 'wb') as f:
            pickle.dump({
                'method': self.method,
                'dimension': self.dimension,
                'metric': self.metric,
                'track_ids': self.track_ids,
                'id_to_index': self.id_to_index,
                'index_to_id': self.index_to_id
            }, f)
    
    def load(self, filepath: Path):
        """
        Load the index from disk.
        
        Args:
            filepath: Path to the index file
        """
        if self.method == 'faiss':
            self.index = faiss.read_index(str(filepath))
        elif self.method == 'hnsw':
            # Load metadata first to get dimension
            metadata_path = filepath.with_suffix('.metadata.pkl')
            with open(metadata_path, 'rb') as f:
                metadata = pickle.load(f)
            self.dimension = metadata['dimension']
            
            # Create and load index
            space = 'cosine' if self.metric == 'cosine' else 'l2'
            self.index = hnswlib.Index(space=space, dim=self.dimension)
            self.index.load_index(str(filepath))
        
        # Load metadata
        metadata_path = filepath.with_suffix('.metadata.pkl')
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)
        
        self.method = metadata['method']
        self.dimension = metadata['dimension']
        self.metric = metadata['metric']
        self.track_ids = metadata['track_ids']
        self.id_to_index = metadata['id_to_index']
        self.index_to_id = metadata['index_to_id']

