"""
Hybrid embeddings combining collaborative and audio embeddings.
"""
import numpy as np
from typing import Dict, Optional, Literal
from sklearn.preprocessing import normalize
from pathlib import Path
import pickle
from .collaborative_embeddings import CollaborativeEmbeddings
from .audio_embeddings import AudioEmbeddings


class HybridEmbeddings:
    """Combine collaborative and audio embeddings into hybrid representations."""
    
    def __init__(self, collaborative_embeddings: CollaborativeEmbeddings, 
                 audio_embeddings: AudioEmbeddings,
                 fusion_method: Literal['concatenate', 'weighted_sum', 'learned'] = 'concatenate',
                 collaborative_weight: float = 0.5,
                 audio_weight: float = 0.5):
        """
        Initialize hybrid embeddings.
        
        Args:
            collaborative_embeddings: CollaborativeEmbeddings instance
            audio_embeddings: AudioEmbeddings instance
            fusion_method: Method to fuse embeddings ('concatenate', 'weighted_sum', 'learned')
            collaborative_weight: Weight for collaborative embeddings (for weighted_sum)
            audio_weight: Weight for audio embeddings (for weighted_sum)
        """
        self.collaborative_embeddings = collaborative_embeddings
        self.audio_embeddings = audio_embeddings
        self.fusion_method = fusion_method
        self.collaborative_weight = collaborative_weight
        self.audio_weight = audio_weight
        self.embeddings = {}
        self._generate_hybrid_embeddings()
    
    def _generate_hybrid_embeddings(self):
        """Generate hybrid embeddings for all tracks."""
        # Get all track IDs from both embedding sources
        collab_embeddings = self.collaborative_embeddings.get_all_embeddings()
        audio_embeddings = self.audio_embeddings.get_all_embeddings()
        
        # Find common tracks
        common_tracks = set(collab_embeddings.keys()) & set(audio_embeddings.keys())
        
        for track_id in common_tracks:
            collab_emb = collab_embeddings[track_id]
            audio_emb = audio_embeddings[track_id]
            
            # Normalize individual embeddings
            collab_emb_norm = normalize([collab_emb])[0]
            audio_emb_norm = normalize([audio_emb])[0]
            
            # Fuse embeddings
            if self.fusion_method == 'concatenate':
                hybrid_emb = np.concatenate([collab_emb_norm, audio_emb_norm])
            elif self.fusion_method == 'weighted_sum':
                # Ensure same dimensionality for weighted sum
                if len(collab_emb_norm) != len(audio_emb_norm):
                    # Use PCA or padding to match dimensions
                    min_dim = min(len(collab_emb_norm), len(audio_emb_norm))
                    collab_emb_norm = collab_emb_norm[:min_dim]
                    audio_emb_norm = audio_emb_norm[:min_dim]
                hybrid_emb = (self.collaborative_weight * collab_emb_norm + 
                             self.audio_weight * audio_emb_norm)
            else:
                raise ValueError(f"Unknown fusion method: {self.fusion_method}")
            
            # Normalize final hybrid embedding
            hybrid_emb = normalize([hybrid_emb])[0]
            self.embeddings[track_id] = hybrid_emb
    
    def get_embedding(self, track_id: str) -> Optional[np.ndarray]:
        """
        Get hybrid embedding for a track.
        
        Args:
            track_id: Track ID
            
        Returns:
            Hybrid embedding vector or None
        """
        return self.embeddings.get(str(track_id))
    
    def get_embedding_by_name(self, track_name: str) -> Optional[np.ndarray]:
        """
        Get hybrid embedding for a track by name.
        
        Args:
            track_name: Track name
            
        Returns:
            Hybrid embedding vector or None
        """
        # Try to get from collaborative embeddings first
        collab_emb = self.collaborative_embeddings.get_embedding_by_name(track_name)
        audio_emb = self.audio_embeddings.get_embedding_by_name(track_name)
        
        if collab_emb is None or audio_emb is None:
            return None
        
        # Normalize and fuse
        collab_emb_norm = normalize([collab_emb])[0]
        audio_emb_norm = normalize([audio_emb])[0]
        
        if self.fusion_method == 'concatenate':
            hybrid_emb = np.concatenate([collab_emb_norm, audio_emb_norm])
        elif self.fusion_method == 'weighted_sum':
            min_dim = min(len(collab_emb_norm), len(audio_emb_norm))
            collab_emb_norm = collab_emb_norm[:min_dim]
            audio_emb_norm = audio_emb_norm[:min_dim]
            hybrid_emb = (self.collaborative_weight * collab_emb_norm + 
                         self.audio_weight * audio_emb_norm)
        
        return normalize([hybrid_emb])[0]
    
    def get_all_embeddings(self) -> Dict[str, np.ndarray]:
        """
        Get all hybrid embeddings.
        
        Returns:
            Dictionary mapping track ID to hybrid embedding vector
        """
        return self.embeddings
    
    def get_embedding_dim(self) -> int:
        """
        Get the dimensionality of hybrid embeddings.
        
        Returns:
            Embedding dimensionality
        """
        if not self.embeddings:
            return 0
        
        first_embedding = next(iter(self.embeddings.values()))
        return len(first_embedding)
    
    def save(self, filepath: Path):
        """
        Save hybrid embeddings.
        
        Args:
            filepath: Path to save the embeddings
        """
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'wb') as f:
            pickle.dump({
                'embeddings': self.embeddings,
                'fusion_method': self.fusion_method,
                'collaborative_weight': self.collaborative_weight,
                'audio_weight': self.audio_weight
            }, f)
    
    def load(self, filepath: Path):
        """
        Load hybrid embeddings.
        
        Args:
            filepath: Path to the embeddings file
        """
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        self.embeddings = data['embeddings']
        self.fusion_method = data['fusion_method']
        self.collaborative_weight = data['collaborative_weight']
        self.audio_weight = data['audio_weight']

