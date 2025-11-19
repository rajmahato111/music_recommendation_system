"""
Ranking model for re-ranking candidate recommendations.
"""
import numpy as np
import pandas as pd
from typing import List, Dict, Optional
from pathlib import Path
import pickle

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

from ..features.feature_engineering import FeatureEngineer
from ..data_loader import DataLoader


class Ranker:
    """Ranking model for re-ranking recommendations."""
    
    def __init__(self, feature_engineer: FeatureEngineer, 
                 model_type: str = 'lightgbm'):
        """
        Initialize ranker.
        
        Args:
            feature_engineer: FeatureEngineer instance
            model_type: Type of ranking model ('lightgbm' or 'linear')
        """
        self.feature_engineer = feature_engineer
        self.data_loader = feature_engineer.data_loader
        self.model_type = model_type
        self.model = None
        self.feature_names = None
        
        if model_type == 'lightgbm' and not LIGHTGBM_AVAILABLE:
            raise ImportError("lightgbm not installed. Install with: pip install lightgbm")
    
    def train(self, training_data: List[Dict], validation_data: Optional[List[Dict]] = None,
              n_estimators: int = 100, learning_rate: float = 0.1, 
              max_depth: int = 7, random_seed: int = 42):
        """
        Train the ranking model.
        
        Args:
            training_data: List of training examples, each with:
                - 'seed_songs': List of seed song dicts
                - 'candidate_tracks': List of (track_id, similarity_score) tuples
                - 'labels': List of relevance labels (1 for relevant, 0 for not)
            validation_data: Optional validation data in same format
            n_estimators: Number of boosting rounds
            learning_rate: Learning rate
            max_depth: Maximum tree depth
            random_seed: Random seed
        """
        # Extract features and labels
        X_train, y_train = self._prepare_data(training_data)
        
        if self.model_type == 'lightgbm':
            train_data = lgb.Dataset(X_train, label=y_train)
            
            params = {
                'objective': 'binary',
                'metric': 'binary_logloss',
                'boosting_type': 'gbdt',
                'num_leaves': 31,
                'learning_rate': learning_rate,
                'feature_fraction': 0.9,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'verbose': -1,
                'seed': random_seed
            }
            
            valid_sets = [train_data]
            valid_names = ['train']
            
            if validation_data:
                X_val, y_val = self._prepare_data(validation_data)
                valid_data = lgb.Dataset(X_val, label=y_val)
                valid_sets.append(valid_data)
                valid_names.append('valid')
            
            self.model = lgb.train(
                params,
                train_data,
                num_boost_round=n_estimators,
                valid_sets=valid_sets,
                valid_names=valid_names,
                callbacks=[lgb.early_stopping(stopping_rounds=10, verbose=False)]
            )
            
            self.feature_names = self.model.feature_name()
        
        elif self.model_type == 'linear':
            from sklearn.linear_model import LogisticRegression
            self.model = LogisticRegression(random_state=random_seed, max_iter=1000)
            self.model.fit(X_train, y_train)
            self.feature_names = list(X_train.columns) if isinstance(X_train, pd.DataFrame) else None
    
    def _prepare_data(self, data: List[Dict]) -> tuple:
        """
        Prepare training data from examples.
        
        Args:
            data: List of training examples
            
        Returns:
            Tuple of (features DataFrame, labels array)
        """
        all_features = []
        all_labels = []
        
        for example in data:
            seed_songs = example['seed_songs']
            candidate_tracks = example['candidate_tracks']
            labels = example['labels']
            
            # Extract features
            features_df = self.feature_engineer.extract_batch_features(
                seed_songs, candidate_tracks
            )
            
            if features_df.empty:
                continue
            
            # Add labels
            features_df['label'] = labels[:len(features_df)]
            
            all_features.append(features_df)
        
        if not all_features:
            return pd.DataFrame(), np.array([])
        
        combined_df = pd.concat(all_features, ignore_index=True)
        
        # Separate features and labels
        label_col = 'label'
        if label_col in combined_df.columns:
            labels = combined_df[label_col].values
            features = combined_df.drop(columns=[label_col, 'track_id'], errors='ignore')
        else:
            labels = np.array([])
            features = combined_df.drop(columns=['track_id'], errors='ignore')
        
        return features, labels
    
    def predict_scores(self, seed_songs: List[Dict], 
                      candidate_tracks: List[tuple]) -> np.ndarray:
        """
        Predict ranking scores for candidate tracks.
        
        Args:
            seed_songs: List of seed song dicts
            candidate_tracks: List of (track_id, similarity_score) tuples
            
        Returns:
            Array of ranking scores
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Extract features
        features_df = self.feature_engineer.extract_batch_features(
            seed_songs, candidate_tracks
        )
        
        if features_df.empty:
            return np.array([])
        
        # Remove track_id column
        features = features_df.drop(columns=['track_id'], errors='ignore')
        
        # Ensure feature order matches training
        if self.feature_names and isinstance(features, pd.DataFrame):
            # Align features with training feature order
            missing_cols = set(self.feature_names) - set(features.columns)
            for col in missing_cols:
                features[col] = 0
            features = features[self.feature_names]
        
        # Predict
        if self.model_type == 'lightgbm':
            scores = self.model.predict(features.values)
        else:
            scores = self.model.predict_proba(features.values)[:, 1]
        
        return scores
    
    def rank(self, seed_songs: List[Dict], candidate_tracks: List[tuple]) -> List[tuple]:
        """
        Rank candidate tracks by predicted scores.
        
        Args:
            seed_songs: List of seed song dicts
            candidate_tracks: List of (track_id, similarity_score) tuples
            
        Returns:
            List of (track_id, score) tuples sorted by score (descending)
        """
        if self.model is None:
            # If no model, return original order with similarity scores
            return candidate_tracks
        
        scores = self.predict_scores(seed_songs, candidate_tracks)
        
        if len(scores) == 0:
            return []
        
        # Combine with track IDs
        ranked = list(zip([t[0] for t in candidate_tracks], scores))
        
        # Sort by score (descending)
        ranked.sort(key=lambda x: x[1], reverse=True)
        
        return ranked
    
    def save(self, filepath: Path):
        """
        Save the ranking model.
        
        Args:
            filepath: Path to save the model
        """
        if self.model is None:
            raise ValueError("Model not trained. Cannot save.")
        
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        if self.model_type == 'lightgbm':
            self.model.save_model(str(filepath))
        else:
            with open(filepath, 'wb') as f:
                pickle.dump(self.model, f)
        
        # Save metadata
        metadata_path = filepath.with_suffix('.metadata.pkl')
        with open(metadata_path, 'wb') as f:
            pickle.dump({
                'model_type': self.model_type,
                'feature_names': self.feature_names
            }, f)
    
    def load(self, filepath: Path):
        """
        Load the ranking model.
        
        Args:
            filepath: Path to the model file
        """
        # Load metadata
        metadata_path = filepath.with_suffix('.metadata.pkl')
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)
        
        self.model_type = metadata['model_type']
        self.feature_names = metadata['feature_names']
        
        # Load model
        if self.model_type == 'lightgbm':
            self.model = lgb.Booster(model_file=str(filepath))
        else:
            with open(filepath, 'rb') as f:
                self.model = pickle.load(f)

