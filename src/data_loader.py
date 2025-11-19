"""
Data loading utilities for the music recommendation system.
"""
import pandas as pd
import os
from pathlib import Path


class DataLoader:
    """Handles loading and basic operations on music data."""
    
    # Numerical columns for similarity calculations
    NUMERICAL_COLS = [
        'valence', 'year', 'acousticness', 'danceability', 'duration_ms', 
        'energy', 'explicit', 'instrumentalness', 'key', 'liveness', 
        'loudness', 'mode', 'popularity', 'speechiness', 'tempo'
    ]
    
    # Metadata columns for display
    METADATA_COLS = ['name', 'artists', 'year']
    
    def __init__(self, data_path=None):
        """
        Initialize the data loader.
        
        Args:
            data_path: Path to the CSV file. If None, uses default location.
        """
        if data_path is None:
            # Default to data/raw/data.csv
            # Try relative to current working directory first
            cwd_path = Path.cwd() / "data" / "raw" / "data.csv"
            if cwd_path.exists():
                data_path = cwd_path
            else:
                # Fall back to relative to this file
                project_root = Path(__file__).parent.parent
                data_path = project_root / "data" / "raw" / "data.csv"
        
        self.data_path = data_path
        self.data = None
        self._load_data()
    
    def _load_data(self):
        """Load the music data from CSV."""
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
        
        self.data = pd.read_csv(self.data_path)
        
        # Convert release_date to datetime if it exists
        if 'release_date' in self.data.columns:
            self.data['release_date'] = pd.to_datetime(self.data['release_date'], errors='coerce')
    
    def get_song_data(self, name):
        """
        Retrieve song data for a given song name.
        
        Args:
            name: Song name (case-insensitive)
            
        Returns:
            pandas Series with song data, or None if not found
        """
        try:
            return self.data[self.data['name'].str.lower() == name.lower()].iloc[0]
        except (IndexError, KeyError):
            return None
    
    def get_song_by_id(self, song_id):
        """
        Retrieve song data by track ID.
        
        Args:
            song_id: Track ID
            
        Returns:
            pandas Series with song data, or None if not found
        """
        try:
            return self.data[self.data['id'] == song_id].iloc[0]
        except (IndexError, KeyError):
            return None
    
    def get_numerical_features(self, song_data):
        """
        Extract numerical features from song data.
        
        Args:
            song_data: pandas Series with song data
            
        Returns:
            numpy array of numerical features
        """
        return song_data[self.NUMERICAL_COLS].values
    
    def get_all_numerical_features(self):
        """
        Get all numerical features from the dataset.
        
        Returns:
            pandas DataFrame with numerical columns
        """
        return self.data[self.NUMERICAL_COLS]
    
    def get_metadata(self, song_data):
        """
        Extract metadata from song data.
        
        Args:
            song_data: pandas Series with song data
            
        Returns:
            dict with metadata
        """
        return song_data[self.METADATA_COLS].to_dict()
    
    def get_data(self):
        """Get the full dataset."""
        return self.data
    
    def get_song_count(self):
        """Get the total number of songs in the dataset."""
        return len(self.data)

