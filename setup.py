"""
Setup script for the music recommendation system.
"""
from setuptools import setup, find_packages

setup(
    name="music-recommendation-system",
    version="1.0.0",
    description="Hybrid Music Recommendation System using Item2Vec and Audio Embeddings",
    packages=find_packages(),
    install_requires=[
        "streamlit>=1.25.0",
        "pandas>=1.5.3",
        "numpy>=1.23.5",
        "scikit-learn>=1.2.2",
        "scipy>=1.10.1",
        "plotly>=5.14.1",
        "gensim>=4.3.0",
        "faiss-cpu>=1.7.4",
        "lightgbm>=4.0.0",
        "hnswlib>=0.7.0",
    ],
    python_requires=">=3.9",
)

