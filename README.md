# Hybrid Music Recommendation System

A hybrid music recommendation system that combines collaborative filtering (Item2Vec) with audio-content embeddings to provide personalized song recommendations. The system uses approximate nearest neighbor (ANN) search for fast retrieval and includes a ranking model for improved relevance.

## Overview

This project implements a comprehensive music recommendation pipeline that addresses the limitations of pure content-based or collaborative filtering approaches. By combining both methods, the system can:

- Leverage playlist co-occurrence patterns (collaborative filtering)
- Utilize audio feature similarity (content-based)
- Handle cold-start scenarios better than pure collaborative methods
- Provide diverse and novel recommendations
- Scale to large datasets with efficient ANN indexing

## Architecture

The system consists of several key components:

1. **Data Preprocessing**: Handles data cleaning, playlist generation, and train/validation/test splitting
2. **Collaborative Embeddings**: Item2Vec model trained on playlist sequences
3. **Audio Embeddings**: Feature extraction from Spotify audio features
4. **Hybrid Embeddings**: Fusion of collaborative and audio embeddings
5. **ANN Indexing**: Fast similarity search using FAISS or HNSW
6. **Ranking Model**: LightGBM-based re-ranking for improved relevance
7. **Evaluation Framework**: Comprehensive metrics and ablation studies

## Project Structure

```
music_recommendation_system/
├── data/
│   ├── raw/
│   │   └── data.csv          # Raw music data
│   ├── playlists/            # Playlist data
│   └── processed/            # Processed data
├── src/
│   ├── data_loader.py        # Data loading utilities
│   ├── data_preprocessing.py # Data preprocessing
│   ├── models/
│   │   └── item2vec.py       # Item2Vec implementation
│   ├── embeddings/
│   │   ├── collaborative_embeddings.py
│   │   ├── audio_embeddings.py
│   │   └── hybrid_embeddings.py
│   ├── retrieval/
│   │   └── ann_index.py      # ANN indexing (FAISS/HNSW)
│   ├── ranking/
│   │   └── ranker.py         # Ranking model
│   ├── features/
│   │   └── feature_engineering.py
│   ├── evaluation/
│   │   ├── metrics.py        # Evaluation metrics
│   │   ├── evaluator.py      # Evaluation pipeline
│   │   └── ablation_studies.py
│   ├── recommender/
│   │   ├── baseline_recommender.py
│   │   └── hybrid_recommender.py
│   └── explainability.py     # Recommendation explanations
├── notebooks/                # Jupyter notebooks for analysis
├── tests/                    # Unit tests
├── results/                  # Evaluation results
├── docs/                     # Documentation
├── app.py                    # Streamlit application
├── requirements.txt          # Python dependencies
├── Dockerfile               # Docker configuration
└── docker-compose.yml       # Docker Compose configuration
```

## Installation

### Prerequisites

- Python 3.9+
- pip

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd music_recommendation_system
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Ensure data file is in `data/raw/data.csv`

## Usage

### Running the Streamlit App

```bash
streamlit run app.py
```

The app will be available at `http://localhost:8501`

### Training Models

The system supports training different components:

1. **Generate Playlists** (if not using external playlist data):
```python
from src.data_loader import DataLoader
from src.data_preprocessing import DataPreprocessor

data_loader = DataLoader()
preprocessor = DataPreprocessor(data_loader)
playlists = preprocessor.create_synthetic_playlists(n_playlists=10000)
```

2. **Train Item2Vec**:
```python
from src.embeddings.collaborative_embeddings import CollaborativeEmbeddings

collab_emb = CollaborativeEmbeddings(data_loader)
collab_emb.train(playlists, epochs=10, vector_size=128)
```

3. **Generate Audio Embeddings**:
```python
from src.embeddings.audio_embeddings import AudioEmbeddings

audio_emb = AudioEmbeddings(data_loader)
audio_emb.generate_embeddings()
```

4. **Create Hybrid Embeddings**:
```python
from src.embeddings.hybrid_embeddings import HybridEmbeddings

hybrid_emb = HybridEmbeddings(collab_emb, audio_emb, fusion_method='concatenate')
```

5. **Build ANN Index**:
```python
from src.retrieval.ann_index import ANNIndex

ann_index = ANNIndex(method='faiss', dimension=hybrid_emb.get_embedding_dim())
ann_index.build_index(hybrid_emb.get_all_embeddings())
```

6. **Train Ranking Model** (optional):
```python
from src.ranking.ranker import Ranker
from src.features.feature_engineering import FeatureEngineer

feature_engineer = FeatureEngineer(data_loader)
ranker = Ranker(feature_engineer, model_type='lightgbm')
ranker.train(training_data, validation_data)
```

## Evaluation

The system includes comprehensive evaluation metrics:

- **Recall@K**: Fraction of relevant items retrieved in top K
- **Precision@K**: Fraction of retrieved items that are relevant
- **NDCG@K**: Normalized Discounted Cumulative Gain
- **MRR**: Mean Reciprocal Rank
- **Diversity**: Measure of recommendation diversity
- **Novelty**: Measure of recommendation novelty

Run evaluation:
```python
from src.evaluation.ablation_studies import AblationStudy

study = AblationStudy(data_loader)
study.evaluate_baseline_content(baseline_recommender, test_playlists)
study.evaluate_hybrid(hybrid_recommender, test_playlists)
results = study.compare_models()
```

## Docker Deployment

### Build and Run with Docker

```bash
docker-compose up --build
```

The application will be available at `http://localhost:8501`

### Manual Docker Build

```bash
docker build -t music-recommender .
docker run -p 8501:8501 music-recommender
```

## Technologies Used

- **Python 3.9+**: Core programming language
- **Pandas**: Data manipulation
- **NumPy**: Numerical computations
- **Scikit-learn**: Machine learning utilities
- **Gensim**: Word2Vec/Item2Vec implementation
- **FAISS**: Fast similarity search
- **LightGBM**: Gradient boosting for ranking
- **Streamlit**: Web application framework
- **Plotly**: Interactive visualizations

## Features

- **Hybrid Recommendation**: Combines collaborative and content-based approaches
- **Fast Retrieval**: ANN indexing for scalable similarity search
- **Re-ranking**: LightGBM-based ranking model for improved relevance
- **Explainability**: Provides explanations for recommendations
- **Comprehensive Evaluation**: Multiple metrics and ablation studies
- **Docker Support**: Easy deployment with containerization

## Evaluation Results

The system supports ablation studies comparing:
- Content-based baseline
- Collaborative-only (Item2Vec)
- Hybrid (collaborative + audio)
- Hybrid with ranking

Results are saved in the `results/` directory.

## Contributing

Feel free to contribute by:
1. Opening issues for bugs or feature requests
2. Submitting pull requests
3. Improving documentation

## Acknowledgments

- Spotify for audio feature data
- Million Playlist Dataset (if used)
- Open source libraries and frameworks
