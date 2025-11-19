# Hybrid Music Recommendation System - Project Analysis

## 📊 Project Overview

A comprehensive hybrid music recommendation system that combines collaborative filtering (Item2Vec) with audio-content embeddings to provide personalized song recommendations. The system uses approximate nearest neighbor (ANN) search for fast retrieval and includes a ranking model for improved relevance.

**Project Type:** Machine Learning / Recommendation System  
**Domain:** Music Technology  
**Status:** Production Ready (Baseline), Hybrid System (Requires Training)

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         DATA SOURCES                                     │
├─────────────────────────────────────────────────────────────────────────┤
│  • CSV Data (Spotify Features)                                          │
│  • Playlist Data (Synthetic/Real)                                       │
│  • Audio Features (15 numerical features)                                │
└────────────────────┬────────────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    DATA PROCESSING LAYER                                 │
├─────────────────────────────────────────────────────────────────────────┤
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐     │
│  │ Data Loader      │  │ Preprocessing    │  │ Feature          │     │
│  │ • CSV Loading    │→ │ • Cleaning       │→ │ Engineering      │     │
│  │ • Data Access    │  │ • Playlist Gen   │  │ • Feature Extract│     │
│  └──────────────────┘  └──────────────────┘  └──────────────────┘     │
└────────────────────┬────────────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      EMBEDDING LAYER                                     │
├─────────────────────────────────────────────────────────────────────────┤
│  ┌──────────────────────┐         ┌──────────────────────┐             │
│  │ Collaborative        │         │ Audio Embeddings     │             │
│  │ Embeddings           │         │                      │             │
│  │ • Item2Vec (Gensim)  │         │ • Spotify Features   │             │
│  │ • Vector Size: 128   │         │ • PCA (Optional)     │             │
│  │ • Window: 5          │         │ • Normalization      │             │
│  └──────────┬───────────┘         └──────────┬───────────┘             │
│             │                                │                          │
│             └────────────┬───────────────────┘                          │
│                          ▼                                               │
│              ┌──────────────────────┐                                   │
│              │ Hybrid Embeddings    │                                   │
│              │ • Concatenation      │                                   │
│              │ • Weighted Sum       │                                   │
│              │ • Dimension: 256+    │                                   │
│              └──────────────────────┘                                   │
└────────────────────┬────────────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    RETRIEVAL & RANKING LAYER                             │
├─────────────────────────────────────────────────────────────────────────┤
│  ┌──────────────────┐         ┌──────────────────┐                     │
│  │ ANN Index        │         │ Ranking Model    │                     │
│  │ • FAISS          │────────→│ • LightGBM       │                     │
│  │ • HNSW           │         │ • Feature-based  │                     │
│  │ • Fast Search    │         │ • Re-ranking     │                     │
│  └──────────────────┘         └──────────────────┘                     │
└────────────────────┬────────────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      RECOMMENDATION LAYER                                │
├─────────────────────────────────────────────────────────────────────────┤
│  ┌──────────────────┐         ┌──────────────────┐                     │
│  │ Baseline         │         │ Hybrid           │                     │
│  │ Recommender      │         │ Recommender      │                     │
│  │ (Content-based)  │         │ (Full System)    │                     │
│  └──────────────────┘         └──────────────────┘                     │
└────────────────────┬────────────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      APPLICATION LAYER                                   │
├─────────────────────────────────────────────────────────────────────────┤
│  ┌──────────────────┐         ┌──────────────────┐                     │
│  │ Streamlit App    │         │ Evaluation       │                     │
│  │ • Web Interface  │         │ Framework        │                     │
│  │ • Visualizations │         │ • Metrics        │                     │
│  │ • Explanations   │         │ • Ablation       │                     │
│  └──────────────────┘         └──────────────────┘                     │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 🔄 Data Flow Diagram

```
┌─────────────┐
│  Raw Data   │
│  (CSV)      │
└──────┬──────┘
       │
       ▼
┌──────────────────────────────────────────────────────────────┐
│                    ETL Pipeline                               │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │ Extract      │→ │ Transform    │→ │ Load         │      │
│  │ • CSV Read   │  │ • Clean      │  │ • Processed  │      │
│  │ • Validate   │  │ • Normalize  │  │ • Playlists  │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└──────┬───────────────────────────────────────────────────────┘
       │
       ├──────────────────────┬──────────────────────┐
       ▼                      ▼                      ▼
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│ Collaborative│    │ Audio        │    │ Feature      │
│ Training     │    │ Embeddings   │    │ Engineering  │
│ • Item2Vec   │    │ • PCA        │    │ • Stats      │
│ • Playlists  │    │ • Normalize  │    │ • Similarity │
└──────┬───────┘    └──────┬───────┘    └──────┬───────┘
       │                   │                    │
       └───────────┬───────┴────────────────────┘
                   ▼
          ┌─────────────────┐
          │ Hybrid Fusion   │
          │ • Concatenate   │
          │ • Weighted Sum  │
          └────────┬────────┘
                   │
                   ▼
          ┌─────────────────┐
          │ ANN Index Build │
          │ • FAISS/HNSW    │
          └────────┬────────┘
                   │
                   ▼
          ┌─────────────────┐
          │ Ranking Model   │
          │ • LightGBM      │
          └────────┬────────┘
                   │
                   ▼
          ┌─────────────────┐
          │ Recommendations │
          │ • Top-K Results │
          └─────────────────┘
```

---

## 🛠️ Technology Stack

### Core Technologies
- **Python 3.9+**: Primary programming language
- **Pandas 2.0+**: Data manipulation and analysis
- **NumPy 1.26+**: Numerical computations
- **Scikit-learn 1.3+**: Machine learning utilities, PCA, normalization

### Machine Learning & Embeddings
- **Gensim 4.3+**: Item2Vec implementation (Word2Vec-based)
- **FAISS**: Fast similarity search (Facebook AI Similarity Search)
- **HNSWlib**: Hierarchical Navigable Small World graphs for ANN
- **LightGBM 4.0+**: Gradient boosting for ranking model

### Web Framework & Visualization
- **Streamlit 1.28+**: Web application framework
- **Plotly 5.17+**: Interactive visualizations

### Deployment
- **Docker**: Containerization
- **Docker Compose**: Multi-container orchestration

### Storage
- **File System**: CSV files, Pickle for models/embeddings
- **In-Memory**: Pandas DataFrames, NumPy arrays

---

## 📈 Model Performance Metrics

### Baseline Model (Content-Based)
| Metric | Value | Status |
|--------|-------|--------|
| **Recall@5** | 0.XXX | ⚠️ Requires Training |
| **Recall@10** | 0.XXX | ⚠️ Requires Training |
| **Precision@5** | 0.XXX | ⚠️ Requires Training |
| **Precision@10** | 0.XXX | ⚠️ Requires Training |
| **NDCG@10** | 0.XXX | ⚠️ Requires Training |
| **MRR** | 0.XXX | ⚠️ Requires Training |
| **Diversity** | 0.XXX | ⚠️ Requires Training |
| **Novelty** | 0.XXX | ⚠️ Requires Training |

### Hybrid Model (Item2Vec + Audio)
| Metric | Value | Status |
|--------|-------|--------|
| **Recall@5** | 0.XXX | ⚠️ Requires Training |
| **Recall@10** | 0.XXX | ⚠️ Requires Training |
| **Precision@5** | 0.XXX | ⚠️ Requires Training |
| **Precision@10** | 0.XXX | ⚠️ Requires Training |
| **NDCG@10** | 0.XXX | ⚠️ Requires Training |
| **MRR** | 0.XXX | ⚠️ Requires Training |
| **Diversity** | 0.XXX | ⚠️ Requires Training |
| **Novelty** | 0.XXX | ⚠️ Requires Training |

### Hybrid Model with Ranking
| Metric | Value | Status |
|--------|-------|--------|
| **Recall@5** | 0.XXX | ⚠️ Requires Training |
| **Recall@10** | 0.XXX | ⚠️ Requires Training |
| **Precision@5** | 0.XXX | ⚠️ Requires Training |
| **Precision@10** | 0.XXX | ⚠️ Requires Training |
| **NDCG@10** | 0.XXX | ⚠️ Requires Training |
| **MRR** | 0.XXX | ⚠️ Requires Training |
| **Diversity** | 0.XXX | ⚠️ Requires Training |
| **Novelty** | 0.XXX | ⚠️ Requires Training |

### Model Comparison (Ablation Study)
| Model Variant | Recall@10 | NDCG@10 | Diversity | Novelty |
|---------------|-----------|---------|-----------|---------|
| Content-Based Baseline | 0.XXX | 0.XXX | 0.XXX | 0.XXX |
| Collaborative Only (Item2Vec) | 0.XXX | 0.XXX | 0.XXX | 0.XXX |
| Hybrid (Item2Vec + Audio) | 0.XXX | 0.XXX | 0.XXX | 0.XXX |
| Hybrid + Ranking | 0.XXX | 0.XXX | 0.XXX | 0.XXX |

---

## 🧩 Component Details

### 1. Data Processing
- **DataLoader**: Handles CSV loading, data access, song lookups
- **DataPreprocessor**: Data cleaning, playlist generation, train/test splits
- **Feature Engineering**: Feature extraction, similarity calculations

### 2. Embedding Models

#### Collaborative Embeddings (Item2Vec)
- **Model**: Gensim Word2Vec (Skip-gram)
- **Vector Size**: 128 dimensions
- **Window Size**: 5
- **Min Count**: 2
- **Training**: Playlist sequences as sentences

#### Audio Embeddings
- **Source**: Spotify audio features (15 features)
- **Processing**: Normalization, optional PCA
- **Features**: valence, energy, danceability, acousticness, etc.

#### Hybrid Embeddings
- **Fusion Methods**: 
  - Concatenation (default)
  - Weighted Sum
- **Dimension**: 256+ (128 collaborative + 128+ audio)

### 3. Retrieval System
- **ANN Index**: FAISS or HNSW
- **Distance Metrics**: Cosine similarity, L2 distance
- **Search Speed**: Sub-linear time complexity
- **Scalability**: Handles millions of tracks

### 4. Ranking Model
- **Algorithm**: LightGBM (Gradient Boosting)
- **Objective**: Binary classification (relevant/not relevant)
- **Features**: 
  - Similarity scores
  - Audio feature differences
  - Popularity metrics
  - Temporal features

### 5. Evaluation Framework
- **Metrics Implemented**:
  - Recall@K
  - Precision@K
  - NDCG@K
  - MRR (Mean Reciprocal Rank)
  - Diversity
  - Novelty
- **Ablation Studies**: Compare model variants
- **Cross-Validation**: Support for train/validation/test splits

---

## 📊 Evaluation Metrics Explained

### Recall@K
Fraction of relevant items retrieved in top K recommendations.
```
Recall@K = |Relevant ∩ Retrieved@K| / |Relevant|
```

### Precision@K
Fraction of retrieved items that are relevant.
```
Precision@K = |Relevant ∩ Retrieved@K| / K
```

### NDCG@K (Normalized Discounted Cumulative Gain)
Measures ranking quality with position-based discounting.
```
NDCG@K = DCG@K / IDCG@K
```

### MRR (Mean Reciprocal Rank)
Average reciprocal rank of first relevant item.
```
MRR = (1/n) * Σ(1/rank_i)
```

### Diversity
Measure of how different recommended items are from each other.
```
Diversity = Average pairwise distance between recommendations
```

### Novelty
Measure of how unexpected/popular recommendations are.
```
Novelty = -log2(popularity) for each item
```

---

## 🚀 Deployment Architecture

### Docker Deployment
```
┌─────────────────────────────────────────┐
│         Docker Container                │
│  ┌───────────────────────────────────┐  │
│  │     Streamlit Application         │  │
│  │  • Port: 8501                     │  │
│  │  • Health Check Enabled           │  │
│  └───────────────────────────────────┘  │
│  ┌───────────────────────────────────┐  │
│  │     Python Environment            │  │
│  │  • Python 3.9                     │  │
│  │  • All Dependencies               │  │
│  └───────────────────────────────────┘  │
│  ┌───────────────────────────────────┐  │
│  │     Data & Models                 │  │
│  │  • /app/data (mounted)            │  │
│  │  • /app/results (mounted)         │  │
│  └───────────────────────────────────┘  │
└─────────────────────────────────────────┘
```

### File Structure
```
music_recommendation_system/
├── data/
│   ├── raw/              # Raw CSV data
│   ├── playlists/        # Playlist data
│   └── processed/        # Processed data, models, embeddings
├── src/                  # Source code
│   ├── data_loader.py
│   ├── data_preprocessing.py
│   ├── models/           # Item2Vec model
│   ├── embeddings/       # Embedding generators
│   ├── retrieval/        # ANN indexing
│   ├── ranking/          # Ranking model
│   ├── recommender/      # Recommendation logic
│   ├── evaluation/       # Metrics and evaluation
│   └── features/         # Feature engineering
├── notebooks/            # Jupyter notebooks for analysis
├── tests/                # Unit tests
├── results/              # Evaluation results
├── docs/                 # Documentation
├── app.py                # Streamlit application
├── requirements.txt      # Dependencies
├── Dockerfile            # Docker configuration
└── docker-compose.yml    # Docker Compose configuration
```

---

## 🔍 System Status

### Operational Components
- ✅ Data Loader
- ✅ Baseline Recommender (Content-based)
- ✅ Evaluation Framework
- ✅ Streamlit Interface
- ✅ Docker Deployment

### Components Requiring Training
- ⚠️ Item2Vec Model (Implemented, needs training)
- ⚠️ Audio Embeddings (Can be generated on demand)
- ⚠️ Hybrid Embeddings (Requires both above)
- ⚠️ ANN Index (Requires hybrid embeddings)
- ⚠️ Ranking Model (Requires training data)

---

## 📝 Key Features

1. **Hybrid Approach**: Combines collaborative and content-based filtering
2. **Scalable Retrieval**: ANN indexing for fast similarity search
3. **Re-ranking**: LightGBM-based ranking for improved relevance
4. **Comprehensive Evaluation**: Multiple metrics and ablation studies
5. **Explainability**: Provides explanations for recommendations
6. **User-Friendly Interface**: Streamlit web application
7. **Docker Support**: Easy deployment with containerization
8. **Modular Design**: Clean separation of concerns

---

## 🎯 Use Cases

1. **Music Discovery**: Find new songs similar to user preferences
2. **Playlist Generation**: Create playlists based on seed songs
3. **Cold Start Handling**: Recommend for new songs using audio features
4. **Personalization**: Leverage collaborative patterns from playlists
5. **Research**: Ablation studies and model comparison

---

## 📚 Next Steps

1. **Train Models**: 
   - Generate playlists from data
   - Train Item2Vec model
   - Generate audio embeddings
   - Create hybrid embeddings

2. **Build Index**:
   - Build ANN index with hybrid embeddings
   - Optimize search parameters

3. **Train Ranking Model**:
   - Collect training data with labels
   - Train LightGBM ranker
   - Evaluate performance

4. **Evaluation**:
   - Run comprehensive evaluation
   - Perform ablation studies
   - Compare model variants

5. **Production**:
   - Deploy with Docker
   - Monitor performance
   - Collect user feedback

---

## 📞 Technical Specifications

### Model Parameters
- **Item2Vec**: 
  - Vector Size: 128
  - Window: 5
  - Min Count: 2
  - Epochs: 10
  - Algorithm: Skip-gram

- **Audio Embeddings**:
  - Features: 15 (Spotify audio features)
  - Optional PCA: Yes
  - Normalization: L2

- **Hybrid Embeddings**:
  - Fusion: Concatenation or Weighted Sum
  - Dimension: 256+ (128 + 128+)

- **ANN Index**:
  - Method: FAISS or HNSW
  - Metric: Cosine similarity
  - Search K: Configurable

- **Ranking Model**:
  - Algorithm: LightGBM
  - Objective: Binary classification
  - Boosting: GBDT

---

## 📊 Data Statistics

- **Total Songs**: Variable (depends on dataset)
- **Features**: 15 numerical audio features
- **Metadata**: Name, Artists, Year, Popularity
- **Playlists**: Synthetic or real playlist data
- **Embedding Dimensions**: 128 (collaborative) + 128+ (audio) = 256+

---

*Last Updated: [Current Date]*  
*Version: 1.0*  
*Status: Production Ready (Baseline), Hybrid System (Requires Training)*

