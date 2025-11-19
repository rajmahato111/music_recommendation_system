# Hybrid Music Recommendation System - Project Summary

## 🎯 Project Overview

A production-ready hybrid music recommendation system combining collaborative filtering (Item2Vec) with audio-content embeddings for personalized song recommendations.

**Key Features:**
- ✅ Hybrid approach (collaborative + content-based)
- ✅ Scalable ANN indexing (FAISS/HNSW)
- ✅ LightGBM-based re-ranking
- ✅ Comprehensive evaluation framework
- ✅ Explainable recommendations
- ✅ Streamlit web interface
- ✅ Docker deployment ready

---

## 🏗️ Architecture

```
Data Sources → Processing → Embeddings → Retrieval → Ranking → Recommendations
     ↓            ↓            ↓            ↓          ↓            ↓
   CSV/JSON   Cleaning    Item2Vec +    ANN Index  LightGBM    Top-K Results
   Playlists  Playlists   Audio Feat.   (FAISS)    Ranker      + Explanations
```

**Components:**
1. **Data Layer**: CSV loading, preprocessing, playlist generation
2. **Embedding Layer**: Item2Vec (128D) + Audio features (128D+) → Hybrid (256D+)
3. **Retrieval Layer**: ANN search with FAISS or HNSW
4. **Ranking Layer**: LightGBM for re-ranking candidates
5. **Application Layer**: Streamlit UI with visualizations

---

## 📊 Technology Stack

| Category | Technologies |
|----------|-------------|
| **Core** | Python 3.9+, Pandas, NumPy, Scikit-learn |
| **ML/Embeddings** | Gensim (Item2Vec), FAISS, HNSWlib, LightGBM |
| **Web/UI** | Streamlit, Plotly |
| **Deployment** | Docker, Docker Compose |

---

## 📈 Model Performance (Placeholder)

| Model | Recall@10 | Precision@10 | NDCG@10 | MRR |
|-------|-----------|--------------|---------|-----|
| Baseline (Content) | 0.35 | 0.28 | 0.40 | 0.35 |
| Item2Vec Only | 0.45 | 0.35 | 0.52 | 0.48 |
| Hybrid | 0.52 | 0.42 | 0.62 | 0.58 |
| Hybrid + Ranking | 0.58 | 0.48 | 0.68 | 0.65 |

*Note: Metrics are placeholders. Update after training models.*

---

## 🔄 Data Flow

1. **Input**: Raw CSV with Spotify audio features
2. **Processing**: Clean, normalize, generate playlists
3. **Training**: 
   - Train Item2Vec on playlist sequences
   - Generate audio embeddings from features
   - Create hybrid embeddings (concatenation/weighted sum)
4. **Indexing**: Build ANN index for fast retrieval
5. **Ranking**: Train LightGBM on candidate features
6. **Output**: Top-K recommendations with explanations

---

## 📁 Project Structure

```
music_recommendation_system/
├── data/              # Raw data, playlists, processed models
├── src/               # Source code
│   ├── models/        # Item2Vec implementation
│   ├── embeddings/    # Collaborative, audio, hybrid
│   ├── retrieval/     # ANN indexing
│   ├── ranking/       # LightGBM ranker
│   ├── recommender/   # Recommendation logic
│   └── evaluation/    # Metrics and evaluation
├── notebooks/         # Analysis notebooks
├── docs/              # Documentation and visualizations
├── app.py             # Streamlit application
└── Dockerfile         # Container configuration
```

---

## 🚀 Quick Start

### Installation
```bash
pip install -r requirements.txt
```

### Run Application
```bash
streamlit run app.py
```

### Docker Deployment
```bash
docker-compose up --build
```

### Training Models
```python
# 1. Generate playlists
playlists = preprocessor.create_synthetic_playlists(n_playlists=10000)

# 2. Train Item2Vec
collab_emb.train(playlists, epochs=10, vector_size=128)

# 3. Generate audio embeddings
audio_emb.generate_embeddings()

# 4. Create hybrid embeddings
hybrid_emb = HybridEmbeddings(collab_emb, audio_emb)

# 5. Build ANN index
ann_index.build_index(hybrid_emb.get_all_embeddings())
```

---

## 📊 Evaluation Metrics

- **Recall@K**: Fraction of relevant items retrieved
- **Precision@K**: Fraction of retrieved items that are relevant
- **NDCG@K**: Ranking quality with position discounting
- **MRR**: Mean reciprocal rank of first relevant item
- **Diversity**: Measure of recommendation variety
- **Novelty**: Measure of recommendation unexpectedness

---

## ✅ Status

### Operational
- ✅ Data loading and preprocessing
- ✅ Baseline content-based recommender
- ✅ Evaluation framework
- ✅ Streamlit interface
- ✅ Docker deployment

### Requires Training
- ⚠️ Item2Vec model
- ⚠️ Audio embeddings
- ⚠️ Hybrid embeddings
- ⚠️ ANN index
- ⚠️ Ranking model

---

## 📚 Documentation

- **Full Analysis**: `docs/PROJECT_ANALYSIS.md`
- **Visualizations**: `docs/visualizations/` (HTML files)
- **Main README**: `README.md`

---

## 🎯 Use Cases

1. **Music Discovery**: Find new songs similar to preferences
2. **Playlist Generation**: Create playlists from seed songs
3. **Cold Start**: Recommend for new songs using audio features
4. **Personalization**: Leverage collaborative patterns
5. **Research**: Model comparison and ablation studies

---

*Last Updated: [Current Date]*  
*Version: 1.0*

