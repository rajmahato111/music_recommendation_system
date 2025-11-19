# Jupyter Notebooks for Music Recommendation System

This directory contains Jupyter notebooks for analyzing and training the hybrid music recommendation system.

## Notebooks Overview

### 1. `01_data_exploration.ipynb`
**Purpose**: Explore and analyze the music dataset

**Contents**:
- Data loading and basic statistics
- Audio features distribution analysis
- Popularity and temporal analysis
- Feature correlations
- Artist analysis
- Data quality checks

**Dependencies**: Core dependencies only (pandas, numpy, matplotlib, seaborn)

---

### 2. `02_item2vec_training.ipynb`
**Purpose**: Train Item2Vec model for collaborative filtering

**Contents**:
- Create synthetic playlists from data
- Train Item2Vec model
- Analyze embeddings
- Find similar tracks

**Dependencies**: Requires `gensim` (install with `pip install gensim`)

**Note**: Run notebook 01 first to understand the data

---

### 3. `03_audio_embeddings_analysis.ipynb`
**Purpose**: Generate and analyze audio embeddings

**Contents**:
- Generate audio embeddings from Spotify features
- PCA analysis and visualization
- Embedding statistics
- Compare full vs PCA-reduced embeddings

**Dependencies**: Core dependencies only

**Note**: Run notebook 01 first to understand the data

---

### 4. `04_hybrid_system_evaluation.ipynb`
**Purpose**: Evaluate hybrid system and compare models

**Contents**:
- Load trained models (from notebooks 02 and 03)
- Create hybrid embeddings
- Build ANN index (FAISS or brute force fallback)
- Evaluate baseline vs hybrid models
- Visualize results

**Dependencies**: Requires `gensim` and optionally `faiss-cpu`

**Note**: Run notebooks 01, 02, and 03 first

---

## Installation

### Core Dependencies
```bash
pip install pandas numpy matplotlib seaborn scikit-learn scipy plotly jupyter notebook
```

### Optional Dependencies
```bash
# For Item2Vec (notebook 02)
pip install gensim

# For ANN indexing (notebook 04, faster)
pip install faiss-cpu

# Install all at once
pip install -r ../requirements.txt
```

## Usage

1. **Start Jupyter Notebook**:
   ```bash
   cd notebooks
   jupyter notebook
   ```

2. **Run notebooks in order**:
   - Start with `01_data_exploration.ipynb`
   - Then `02_item2vec_training.ipynb` (requires gensim)
   - Then `03_audio_embeddings_analysis.ipynb`
   - Finally `04_hybrid_system_evaluation.ipynb` (requires gensim, optionally faiss-cpu)

3. **Run cells sequentially**: Use `Shift+Enter` to run each cell

## Troubleshooting

### Missing Dependencies
- If you get `ModuleNotFoundError`, install the missing package
- See `INSTALL_DEPENDENCIES.md` for detailed instructions

### Gensim Warnings
- If you see gensim warnings during training, they're harmless
- The notebook uses `workers=1` to minimize warnings

### FAISS Installation Issues
- On Apple Silicon (M1/M2), try: `pip install faiss-cpu --no-cache-dir`
- If FAISS fails, the notebook will automatically use brute force search

## Output Files

The notebooks generate the following files in `../data/processed/`:
- `item2vec_model/` - Trained Item2Vec model (from notebook 02)
- `audio_embeddings/` - Audio embeddings (from notebook 03)

## Notes

- Notebooks are designed to be run sequentially
- Each notebook can be run independently, but results from previous notebooks are needed for later ones
- The notebooks include error handling for missing dependencies
- All notebooks include visualizations and analysis

