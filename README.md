# Spotify Hybrid Recommender System

A production-grade hybrid music recommendation engine that combines **NMF-based collaborative filtering** and **content-based filtering** into a single blended score. Served via a FastAPI microservice with FAISS vector similarity search, containerised with Docker Compose, and deployed on AWS EC2 via CodeDeploy.

---

## How It Works

The system exposes three filtering strategies:

| Method | Signal | Available for |
|---|---|---|
| **Content-Based** | Audio features (tempo, energy, key, tags) | All 50,683 songs |
| **Collaborative** | NMF latent factors from user play history | 30,459 songs with listening data |
| **Hybrid** | Weighted blend of both signals | 30,459 songs with listening data |

### Hybrid Scoring

```
combined[i] = w * minmax(content_scores[i]) + (1 - w) * minmax(collab_scores[i])
```

`w` is controlled by the **diversity slider** in the UI (0 = pure content, 1 = pure collaborative). The system auto-detects whether a song has listening history and falls back gracefully to content-based for cold-start songs.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     User Browser                        │
└─────────────────────┬───────────────────────────────────┘
                      │ HTTP
┌─────────────────────▼───────────────────────────────────┐
│              Streamlit Frontend  (port 8501)             │
│  - Song + artist input                                   │
│  - Filter selector (Content / Collaborative / Hybrid)    │
│  - Diversity slider                                      │
└─────────────────────┬───────────────────────────────────┘
                      │ HTTP (httpx)
┌─────────────────────▼───────────────────────────────────┐
│               FastAPI Backend  (port 8000)               │
│  POST /api/v1/recommend                                  │
│  GET  /api/v1/songs/search                               │
│  GET  /health                                            │
│                                                          │
│  ┌──────────────┐  ┌──────────────┐  ┌───────────────┐  │
│  │   Content    │  │Collaborative │  │    Hybrid     │  │
│  │ Recommender  │  │ Recommender  │  │  Recommender  │  │
│  │              │  │              │  │               │  │
│  │ FAISS        │  │ NMF (n=50)   │  │ Blends both   │  │
│  │ IndexFlatIP  │  │ FAISS        │  │ via numpy dot │  │
│  │ 50,683 songs │  │ 30,459 songs │  │ product       │  │
│  └──────────────┘  └──────────────┘  └───────────────┘  │
└─────────────────────────────────────────────────────────┘
```

Both services run as separate Docker containers orchestrated by Docker Compose.

---

## Offline Evaluation

Leave-one-out evaluation over 500 sampled users:

| Method | Precision@10 | Recall@10 | NDCG@10 |
|---|---|---|---|
| Content-Based | 0.0142 | 0.0144 | 0.0201 |
| Collaborative | 0.0336 | 0.0261 | 0.0444 |
| **Hybrid** | **0.0408** | **0.0327** | **0.0547** |

Hybrid outperforms content-based by **171%** and collaborative by **23%** on NDCG@10.

---

## Tech Stack

**ML / Data**
- `scikit-learn` — NMF matrix factorisation, TF-IDF, ColumnTransformer
- `faiss-cpu` — FAISS IndexFlatIP for sub-millisecond similarity search
- `category-encoders` — frequency encoding for high-cardinality artist column
- `dask` — out-of-core reading of the 574 MB user listening history CSV
- `scipy` — CSR sparse matrix for the (30,459 × 962,037) interaction matrix

**API / Frontend**
- `FastAPI` — async REST API with Pydantic request/response validation
- `Streamlit` — interactive frontend with diversity slider
- `httpx` — async HTTP client for frontend → backend calls

**MLOps**
- `DVC` — data versioning with S3 remote
- `MLflow` — experiment tracking (hyperparameters + NDCG@10 per NMF run)
- Structured JSON logging — `RotatingFileHandler`, CloudWatch-ready
- `Docker Compose` — decoupled backend + frontend containers
- `GitHub Actions` — CI/CD: test → Docker build → ECR push → CodeDeploy
- `AWS EC2 + ECR + CodeDeploy` — cloud deployment (AllAtOnce strategy)

**Dataset**
- [Million Song Dataset + Spotify + LastFM](https://www.kaggle.com/datasets/undefinenull/million-song-dataset-spotify-lastfm) on Kaggle
- 50,683 songs with audio features, 9.7M user play-count records

---

## Project Structure

```
spotify-hybrid-recommender-system/
├── backend/
│   ├── main.py                  # FastAPI app with lifespan model loading
│   ├── config.py                # Pydantic settings (env-overridable)
│   ├── models.py                # Request/response schemas
│   ├── logger.py                # Structured JSON logger
│   ├── pipeline/
│   │   ├── data_cleaning.py     # Dedup, lowercase, fillna
│   │   ├── interaction_matrix.py# NMF training + collab FAISS index
│   │   └── feature_engineering.py # Content FAISS + hybrid FAISS index
│   └── recommenders/
│       ├── content_based.py     # FAISS-backed content recommender
│       ├── collaborative.py     # NMF embedding recommender
│       └── hybrid.py            # Weighted score blending
├── frontend/
│   └── app.py                   # Streamlit UI
├── scripts/
│   ├── train_pipeline.py        # Run pipeline stages + MLflow tracking
│   ├── evaluate.py              # Offline evaluation (Precision/Recall/NDCG@K)
│   └── show_runs.py             # Print MLflow runs from terminal
├── deploy/
│   ├── docker-compose.deploy.yml# Production compose (ECR images)
│   └── scripts/
│       ├── install_dependencies.sh
│       └── start_docker.sh
├── tests/
│   └── test_api.py              # Integration tests (health, search, recommend)
├── Dockerfile.backend
├── Dockerfile.frontend
├── docker-compose.yml           # Local development
├── appspec.yml                  # CodeDeploy hooks
├── dvc.yaml                     # 3-stage DVC pipeline
└── pyproject.toml               # Single source of dependencies
```

---

## Local Setup

### Prerequisites
- Python 3.11+
- [Kaggle dataset](https://www.kaggle.com/datasets/undefinenull/million-song-dataset-spotify-lastfm) — place `Music Info.csv` and `User Listening History.csv` in `data/`

### Install

```bash
git clone https://github.com/syedtaiyabullah/Spotify-Hybrid-Recommender-System
cd spotify-hybrid-recommender-system
pip install -e ".[pipeline,dev]"
```

### Train the pipeline

```bash
# All stages: clean → collab → content  (~10 min)
python scripts/train_pipeline.py

# Or individual stages
python scripts/train_pipeline.py clean
python scripts/train_pipeline.py collab   # NMF + FAISS, logs to MLflow
python scripts/train_pipeline.py content  # Content + hybrid FAISS indexes
```

### Run locally

```bash
# Terminal 1 — backend
uvicorn backend.main:app --reload --port 8000

# Terminal 2 — frontend
streamlit run frontend/app.py
```

Open `http://localhost:8501`

### Run with Docker Compose

```bash
docker compose up --build
```

Open `http://localhost:8501`

---

## MLflow Experiment Tracking

Every collab training run is automatically logged:

```bash
# View runs in the terminal
python scripts/show_runs.py

# Launch dashboard
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```

Logged per run: `nmf_components`, `nmf_max_iter`, `nmf_reconstruction_err`, `n_tracks`, `n_users`, `NDCG@10` for all three methods.

---

## Offline Evaluation

```bash
# Quick (100 users, ~1 min)
python scripts/evaluate.py --n-users 100

# Full (500 users, ~2 min)
python scripts/evaluate.py --n-users 500

# Custom K
python scripts/evaluate.py --n-users 300 --k 20
```

---

## API Reference

### `GET /health`
```json
{ "status": "ok", "content_songs": 50683, "hybrid_songs": 30459 }
```

### `GET /api/v1/songs/search?q=radiohead`
```json
[
  { "name": "No Surprises", "artist": "Radiohead", "in_hybrid_pool": true }
]
```

### `POST /api/v1/recommend`
```json
{
  "song_name": "no surprises",
  "artist_name": "radiohead",
  "k": 10,
  "diversity": 0.5,
  "force_method": "auto"
}
```

`force_method`: `"auto"` (default) | `"content"` | `"collaborative"` | `"hybrid"`

`diversity`: `0.0` = fully content-based, `1.0` = fully collaborative

**Response:**
```json
{
  "query_song": "No Surprises",
  "query_artist": "Radiohead",
  "method": "hybrid",
  "recommendations": [
    { "name": "Fake Plastic Trees", "artist": "Radiohead", "score": 0.94 }
  ]
}
```

---

## Environment Variables

All settings can be overridden via environment variables prefixed with `APP_`:

| Variable | Default | Description |
|---|---|---|
| `APP_NMF_COMPONENTS` | `50` | NMF latent factors |
| `APP_NMF_MAX_ITER` | `300` | NMF max iterations |
| `API_URL` | `http://localhost:8000` | Backend URL (frontend) |

---

## CI/CD Pipeline

On every `git push`:

1. **Test** — `dvc pull` (models from S3) → start API → `pytest tests/`
2. **Build** — Docker build for backend and frontend
3. **Push** — Images pushed to Amazon ECR
4. **Deploy** — CodeDeploy AllAtOnce to EC2, pulls ECR images and runs `docker compose up`
