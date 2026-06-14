"""
Builds the content-based feature pipeline using a sklearn ColumnTransformer
and saves two FAISS indexes:
  - content.faiss  : all songs in cleaned_data (used for pure content recommendations)
  - hybrid_content.faiss : collab-filtered subset (used for hybrid blending)

FAISS uses IndexFlatIP (exact inner product). After L2-normalising the feature
vectors, inner product == cosine similarity.
"""
import numpy as np
import pandas as pd
import joblib
import faiss
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from category_encoders.count import CountEncoder

from .data_cleaning import data_for_content_filtering

_FREQ_COLS = ["year"]
_OHE_COLS = ["artist", "time_signature", "key"]
_TFIDF_COL = "tags"
_STD_COLS = ["duration_ms", "loudness", "tempo"]
_MINMAX_COLS = [
    "danceability", "energy", "speechiness", "acousticness",
    "instrumentalness", "liveness", "valence",
]


def _build_transformer() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("freq_encode", CountEncoder(normalize=True, return_df=True), _FREQ_COLS),
            ("ohe", OneHotEncoder(handle_unknown="ignore"), _OHE_COLS),
            ("tfidf", TfidfVectorizer(max_features=85), _TFIDF_COL),
            ("std_scale", StandardScaler(), _STD_COLS),
            ("minmax_scale", MinMaxScaler(), _MINMAX_COLS),
        ],
        remainder="passthrough",
        n_jobs=-1,
    )


def _to_float32(matrix) -> np.ndarray:
    if hasattr(matrix, "toarray"):
        return matrix.toarray().astype(np.float32)
    return np.asarray(matrix, dtype=np.float32)


def _build_faiss_index(vectors: np.ndarray) -> faiss.IndexFlatIP:
    vecs = vectors.copy()
    faiss.normalize_L2(vecs)
    index = faiss.IndexFlatIP(vecs.shape[1])
    index.add(vecs)
    return index


def run(
    cleaned_data_path: Path,
    transformer_path: Path,
    content_faiss_path: Path,
    collab_data_path: Path | None,
    hybrid_faiss_path: Path | None,
) -> None:
    songs = pd.read_csv(cleaned_data_path)
    features = data_for_content_filtering(songs)

    transformer = _build_transformer()
    transformer.fit(features)
    joblib.dump(transformer, transformer_path)
    print(f"Transformer saved → {transformer_path}")

    all_vectors = _to_float32(transformer.transform(features))
    faiss.write_index(_build_faiss_index(all_vectors), str(content_faiss_path))
    print(f"Content FAISS index saved ({len(songs)} songs, dim={all_vectors.shape[1]}) → {content_faiss_path}")

    if collab_data_path and collab_data_path.exists() and hybrid_faiss_path:
        collab_songs = pd.read_csv(collab_data_path)
        collab_features = data_for_content_filtering(collab_songs)
        hybrid_vectors = _to_float32(transformer.transform(collab_features))
        faiss.write_index(_build_faiss_index(hybrid_vectors), str(hybrid_faiss_path))
        print(f"Hybrid content FAISS index saved ({len(collab_songs)} songs) → {hybrid_faiss_path}")
