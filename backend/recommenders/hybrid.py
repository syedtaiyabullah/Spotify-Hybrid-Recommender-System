"""
Hybrid recommender: weighted blend of content-based and collaborative scores.

Only available for songs that appear in the collaborative pool (i.e., songs
that have user listening history). Falls back to content-based otherwise.

Scoring:
  combined[i] = weight_content * norm(content_scores[i])
              + (1 - weight_content) * norm(collab_scores[i])

where norm() is min-max normalisation to [0, 1].

Both score vectors are full-length (all songs in the collab pool), computed via
a single numpy matrix-multiply on the cached normalised embeddings rather than
FAISS top-k retrieval — this is fast because the embedding dimension (50) is small.
"""
import numpy as np
import pandas as pd
import faiss

from ..config import Settings
from .content_based import ContentRecommender
from .collaborative import CollaborativeRecommender


def _minmax(arr: np.ndarray) -> np.ndarray:
    mn, mx = arr.min(), arr.max()
    if mx == mn:
        return np.zeros_like(arr)
    return (arr - mn) / (mx - mn)


class HybridRecommender:
    def __init__(
        self,
        collab_rec: CollaborativeRecommender,
        hybrid_content_vectors: np.ndarray,  # L2-normalised, shape (n_collab, d_content)
    ):
        self._collab_rec = collab_rec
        self._hcv = hybrid_content_vectors  # aligned with collab_songs row order

    @classmethod
    def load(
        cls,
        settings: Settings,
        collab_rec: CollaborativeRecommender,
    ) -> "HybridRecommender":
        index = faiss.read_index(str(settings.hybrid_content_faiss_path))
        n, d = index.ntotal, index.d
        vectors = np.empty((n, d), dtype=np.float32)
        index.reconstruct_n(0, n, vectors)
        return cls(collab_rec, vectors)

    def recommend(
        self,
        song_name: str,
        artist_name: str,
        k: int,
        weight_content: float,
    ) -> pd.DataFrame:
        song_name, artist_name = song_name.lower(), artist_name.lower()
        collab_songs = self._collab_rec.songs

        mask = (collab_songs["name"] == song_name) & (collab_songs["artist"] == artist_name)
        pool_pos = int(collab_songs.index[mask][0])  # positional index in collab_songs

        # Content scores over the collab subset
        query_c = self._hcv[pool_pos : pool_pos + 1]
        content_scores = (query_c @ self._hcv.T).ravel()

        # Collaborative scores over the collab subset
        track_id = collab_songs.iloc[pool_pos]["track_id"]
        track_idx = self._collab_rec.get_track_idx(track_id)
        collab_scores = self._collab_rec.get_scores(track_idx)

        combined = (
            weight_content * _minmax(content_scores)
            + (1.0 - weight_content) * _minmax(collab_scores)
        )
        combined[pool_pos] = -1.0  # exclude query song

        top_pos = np.argsort(combined)[::-1][:k]
        result = collab_songs.iloc[top_pos].copy()
        result["score"] = combined[top_pos]
        return result[["name", "artist", "spotify_preview_url", "score"]].reset_index(drop=True)
