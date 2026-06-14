import numpy as np
import pandas as pd
import joblib
import faiss
from sklearn.compose import ColumnTransformer

from ..config import Settings
from ..pipeline.data_cleaning import data_for_content_filtering


class ContentRecommender:
    """
    Content-based recommender backed by a FAISS IndexFlatIP.

    Features are built by a sklearn ColumnTransformer (TF-IDF on tags,
    OHE on artist/key/time_signature, frequency encoding on year,
    standard/min-max scaling on audio features).

    Vectors are L2-normalised so inner product == cosine similarity.
    The normalised matrix is also kept in memory for full-score queries
    needed by the hybrid recommender.
    """

    def __init__(
        self,
        index: faiss.IndexFlatIP,
        transformer: ColumnTransformer,
        songs: pd.DataFrame,
        vectors: np.ndarray,  # L2-normalised, shape (n_songs, d)
    ):
        self._index = index
        self._transformer = transformer
        self._songs = songs
        self._vectors = vectors

    @classmethod
    def load(cls, settings: Settings) -> "ContentRecommender":
        index = faiss.read_index(str(settings.content_faiss_path))
        transformer = joblib.load(settings.content_transformer_path)
        songs = pd.read_csv(settings.cleaned_data_path)
        features = data_for_content_filtering(songs)
        raw = transformer.transform(features)
        if hasattr(raw, "toarray"):
            raw = raw.toarray()
        vectors = np.asarray(raw, dtype=np.float32)
        faiss.normalize_L2(vectors)
        return cls(index, transformer, songs, vectors)

    def recommend(self, song_name: str, artist_name: str, k: int) -> pd.DataFrame:
        song_name, artist_name = song_name.lower(), artist_name.lower()
        mask = (self._songs["name"] == song_name) & (self._songs["artist"] == artist_name)
        if not mask.any():
            raise ValueError(f"Song not found: '{song_name}' by '{artist_name}'")

        idx = int(self._songs.index[mask][0])
        query = self._vectors[idx : idx + 1].copy()
        scores, indices = self._index.search(query, k + 1)

        results = self._songs.iloc[indices[0]].copy()
        results["score"] = scores[0]
        return (
            results[results.index != idx][["name", "artist", "spotify_preview_url", "score"]]
            .head(k)
            .reset_index(drop=True)
        )

    def get_scores(self, song_idx: int) -> np.ndarray:
        """Cosine similarity from song_idx to every song (for hybrid blending)."""
        query = self._vectors[song_idx : song_idx + 1]
        return (query @ self._vectors.T).ravel()

    @property
    def songs(self) -> pd.DataFrame:
        return self._songs
