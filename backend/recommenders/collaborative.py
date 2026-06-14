import numpy as np
import pandas as pd
import faiss

from ..config import Settings


class CollaborativeRecommender:
    """
    Collaborative filtering recommender using NMF track embeddings.

    NMF factorises the (n_tracks × n_users) play-count matrix into
    track embeddings W (n_tracks × n_components). Songs with similar
    embeddings attract the same listeners.

    A FAISS IndexFlatIP over the L2-normalised embeddings gives fast
    cosine-similarity lookup. The raw normalised matrix is also kept
    in memory for full-score queries used by the hybrid recommender.

    Alignment: track_ids[i], embeddings[i], and collab_songs.iloc[i]
    all refer to the same track (both sorted by track_id alphabetically).
    """

    def __init__(
        self,
        index: faiss.IndexFlatIP,
        embeddings: np.ndarray,  # L2-normalised, shape (n_tracks, n_components)
        songs: pd.DataFrame,
        track_ids: np.ndarray,
    ):
        self._index = index
        self._embeddings = embeddings
        self._songs = songs
        self._track_ids = track_ids

    @classmethod
    def load(cls, settings: Settings) -> "CollaborativeRecommender":
        index = faiss.read_index(str(settings.collab_faiss_path))
        track_ids = np.load(settings.track_ids_path, allow_pickle=True)
        songs = pd.read_csv(settings.collab_data_path)

        # Reconstruct the stored (already normalised) vectors from the flat index
        n, d = index.ntotal, index.d
        embeddings = np.empty((n, d), dtype=np.float32)
        index.reconstruct_n(0, n, embeddings)

        return cls(index, embeddings, songs, track_ids)

    def recommend(self, song_name: str, artist_name: str, k: int) -> pd.DataFrame:
        song_name, artist_name = song_name.lower(), artist_name.lower()
        mask = (self._songs["name"] == song_name) & (self._songs["artist"] == artist_name)
        if not mask.any():
            raise ValueError(f"Song not in collaborative pool: '{song_name}' by '{artist_name}'")

        track_id = self._songs.loc[mask, "track_id"].values[0]
        track_idx = self.get_track_idx(track_id)
        query = self._embeddings[track_idx : track_idx + 1].copy()
        scores, indices = self._index.search(query, k + 1)

        rec_track_ids = self._track_ids[indices[0]]
        score_map = dict(zip(rec_track_ids, scores[0]))
        result = (
            self._songs[self._songs["track_id"].isin(rec_track_ids)]
            .copy()
            .assign(score=lambda df: df["track_id"].map(score_map))
            .sort_values("score", ascending=False)
        )
        return (
            result[result["track_id"] != track_id]
            [["name", "artist", "spotify_preview_url", "score"]]
            .head(k)
            .reset_index(drop=True)
        )

    def get_scores(self, track_idx: int) -> np.ndarray:
        """Cosine similarity from track_idx to every track in the pool (for hybrid blending)."""
        query = self._embeddings[track_idx : track_idx + 1]
        return (query @ self._embeddings.T).ravel()

    def get_track_idx(self, track_id: str) -> int:
        return int(np.where(self._track_ids == track_id)[0][0])

    @property
    def songs(self) -> pd.DataFrame:
        return self._songs

    @property
    def track_ids(self) -> np.ndarray:
        return self._track_ids
