"""
Builds the collaborative filtering pipeline:
  1. Reads user listening history with Dask (handles large CSV efficiently)
  2. Constructs a sparse (n_tracks × n_users) play-count matrix
  3. Trains NMF to produce 50-dimensional track embeddings
  4. Saves a FAISS index over the normalised track embeddings

Track embeddings from NMF capture latent listening patterns shared across users.
Cosine similarity on normalised embeddings (via FAISS IndexFlatIP) finds tracks
that appeal to the same audience — the collaborative signal.

Alignment guarantee: both track_ids.npy and collab_filtered_data.csv are sorted
by track_id (alphabetically), so row i in each array/DataFrame refers to the
same track throughout the recommendation pipeline.
"""
import numpy as np
import pandas as pd
import joblib
import faiss
import dask.dataframe as dd
from pathlib import Path
from scipy.sparse import csr_matrix
from sklearn.decomposition import NMF


def _build_interaction_matrix(
    history_data: dd.DataFrame,
) -> tuple[csr_matrix, np.ndarray]:
    df = history_data.copy()
    df["playcount"] = df["playcount"].astype(np.float64)
    df = df.categorize(columns=["user_id", "track_id"])

    user_idx = df["user_id"].cat.codes
    track_idx = df["track_id"].cat.codes
    track_ids = df["track_id"].cat.categories.values  # sorted alphabetically

    df = df.assign(user_idx=user_idx, track_idx=track_idx)
    agg = (
        df.groupby(["track_idx", "user_idx"])["playcount"]
        .sum()
        .reset_index()
        .compute()
    )

    n_tracks = int(agg["track_idx"].max()) + 1
    n_users = int(agg["user_idx"].max()) + 1
    matrix = csr_matrix(
        (agg["playcount"].values, (agg["track_idx"].values, agg["user_idx"].values)),
        shape=(n_tracks, n_users),
    )
    return matrix, track_ids


def _train_nmf(
    matrix: csr_matrix, n_components: int, max_iter: int
) -> tuple[NMF, np.ndarray]:
    nmf = NMF(
        n_components=n_components,
        init="nndsvda",
        random_state=42,
        max_iter=max_iter,
    )
    track_embeddings = nmf.fit_transform(matrix).astype(np.float32)
    return nmf, track_embeddings


def _build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    vecs = embeddings.copy()
    faiss.normalize_L2(vecs)
    index = faiss.IndexFlatIP(vecs.shape[1])
    index.add(vecs)
    return index


def run(
    user_history_path: Path,
    songs_data_path: Path,
    track_ids_path: Path,
    collab_data_path: Path,
    nmf_path: Path,
    collab_faiss_path: Path,
    n_components: int = 50,
    max_iter: int = 300,
) -> dict:
    """
    Run the collaborative filtering pipeline.

    Returns a stats dict with training metadata, used by MLflow logging
    in train_pipeline.py.
    """
    print("Reading user listening history...")
    user_data = dd.read_csv(user_history_path)
    songs = pd.read_csv(songs_data_path)

    print("Building interaction matrix...")
    matrix, track_ids = _build_interaction_matrix(user_data)
    np.save(track_ids_path, track_ids, allow_pickle=True)
    print(f"Interaction matrix shape: {matrix.shape}")

    # Filter songs to those present in user history, sorted to match track_ids order
    collab_songs = (
        songs[songs["track_id"].isin(track_ids)]
        .sort_values("track_id")
        .reset_index(drop=True)
    )
    collab_songs.to_csv(collab_data_path, index=False)
    print(f"Collab filtered data saved ({len(collab_songs)} songs) -> {collab_data_path}")

    print(f"Training NMF (n_components={n_components})...")
    nmf, embeddings = _train_nmf(matrix, n_components, max_iter)
    joblib.dump(nmf, nmf_path)

    faiss.write_index(_build_faiss_index(embeddings), str(collab_faiss_path))
    print(f"Collab FAISS index saved ({len(track_ids)} tracks, dim={n_components}) -> {collab_faiss_path}")

    return {
        "n_tracks":               int(matrix.shape[0]),
        "n_users":                int(matrix.shape[1]),
        "collab_songs":           int(len(collab_songs)),
        "nmf_reconstruction_err": float(nmf.reconstruction_err_),
        "nmf_n_iter":             int(nmf.n_iter_),
    }
