"""
scripts/evaluate.py
-------------------
Offline evaluation of the three recommendation methods using a
leave-one-out strategy on User Listening History.

Strategy
--------
For each sampled user:
  Seed        = their most-played track  (used as the recommendation query)
  Ground truth = all other tracks they listened to  (held-out "relevant" items)
  Ask each method: "recommend K songs similar to Seed"
  Score: how many ground-truth tracks appear in the top-K list?

Metrics
-------
  Precision@K   -- fraction of the K recommendations that are relevant
  Recall@K      -- fraction of the user's relevant songs that were retrieved
  NDCG@K        -- like precision but rewards hitting relevant items earlier

Usage
-----
  # Quick test (100 users, ~1 min)
  python scripts/evaluate.py --n-users 100

  # Full evaluation (500 users, default)
  python scripts/evaluate.py

  # Custom K cutoff
  python scripts/evaluate.py --n-users 300 --k 20
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

# Make backend importable from the scripts/ folder
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from backend.config import settings
from backend.recommenders.collaborative import CollaborativeRecommender
from backend.recommenders.content_based import ContentRecommender
from backend.recommenders.hybrid import HybridRecommender


# -------------------------------------------------------------------------
# Metric helpers
# -------------------------------------------------------------------------

def precision_at_k(recs: list, relevant: set, k: int) -> float:
    """Fraction of top-K recommendations that are relevant."""
    return len(set(recs[:k]) & relevant) / k


def recall_at_k(recs: list, relevant: set, k: int) -> float:
    """Fraction of all relevant items that appear in top-K."""
    hits = len(set(recs[:k]) & relevant)
    return hits / len(relevant)


def ndcg_at_k(recs: list, relevant: set, k: int) -> float:
    """
    Normalised Discounted Cumulative Gain @ K.
    Rewards finding relevant items AND finding them near the top.
    """
    dcg = sum(
        1.0 / np.log2(i + 2)
        for i, track_id in enumerate(recs[:k])
        if track_id in relevant
    )
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(ideal_hits))
    return dcg / idcg if idcg > 0 else 0.0


# -------------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------------

def _build_track_lookup(songs_df: pd.DataFrame) -> dict:
    """(name_lower, artist_lower) -> track_id"""
    return {
        (row["name"], row["artist"]): row["track_id"]
        for _, row in songs_df.iterrows()
    }


def _recs_to_track_ids(df_recs: pd.DataFrame, lookup: dict) -> list:
    """Convert a recommendations DataFrame (name, artist) to a list of track_ids."""
    ids = []
    for _, row in df_recs.iterrows():
        key = (row["name"].lower(), row["artist"].lower())
        tid = lookup.get(key)
        if tid is not None:
            ids.append(tid)
    return ids


# -------------------------------------------------------------------------
# Core evaluation — returns a dict, used by both CLI and MLflow
# -------------------------------------------------------------------------

def compute_metrics(
    n_users: int = 100,
    k: int = 10,
    seed: int = 42,
    verbose: bool = True,
) -> dict:
    """
    Run offline leave-one-out evaluation and return a metrics dict.

    Returns
    -------
    {
        "content": {"precision": float, "recall": float, "ndcg": float, "n": int},
        "collab":  {"precision": float, "recall": float, "ndcg": float, "n": int},
        "hybrid":  {"precision": float, "recall": float, "ndcg": float, "n": int},
    }
    """
    rng = np.random.default_rng(seed)

    # -- Load models ----------------------------------------------------------
    if verbose:
        print("Loading models...")
    content_rec = ContentRecommender.load(settings)
    collab_rec  = CollaborativeRecommender.load(settings)
    hybrid_rec  = HybridRecommender.load(settings, collab_rec)
    if verbose:
        print(f"  Content pool : {len(content_rec.songs):>6,} songs")
        print(f"  Collab pool  : {len(collab_rec.songs):>6,} songs")

    track_lookup = _build_track_lookup(content_rec.songs)
    id_to_info   = {
        row["track_id"]: (row["name"], row["artist"])
        for _, row in collab_rec.songs.iterrows()
    }
    valid_track_ids = set(collab_rec.songs["track_id"])

    # -- Load and filter user listening history --------------------------------
    if verbose:
        print("\nLoading user listening history...")
    history = pd.read_csv(
        settings.user_history_path,
        usecols=["user_id", "track_id", "playcount"],
        dtype={"user_id": str, "track_id": str, "playcount": float},
    )
    history = history[history["track_id"].isin(valid_track_ids)]

    # -- Sample eligible users ------------------------------------------------
    MIN_SONGS = 5
    user_song_counts = (
        history.groupby("user_id")["track_id"]
        .nunique()
        .reset_index(name="n_tracks")
    )
    eligible_users = user_song_counts[
        user_song_counts["n_tracks"] >= MIN_SONGS
    ]["user_id"].values

    sample_size   = min(n_users, len(eligible_users))
    sampled_users = rng.choice(eligible_users, size=sample_size, replace=False)

    if verbose:
        print(f"  Eligible users (>={MIN_SONGS} songs): {len(eligible_users):,}")
        print(f"  Evaluating {sample_size:,} users  (K={k})\n")

    # Pre-group history for fast per-user lookup
    user_history_subset = history[history["user_id"].isin(sampled_users)]
    user_tracks = {
        uid: grp.sort_values("playcount", ascending=False)["track_id"].tolist()
        for uid, grp in user_history_subset.groupby("user_id")
    }

    # -- Evaluation loop -------------------------------------------------------
    accumulator = {
        m: {"precision": [], "recall": [], "ndcg": []}
        for m in ("content", "collab", "hybrid")
    }
    skipped = 0

    for user_id in tqdm(sampled_users, desc="Evaluating", unit="user", disable=not verbose):
        track_ids_sorted = user_tracks.get(user_id, [])

        if len(track_ids_sorted) < 2:
            skipped += 1
            continue

        seed_track_id = track_ids_sorted[0]
        ground_truth  = set(track_ids_sorted[1:])

        if seed_track_id not in id_to_info:
            skipped += 1
            continue

        seed_name, seed_artist = id_to_info[seed_track_id]

        # Content
        try:
            recs_c = _recs_to_track_ids(
                content_rec.recommend(seed_name, seed_artist, k=k), track_lookup
            )
        except Exception:
            recs_c = []

        # Collaborative (weight_content=0)
        try:
            recs_co = _recs_to_track_ids(
                hybrid_rec.recommend(seed_name, seed_artist, k=k, weight_content=0.0),
                track_lookup,
            )
        except Exception:
            recs_co = []

        # Hybrid (50/50)
        try:
            recs_h = _recs_to_track_ids(
                hybrid_rec.recommend(seed_name, seed_artist, k=k, weight_content=0.5),
                track_lookup,
            )
        except Exception:
            recs_h = []

        for recs, method in [
            (recs_c,  "content"),
            (recs_co, "collab"),
            (recs_h,  "hybrid"),
        ]:
            if not recs:
                continue
            acc = accumulator[method]
            acc["precision"].append(precision_at_k(recs, ground_truth, k))
            acc["recall"].append(recall_at_k(recs, ground_truth, k))
            acc["ndcg"].append(ndcg_at_k(recs, ground_truth, k))

    if verbose:
        print(f"\n  Skipped {skipped} users (seed outside collab pool or too few songs)\n")

    # -- Aggregate -------------------------------------------------------------
    results = {}
    for method in ("content", "collab", "hybrid"):
        acc = accumulator[method]
        n   = len(acc["precision"])
        if n == 0:
            results[method] = {"precision": 0.0, "recall": 0.0, "ndcg": 0.0, "n": 0}
        else:
            results[method] = {
                "precision": float(np.mean(acc["precision"])),
                "recall":    float(np.mean(acc["recall"])),
                "ndcg":      float(np.mean(acc["ndcg"])),
                "n":         n,
            }

    return results


# -------------------------------------------------------------------------
# CLI: pretty-print the results table
# -------------------------------------------------------------------------

def evaluate(n_users: int = 500, k: int = 10, seed: int = 42) -> None:
    print("=" * 60)
    print(f"  OFFLINE EVALUATION  (n_users={n_users}, K={k})")
    print("=" * 60)

    results = compute_metrics(n_users=n_users, k=k, seed=seed, verbose=True)

    print("=" * 60)
    print(f"  RESULTS  (K={k})")
    print("=" * 60)
    header = f"{'Method':<18} {'Precision@'+str(k):<16} {'Recall@'+str(k):<14} {'NDCG@'+str(k):<12} {'N users'}"
    print(header)
    print("-" * 72)

    for method, label in [
        ("content", "Content-Based"),
        ("collab",  "Collaborative"),
        ("hybrid",  "Hybrid"),
    ]:
        r = results[method]
        if r["n"] == 0:
            print(f"{label:<18} {'N/A':>12}   {'N/A':>10}    {'N/A':>10}   0")
        else:
            print(
                f"{label:<18} {r['precision']:>12.4f}   {r['recall']:>10.4f}"
                f"    {r['ndcg']:>10.4f}   {r['n']:>6,}"
            )

    print("-" * 72)

    hybrid  = results["hybrid"]["ndcg"]
    content = results["content"]["ndcg"]
    collab  = results["collab"]["ndcg"]

    if results["hybrid"]["n"] > 0 and hybrid > content and hybrid > collab:
        imp_c = (hybrid - content) / content * 100
        imp_co = (hybrid - collab) / collab * 100
        print(f"\n  [OK] Hybrid outperforms Content by  {imp_c:+.1f}%  NDCG@{k}")
        print(f"  [OK] Hybrid outperforms Collab  by  {imp_co:+.1f}%  NDCG@{k}")
    else:
        best = max(results, key=lambda m: results[m]["ndcg"])
        print(f"\n  [INFO] Best method by NDCG@{k}: {best}")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Offline evaluation of Content / Collaborative / Hybrid recommenders"
    )
    parser.add_argument("--n-users", type=int, default=500,
                        help="Number of users to sample (default: 500)")
    parser.add_argument("--k",       type=int, default=10,
                        help="Top-K cutoff (default: 10)")
    parser.add_argument("--seed",    type=int, default=42,
                        help="Random seed (default: 42)")
    args = parser.parse_args()
    evaluate(n_users=args.n_users, k=args.k, seed=args.seed)
