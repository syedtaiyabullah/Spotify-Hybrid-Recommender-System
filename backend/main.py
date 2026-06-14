import time
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Request

from .config import settings
from .logger import get_logger
from .models import HealthResponse, RecommendRequest, RecommendResponse, SearchResult, Song
from .recommenders.collaborative import CollaborativeRecommender
from .recommenders.content_based import ContentRecommender
from .recommenders.hybrid import HybridRecommender

logger = get_logger()


# ── Startup / shutdown ────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading models...", extra={"event": "startup_begin"})
    t0 = time.perf_counter()

    app.state.content_rec = ContentRecommender.load(settings)
    logger.info(
        "Content recommender ready",
        extra={
            "event": "model_loaded",
            "model": "content",
            "songs": len(app.state.content_rec.songs),
        },
    )

    app.state.collab_rec = CollaborativeRecommender.load(settings)
    logger.info(
        "Collaborative recommender ready",
        extra={
            "event": "model_loaded",
            "model": "collaborative",
            "tracks": len(app.state.collab_rec.track_ids),
            "nmf_components": settings.nmf_components,
        },
    )

    app.state.hybrid_rec = HybridRecommender.load(settings, app.state.collab_rec)
    logger.info(
        "Hybrid recommender ready",
        extra={"event": "model_loaded", "model": "hybrid"},
    )

    elapsed = round((time.perf_counter() - t0) * 1000)
    logger.info(
        "All models loaded — API ready",
        extra={"event": "startup_complete", "duration_ms": elapsed},
    )
    yield

    logger.info("API shutting down", extra={"event": "shutdown"})


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Spotify Hybrid Recommender API",
    version="1.0.0",
    description="Hybrid music recommendation using NMF matrix factorisation and FAISS similarity search.",
    lifespan=lifespan,
)


# ── Middleware: log every request ─────────────────────────────────────────────

@app.middleware("http")
async def log_requests(request: Request, call_next):
    t0 = time.perf_counter()
    response = await call_next(request)
    duration_ms = round((time.perf_counter() - t0) * 1000, 2)

    logger.info(
        f"{request.method} {request.url.path}",
        extra={
            "event": "http_request",
            "method": request.method,
            "path": request.url.path,
            "status_code": response.status_code,
            "duration_ms": duration_ms,
        },
    )
    return response


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["meta"])
async def health():
    content_rec: ContentRecommender = app.state.content_rec
    collab_rec: CollaborativeRecommender = app.state.collab_rec
    return HealthResponse(
        status="ok",
        content_songs=len(content_rec.songs),
        hybrid_songs=len(collab_rec.songs),
    )


@app.get("/api/v1/songs/search", response_model=list[SearchResult], tags=["songs"])
async def search_songs(q: str = Query(min_length=1), limit: int = Query(default=10, le=50)):
    content_rec: ContentRecommender = app.state.content_rec
    collab_rec: CollaborativeRecommender = app.state.collab_rec

    q_lower = q.lower()
    mask = content_rec.songs["name"].str.contains(q_lower, case=False, na=False)
    matches = content_rec.songs[mask].head(limit)

    collab_names = set(
        (collab_rec.songs["name"] + "|||" + collab_rec.songs["artist"]).str.lower()
    )

    results = [
        SearchResult(
            name=row["name"].title(),
            artist=row["artist"].title(),
            in_hybrid_pool=(row["name"].lower() + "|||" + row["artist"].lower()) in collab_names,
        )
        for _, row in matches.iterrows()
    ]

    logger.info(
        f"Search: '{q}' → {len(results)} results",
        extra={"event": "search", "query": q, "results_count": len(results)},
    )
    return results


@app.post("/api/v1/recommend", response_model=RecommendResponse, tags=["recommend"])
async def recommend(req: RecommendRequest):
    content_rec: ContentRecommender = app.state.content_rec
    collab_rec: CollaborativeRecommender = app.state.collab_rec
    hybrid_rec: HybridRecommender = app.state.hybrid_rec

    song_name   = req.song_name.lower().strip()
    artist_name = req.artist_name.lower().strip()
    t0 = time.perf_counter()

    collab_songs = collab_rec.songs
    in_hybrid_pool = (
        (collab_songs["name"] == song_name) & (collab_songs["artist"] == artist_name)
    ).any()

    # Resolve which method to use
    force = req.force_method
    if force == "auto":
        use_method = "hybrid" if in_hybrid_pool else "content"
    elif force in ("collaborative", "hybrid") and not in_hybrid_pool:
        logger.warning(
            f"Method '{force}' requested but '{song_name}' has no listening history",
            extra={
                "event": "method_unavailable",
                "song": song_name,
                "artist": artist_name,
                "requested_method": force,
            },
        )
        raise HTTPException(
            status_code=400,
            detail=f"'{force}' filtering requires user listening history for this song.",
        )
    else:
        use_method = force

    try:
        if use_method == "content":
            df = content_rec.recommend(song_name, artist_name, req.k)
            method = "content"
        elif use_method == "collaborative":
            df = hybrid_rec.recommend(song_name, artist_name, req.k, weight_content=0.0)
            method = "collaborative"
        else:
            weight_content = 1.0 - req.diversity
            df = hybrid_rec.recommend(song_name, artist_name, req.k, weight_content)
            method = "hybrid"

    except ValueError as exc:
        logger.warning(
            f"Song not found: '{song_name}' by '{artist_name}'",
            extra={
                "event": "song_not_found",
                "song": song_name,
                "artist": artist_name,
            },
        )
        raise HTTPException(status_code=404, detail=str(exc))

    except Exception as exc:
        logger.error(
            f"Unexpected error during recommendation: {exc}",
            extra={
                "event": "recommendation_error",
                "song": song_name,
                "artist": artist_name,
                "method": use_method,
            },
            exc_info=True,
        )
        raise HTTPException(status_code=500, detail="Internal server error")

    duration_ms = round((time.perf_counter() - t0) * 1000, 2)
    logger.info(
        f"Recommended {len(df)} songs for '{song_name}' via {method}",
        extra={
            "event":         "recommendation_served",
            "song":          song_name,
            "artist":        artist_name,
            "method":        method,
            "k":             req.k,
            "diversity":     req.diversity,
            "results_count": len(df),
            "duration_ms":   duration_ms,
        },
    )

    def _preview(val) -> str | None:
        return val if (isinstance(val, str) and val.startswith("http")) else None

    songs = [
        Song(
            name=row["name"].title(),
            artist=row["artist"].title(),
            spotify_preview_url=_preview(row.get("spotify_preview_url")),
            score=float(row["score"]) if pd.notna(row.get("score")) else None,
        )
        for _, row in df.iterrows()
    ]

    return RecommendResponse(
        query_song=song_name.title(),
        query_artist=artist_name.title(),
        method=method,
        recommendations=songs,
    )
