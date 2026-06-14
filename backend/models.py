from typing import Literal
from pydantic import BaseModel, Field


class RecommendRequest(BaseModel):
    song_name: str
    artist_name: str
    k: int = Field(default=10, ge=1, le=50)
    diversity: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="0 = fully content-based, 1 = fully collaborative",
    )
    force_method: Literal["content", "collaborative", "hybrid", "auto"] = Field(
        default="auto",
        description="auto = detect from pool; force a specific method otherwise",
    )


class Song(BaseModel):
    name: str
    artist: str
    spotify_preview_url: str | None = None
    score: float | None = None


class RecommendResponse(BaseModel):
    query_song: str
    query_artist: str
    method: Literal["content", "collaborative", "hybrid"]
    recommendations: list[Song]


class SearchResult(BaseModel):
    name: str
    artist: str
    in_hybrid_pool: bool


class HealthResponse(BaseModel):
    status: str
    content_songs: int
    hybrid_songs: int
