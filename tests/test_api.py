"""
Integration tests for the FastAPI backend.
Assumes the server is already running on http://localhost:8000.
The CI pipeline starts the server before pytest and stops it after.
"""
import time

import pytest
import requests

API_URL = "http://localhost:8000"


def _wait_for_server(timeout: int = 90) -> bool:
    start = time.time()
    while time.time() - start < timeout:
        try:
            resp = requests.get(f"{API_URL}/health", timeout=3)
            if resp.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(3)
    return False


@pytest.fixture(scope="session", autouse=True)
def wait_for_api():
    assert _wait_for_server(), "API server did not become healthy within timeout"


def test_health_endpoint():
    resp = requests.get(f"{API_URL}/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["content_songs"] > 0
    assert body["hybrid_songs"] > 0


def test_search_returns_results():
    resp = requests.get(f"{API_URL}/api/v1/songs/search", params={"q": "love"})
    assert resp.status_code == 200
    results = resp.json()
    assert isinstance(results, list)
    for r in results:
        assert "name" in r and "artist" in r and "in_hybrid_pool" in r


def test_recommend_content_based():
    search = requests.get(
        f"{API_URL}/api/v1/songs/search", params={"q": "love", "limit": 20}
    ).json()
    content_song = next((s for s in search if not s["in_hybrid_pool"]), None)
    if content_song is None:
        pytest.skip("No content-only song found in search results")

    resp = requests.post(
        f"{API_URL}/api/v1/recommend",
        json={
            "song_name": content_song["name"],
            "artist_name": content_song["artist"],
            "k": 5,
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["method"] == "content"
    assert 0 < len(body["recommendations"]) <= 5


def test_recommend_hybrid():
    search = requests.get(
        f"{API_URL}/api/v1/songs/search", params={"q": "love", "limit": 20}
    ).json()
    hybrid_song = next((s for s in search if s["in_hybrid_pool"]), None)
    if hybrid_song is None:
        pytest.skip("No hybrid-eligible song found in search results")

    resp = requests.post(
        f"{API_URL}/api/v1/recommend",
        json={
            "song_name": hybrid_song["name"],
            "artist_name": hybrid_song["artist"],
            "k": 5,
            "diversity": 0.5,
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["method"] == "hybrid"
    assert 0 < len(body["recommendations"]) <= 5


def test_recommend_unknown_song_returns_404():
    resp = requests.post(
        f"{API_URL}/api/v1/recommend",
        json={
            "song_name": "this song definitely does not exist xyz123",
            "artist_name": "no artist",
        },
    )
    assert resp.status_code == 404
