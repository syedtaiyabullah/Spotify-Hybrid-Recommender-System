import os

import httpx
import pandas as pd
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")
_CLIENT_TIMEOUT = httpx.Timeout(30.0)

FILTER_CONTENT = "Content-Based Filtering"
FILTER_COLLAB  = "Collaborative Filtering"
FILTER_HYBRID  = "Hybrid Recommender System"

_METHOD_MAP = {
    FILTER_CONTENT: "content",
    FILTER_COLLAB:  "collaborative",
    FILTER_HYBRID:  "hybrid",
}

_METHOD_LABELS = {
    "content":       "Content-Based Filtering",
    "collaborative": "Collaborative Filtering",
    "hybrid":        "Hybrid Recommender System",
}


# ── API helpers ────────────────────────────────────────────────────────────────

def get_recommendations(
    song_name: str,
    artist_name: str,
    k: int,
    diversity: float,
    force_method: str = "auto",
) -> dict | None:
    try:
        resp = httpx.post(
            f"{API_URL}/api/v1/recommend",
            json={
                "song_name":    song_name,
                "artist_name":  artist_name,
                "k":            k,
                "diversity":    diversity,
                "force_method": force_method,
            },
            timeout=_CLIENT_TIMEOUT,
        )
        if resp.status_code == 404:
            return {"error": "not_found"}
        if resp.status_code == 400:
            return {"error": "method_unavailable", "detail": resp.json().get("detail", "")}
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        st.error(f"Could not reach the recommendation service: {exc}")
        return None


# ── Page config ────────────────────────────────────────────────────────────────

st.set_page_config(page_title="Spotify Recommender", page_icon="🎵", layout="centered")

st.title("🎵 Spotify Song Recommender")
st.write("Enter a song and discover what to listen to next.")

# ── Inputs ─────────────────────────────────────────────────────────────────────

col1, col2 = st.columns(2)
with col1:
    song_name = st.text_input("Enter the song name:", placeholder="e.g. bohemian rhapsody")
    if song_name:
        st.write(f"You entered: {song_name.title()}")
with col2:
    artist_name = st.text_input("Enter the artist name:", placeholder="e.g. queen")
    if artist_name:
        st.write(f"You entered: {artist_name.title()}")

k = st.selectbox("How many recommendations do you want?", [5, 10, 15, 20], index=1)

# ── Filtering type selector ────────────────────────────────────────────────────

filtering_type = st.selectbox(
    "Select the type of filtering:",
    [FILTER_HYBRID, FILTER_CONTENT, FILTER_COLLAB],
    index=0,
    help=(
        "Hybrid — blends audio features + listening patterns (recommended).\n"
        "Content-Based — audio features only (tempo, energy, key, tags…).\n"
        "Collaborative — listening patterns only (what similar users play).\n\n"
        "Note: Hybrid and Collaborative require the song to have user listening history."
    ),
    key="filtering_type",
)

# ── Diversity slider (only for hybrid) ────────────────────────────────────────

diversity = 0.5
if filtering_type == FILTER_HYBRID:
    diversity_int = st.slider(
        "Diversity in Recommendations",
        min_value=1,
        max_value=9,
        value=5,
        step=1,
        help="1 = mostly audio features (personalised).  9 = mostly listening patterns (diverse).",
    )
    diversity = diversity_int / 10

    chart_data = pd.DataFrame(
        {"type": ["Personalised", "Diverse"], "ratio": [10 - diversity_int, diversity_int]}
    )
    st.bar_chart(chart_data, x="type", y="ratio", use_container_width=True)

# ── Recommend button ───────────────────────────────────────────────────────────

can_submit = bool(song_name.strip() and artist_name.strip())
if st.button("Get Recommendations", type="primary", disabled=not can_submit):
    with st.spinner("Finding recommendations..."):
        result = get_recommendations(
            song_name.strip().lower(),
            artist_name.strip().lower(),
            k,
            diversity,
            force_method=_METHOD_MAP[filtering_type],
        )

    if result is None:
        pass  # error already shown by get_recommendations

    elif result.get("error") == "not_found":
        st.error(
            f"Could not find **{song_name.title()}** by **{artist_name.title()}** "
            "in the database. Check the spelling or try another song."
        )

    elif result.get("error") == "method_unavailable":
        st.warning(
            f"**{filtering_type}** is not available for this song — "
            "it has no user listening history in the dataset. "
            "Try **Content-Based Filtering** instead."
        )

    else:
        label = _METHOD_LABELS.get(result["method"], result["method"])
        st.success(
            f"Recommendations for **{result['query_song']}** by "
            f"**{result['query_artist']}** — {label}"
        )

        for i, song in enumerate(result["recommendations"]):
            with st.container():
                if i == 0:
                    st.markdown("## Currently Playing")
                    st.markdown(f"#### **{song['name']}** by **{song['artist']}**")
                elif i == 1:
                    st.markdown("### Next Up 🎵")
                    st.markdown(f"#### {i}. **{song['name']}** by **{song['artist']}**")
                else:
                    st.markdown(f"#### {i}. **{song['name']}** by **{song['artist']}**")

                preview = song.get("spotify_preview_url")
                if preview:
                    st.audio(preview)
                st.write("---")
