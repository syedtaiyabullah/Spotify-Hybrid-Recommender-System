import pandas as pd
from pathlib import Path


def clean_data(data: pd.DataFrame) -> pd.DataFrame:
    return (
        data
        .drop_duplicates(subset="track_id")
        .drop(columns=["genre", "spotify_id"])
        .fillna({"tags": "no_tags"})
        .assign(
            name=lambda x: x["name"].str.lower(),
            artist=lambda x: x["artist"].str.lower(),
            tags=lambda x: x["tags"].str.lower(),
        )
        .reset_index(drop=True)
    )


def data_for_content_filtering(data: pd.DataFrame) -> pd.DataFrame:
    return data.drop(columns=["track_id", "name", "spotify_preview_url"])


def run(music_info_path: Path, output_path: Path) -> None:
    data = pd.read_csv(music_info_path)
    cleaned = clean_data(data)
    cleaned.to_csv(output_path, index=False)
    print(f"Saved {len(cleaned)} rows → {output_path}")
