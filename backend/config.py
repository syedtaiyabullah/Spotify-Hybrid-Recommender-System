from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    data_dir: Path = Path("data")
    models_dir: Path = Path("models")

    # Raw data
    music_info_path: Path = Path("data/Music Info.csv")
    user_history_path: Path = Path("data/User Listening History.csv")

    # Processed data
    cleaned_data_path: Path = Path("data/cleaned_data.csv")
    collab_data_path: Path = Path("data/collab_filtered_data.csv")
    track_ids_path: Path = Path("data/track_ids.npy")

    # Model artifacts
    content_transformer_path: Path = Path("models/content_transformer.joblib")
    content_faiss_path: Path = Path("models/content.faiss")
    hybrid_content_faiss_path: Path = Path("models/hybrid_content.faiss")
    nmf_model_path: Path = Path("models/nmf_model.joblib")
    collab_faiss_path: Path = Path("models/collab.faiss")

    # NMF hyperparameters
    nmf_components: int = 50
    nmf_max_iter: int = 300

    model_config = {"env_prefix": "APP_"}


settings = Settings()
