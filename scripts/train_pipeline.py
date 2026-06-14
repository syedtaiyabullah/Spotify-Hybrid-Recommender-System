"""
Runs the full training pipeline or individual stages.

Usage:
    python scripts/train_pipeline.py          # all stages
    python scripts/train_pipeline.py clean    # data cleaning only
    python scripts/train_pipeline.py collab   # NMF + collab FAISS only
    python scripts/train_pipeline.py content  # content transformer + FAISS indexes

Stage order for a full run: clean -> collab -> content
(content must run after collab so the hybrid FAISS index can be built)

MLflow
------
The collab stage is wrapped in an MLflow run that logs:
  Parameters : nmf_components, nmf_max_iter, nmf_init
  Metrics    : training_duration_seconds, n_tracks, n_users, collab_songs,
               nmf_reconstruction_err, nmf_n_iter,
               eval_ndcg@10_hybrid, eval_ndcg@10_collab, eval_ndcg@10_content,
               eval_precision@10_hybrid, eval_recall@10_hybrid
  Artifact   : models/nmf_model.joblib

View results:
    mlflow ui
    then open http://127.0.0.1:5000
"""
import sys
import time
from pathlib import Path

# Allow running as a plain script from the project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.config import settings
from backend.pipeline import data_cleaning, feature_engineering, interaction_matrix


# -------------------------------------------------------------------------
# Stage runners
# -------------------------------------------------------------------------

def run_clean() -> None:
    print("=== Stage 1: Data Cleaning ===")
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    data_cleaning.run(settings.music_info_path, settings.cleaned_data_path)


def run_collab() -> None:
    """
    Train NMF + build collab FAISS index, wrapped in an MLflow run.
    Falls back to plain training if MLflow is not installed.
    """
    print("=== Stage 2: Collaborative Filtering (NMF + FAISS) ===")
    settings.models_dir.mkdir(parents=True, exist_ok=True)

    try:
        import mlflow
        _run_collab_with_mlflow(mlflow)
    except ImportError:
        print("  [INFO] mlflow not installed — running without experiment tracking.")
        print("  Install with: pip install mlflow")
        _run_collab_plain()


def run_content() -> None:
    print("=== Stage 3: Content Feature Engineering + FAISS ===")
    settings.models_dir.mkdir(parents=True, exist_ok=True)
    feature_engineering.run(
        cleaned_data_path=settings.cleaned_data_path,
        transformer_path=settings.content_transformer_path,
        content_faiss_path=settings.content_faiss_path,
        collab_data_path=settings.collab_data_path,
        hybrid_faiss_path=settings.hybrid_content_faiss_path,
    )


# -------------------------------------------------------------------------
# Collab helpers
# -------------------------------------------------------------------------

def _collab_kwargs() -> dict:
    return dict(
        user_history_path=settings.user_history_path,
        songs_data_path=settings.cleaned_data_path,
        track_ids_path=settings.track_ids_path,
        collab_data_path=settings.collab_data_path,
        nmf_path=settings.nmf_model_path,
        collab_faiss_path=settings.collab_faiss_path,
        n_components=settings.nmf_components,
        max_iter=settings.nmf_max_iter,
    )


def _run_collab_plain() -> None:
    interaction_matrix.run(**_collab_kwargs())


def _run_collab_with_mlflow(mlflow) -> None:
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("spotify-hybrid-recommender")

    run_name = f"nmf_c{settings.nmf_components}_i{settings.nmf_max_iter}"

    with mlflow.start_run(run_name=run_name):

        # -- Log hyperparameters ----------------------------------------------
        mlflow.log_params({
            "nmf_components": settings.nmf_components,
            "nmf_max_iter":   settings.nmf_max_iter,
            "nmf_init":       "nndsvda",
            "faiss_metric":   "IndexFlatIP (cosine)",
        })

        # -- Train ------------------------------------------------------------
        print(f"  MLflow run: {run_name}")
        t0    = time.perf_counter()
        stats = interaction_matrix.run(**_collab_kwargs())
        training_duration = round(time.perf_counter() - t0, 1)

        # -- Log training metrics ---------------------------------------------
        mlflow.log_metrics({
            "training_duration_seconds": training_duration,
            "n_tracks":                  stats["n_tracks"],
            "n_users":                   stats["n_users"],
            "collab_songs":              stats["collab_songs"],
            "nmf_reconstruction_err":    stats["nmf_reconstruction_err"],
            "nmf_n_iter":                stats["nmf_n_iter"],
        })
        print(f"  Training done in {training_duration}s  "
              f"(reconstruction_err={stats['nmf_reconstruction_err']:.4f}, "
              f"n_iter={stats['nmf_n_iter']})")

        # -- Run offline evaluation and log results ---------------------------
        print("\n  Running offline evaluation (n_users=100)...")
        try:
            from scripts.evaluate import compute_metrics
            eval_results = compute_metrics(n_users=100, k=10, verbose=False)

            mlflow.log_metrics({
                "eval_ndcg_at_10_content":    eval_results["content"]["ndcg"],
                "eval_ndcg_at_10_collab":     eval_results["collab"]["ndcg"],
                "eval_ndcg_at_10_hybrid":     eval_results["hybrid"]["ndcg"],
                "eval_precision_at_10_hybrid": eval_results["hybrid"]["precision"],
                "eval_recall_at_10_hybrid":    eval_results["hybrid"]["recall"],
            })

            print(f"  Evaluation done:")
            print(f"    Content   NDCG@10 = {eval_results['content']['ndcg']:.4f}")
            print(f"    Collab    NDCG@10 = {eval_results['collab']['ndcg']:.4f}")
            print(f"    Hybrid    NDCG@10 = {eval_results['hybrid']['ndcg']:.4f}")

        except Exception as exc:
            print(f"  [WARN] Evaluation failed, metrics not logged: {exc}")

        # -- Log model artifact -----------------------------------------------
        if settings.nmf_model_path.exists():
            mlflow.log_artifact(str(settings.nmf_model_path), artifact_path="models")
            print(f"  Artifact logged: {settings.nmf_model_path}")

        print(f"\n  MLflow run complete. View with: mlflow ui")


# -------------------------------------------------------------------------
# Entry point
# -------------------------------------------------------------------------

def main() -> None:
    stage = sys.argv[1] if len(sys.argv) > 1 else "all"
    valid = {"all", "clean", "collab", "content"}
    if stage not in valid:
        print(f"Unknown stage '{stage}'. Choose from: {valid}")
        sys.exit(1)

    if stage in ("all", "clean"):
        run_clean()
    if stage in ("all", "collab"):
        run_collab()
    if stage in ("all", "content"):
        run_content()

    print("\n=== Pipeline complete ===")


if __name__ == "__main__":
    main()
