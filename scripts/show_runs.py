"""Show all MLflow runs in the terminal."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import mlflow

mlflow.set_tracking_uri("sqlite:///mlflow.db")
client = mlflow.tracking.MlflowClient()

exp = client.get_experiment_by_name("spotify-hybrid-recommender")
if exp is None:
    print("No experiment found.")
    sys.exit(0)

runs = client.search_runs(exp.experiment_id, order_by=["start_time DESC"])
print(f"Experiment : {exp.name}")
print(f"Runs found : {len(runs)}")
print("-" * 55)

for run in runs:
    p = run.data.params
    m = run.data.metrics
    print(f"Run        : {run.info.run_name}")
    print(f"  nmf_components  = {p.get('nmf_components')}")
    print(f"  nmf_max_iter    = {p.get('nmf_max_iter')}")
    print(f"  NDCG@10 hybrid  = {m.get('eval_ndcg_at_10_hybrid', 0):.4f}")
    print(f"  NDCG@10 collab  = {m.get('eval_ndcg_at_10_collab', 0):.4f}")
    print(f"  NDCG@10 content = {m.get('eval_ndcg_at_10_content', 0):.4f}")
    print(f"  Precision@10    = {m.get('eval_precision_at_10_hybrid', 0):.4f}")
    print(f"  Recall@10       = {m.get('eval_recall_at_10_hybrid', 0):.4f}")
    print("-" * 55)
