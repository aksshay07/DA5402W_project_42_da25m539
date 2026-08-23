import json
import os
from datetime import datetime, timezone

from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator, ShortCircuitOperator

from airflow import DAG

default_args = {"owner": "mlops", "retries": 1}

DRIFT_REPORT_PATH = "/opt/airflow/monitoring/drift_result.json"
RETRAIN_HISTORY_PATH = "/opt/airflow/monitoring/retrain_history.jsonl"
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")


def drift_detected():
    if not os.path.exists(DRIFT_REPORT_PATH):
        return False
    with open(DRIFT_REPORT_PATH) as f:
        result = json.load(f)
    return result.get("drift_detected", False)


def compare_and_promote(**kwargs):
    """Compare newly trained DistilBERT model against current Production.
    Promote the new version if its macro_f1 is equal or better."""
    import mlflow
    from mlflow.tracking import MlflowClient

    mlflow.set_tracking_uri(MLFLOW_URI)
    client = MlflowClient(tracking_uri=MLFLOW_URI)
    model_name = "sentiment-distilbert"

    versions = client.search_model_versions(f"name='{model_name}'")
    if not versions:
        print("[promote] No model versions found -- skipping.")
        return

    prod_versions = [v for v in versions if v.current_stage == "Production"]
    latest_version = max(versions, key=lambda v: int(v.version))

    old_f1 = 0.0
    if prod_versions:
        prod_v = prod_versions[0]
        run = client.get_run(prod_v.run_id)
        old_f1 = run.data.metrics.get("macro_f1", 0.0)
        print(f"[promote] Current Production: v{prod_v.version}  F1={old_f1:.4f}")

    new_run = client.get_run(latest_version.run_id)
    new_f1 = new_run.data.metrics.get("macro_f1", 0.0)
    print(f"[promote] New candidate: v{latest_version.version}  F1={new_f1:.4f}")

    promoted = False
    if new_f1 >= old_f1:
        for pv in prod_versions:
            client.transition_model_version_stage(
                name=model_name, version=pv.version, stage="Archived",
            )
        client.transition_model_version_stage(
            name=model_name, version=latest_version.version, stage="Production",
        )
        print(f"[promote] Promoted v{latest_version.version} to Production")
        promoted = True
    else:
        print(f"[promote] New model not better ({new_f1:.4f} < {old_f1:.4f}). "
              "Keeping current Production.")

    log_entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "old_version": prod_versions[0].version if prod_versions else None,
        "old_f1": round(old_f1, 4),
        "new_version": latest_version.version,
        "new_f1": round(new_f1, 4),
        "promoted": promoted,
        "model_name": model_name,
    }
    os.makedirs(os.path.dirname(RETRAIN_HISTORY_PATH), exist_ok=True)
    with open(RETRAIN_HISTORY_PATH, "a") as f:
        f.write(json.dumps(log_entry) + "\n")


with DAG(
    dag_id="retrain_dag",
    default_args=default_args,
    description="Check drift report, retrain models if drift detected, compare and promote",
    schedule_interval="@daily",
    start_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
    catchup=False,
    tags=["retrain", "drift", "mlflow"],
) as dag:

    run_drift_report = BashOperator(
        task_id="run_drift_report",
        bash_command="python /opt/airflow/monitoring/drift_detector.py",
        cwd="/opt/airflow",
    )

    check_drift = ShortCircuitOperator(
        task_id="check_drift",
        python_callable=drift_detected,
    )

    retrain_logreg = BashOperator(
        task_id="retrain_logreg",
        bash_command="python /opt/airflow/training/train_logreg.py",
        cwd="/opt/airflow",
    )

    retrain_distilbert = BashOperator(
        task_id="retrain_distilbert",
        bash_command="python /opt/airflow/training/train_distilbert.py",
        cwd="/opt/airflow",
    )

    promote_best_model = PythonOperator(
        task_id="compare_and_promote",
        python_callable=compare_and_promote,
    )

    run_drift_report >> check_drift >> [retrain_logreg, retrain_distilbert]
    [retrain_logreg, retrain_distilbert] >> promote_best_model
