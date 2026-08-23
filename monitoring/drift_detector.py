"""
Enhanced drift detector: compares the label distribution of recent
predictions against the training reference set.

Computes:
  - Population Stability Index (PSI)
  - Per-label distribution comparison
  - Evidently DataDrift report

Appends each check to monitoring/drift_history.jsonl so the dashboard
can render time-series drift trends.
Writes the latest result to monitoring/drift_result.json for the
retrain_dag to consume.
"""
import json
import os
import sqlite3
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from evidently.metric_preset import DataDriftPreset
from evidently.report import Report

REFERENCE_PATH = os.getenv("PROCESSED_PATH", "data/processed")
DB_PATH = os.getenv("PREDICTION_LOG_DB", "monitoring/predictions.db")
RESULT_PATH = os.getenv("DRIFT_RESULT_PATH", "monitoring/drift_result.json")

HISTORY_PATH = os.getenv(
    "DRIFT_HISTORY_PATH",
    "/opt/airflow/monitoring/drift_history.jsonl"
)

PSI_THRESHOLD = float(os.getenv("PSI_THRESHOLD", "0.2"))
WINDOW_SIZE = int(os.getenv("DRIFT_WINDOW_SIZE", "200"))

LABEL_NAMES = ["sadness", "joy", "love", "anger", "fear", "surprise"]
NUM_LABELS = len(LABEL_NAMES)


def compute_psi(reference_dist: np.ndarray, current_dist: np.ndarray,
                epsilon: float = 1e-6) -> float:
    ref = np.clip(reference_dist, epsilon, None)
    cur = np.clip(current_dist, epsilon, None)
    ref = ref / ref.sum()
    cur = cur / cur.sum()
    return float(np.sum((cur - ref) * np.log(cur / ref)))


def load_reference_distribution() -> np.ndarray:
    import glob
    files = glob.glob(f"{REFERENCE_PATH}/**/*.parquet", recursive=True) or \
            glob.glob(f"{REFERENCE_PATH}/*.parquet")
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    counts = df["label"].value_counts().sort_index()
    dist = np.zeros(NUM_LABELS)
    for label_id, count in counts.items():
        if 0 <= label_id < NUM_LABELS:
            dist[label_id] = count
    return dist / dist.sum()


def load_current_distribution(window: int = WINDOW_SIZE):
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql(
        f"SELECT label_id FROM predictions ORDER BY id DESC LIMIT {window}",
        conn,
    )
    conn.close()
    if df.empty:
        return None, 0
    counts = df["label_id"].value_counts().sort_index()
    dist = np.zeros(NUM_LABELS)
    for label_id, count in counts.items():
        if 0 <= label_id < NUM_LABELS:
            dist[label_id] = count
    total = int(dist.sum())
    return dist / max(dist.sum(), 1), total


def run_evidently_drift(reference_path: str, db_path: str, window: int):
    import glob
    files = glob.glob(f"{reference_path}/**/*.parquet", recursive=True) or \
            glob.glob(f"{reference_path}/*.parquet")
    ref_df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    ref_df = ref_df[["label"]].rename(columns={"label": "label_id"})

    conn = sqlite3.connect(db_path)
    cur_df = pd.read_sql(
        f"SELECT label_id FROM predictions ORDER BY id DESC LIMIT {window}",
        conn,
    )
    conn.close()
    if cur_df.empty:
        return None

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=ref_df, current_data=cur_df)
    result = report.as_dict()
    return result["metrics"][0]["result"]["share_of_drifted_columns"]


def main():
    now = datetime.now(timezone.utc).isoformat()

    cur_dist, sample_count = load_current_distribution(WINDOW_SIZE)

    if cur_dist is None or sample_count == 0:
        result = {
            "timestamp": now,
            "drift_detected": False,
            "reason": "no_predictions",
            "psi": 0.0,
            "sample_count": 0,
        }
        _write_result(result)
        print("[drift] No predictions logged yet -- skipping.")
        return

    ref_dist = load_reference_distribution()
    psi = compute_psi(ref_dist, cur_dist)

    evidently_share = None
    try:
        evidently_share = run_evidently_drift(REFERENCE_PATH, DB_PATH, WINDOW_SIZE)
    except Exception as e:
        print(f"[drift] Evidently check failed (non-fatal): {e}")

    drift_detected = psi >= PSI_THRESHOLD

    ref_dict = {LABEL_NAMES[i]: round(float(ref_dist[i]), 4) for i in range(NUM_LABELS)}
    cur_dict = {LABEL_NAMES[i]: round(float(cur_dist[i]), 4) for i in range(NUM_LABELS)}

    result = {
        "timestamp": now,
        "drift_detected": drift_detected,
        "psi": round(psi, 4),
        "psi_threshold": PSI_THRESHOLD,
        "evidently_drift_share": evidently_share,
        "sample_count": sample_count,
        "reference_distribution": ref_dict,
        "current_distribution": cur_dict,
    }

    _write_result(result)
    _append_history(result)

    status = "DRIFT DETECTED" if drift_detected else "No drift"
    print(f"[drift] {status}  PSI={psi:.4f} (threshold={PSI_THRESHOLD})  "
          f"samples={sample_count}")


def _write_result(result: dict):
    os.makedirs(os.path.dirname(RESULT_PATH) or ".", exist_ok=True)
    with open(RESULT_PATH, "w") as f:
        json.dump(result, f, indent=2)


def _append_history(result: dict):
    os.makedirs(os.path.dirname(HISTORY_PATH) or ".", exist_ok=True)
    with open(HISTORY_PATH, "a") as f:
        f.write(json.dumps(result) + "\n")


if __name__ == "__main__":
    main()
