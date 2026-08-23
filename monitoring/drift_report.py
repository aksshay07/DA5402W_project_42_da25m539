"""
Evidently drift report: compares the distribution of recent predictions
against the training reference set to detect feature/label drift.
Writes a drift_result.json that retrain_dag reads to decide whether to retrain.
"""
import json
import os
import sqlite3

import pandas as pd
from evidently.metric_preset import DataDriftPreset
from evidently.report import Report

REFERENCE_PATH = os.getenv("PROCESSED_PATH", "data/processed")
DB_PATH = os.getenv("PREDICTION_LOG_DB", "monitoring/predictions.db")
RESULT_PATH = "monitoring/drift_result.json"
DRIFT_SHARE_THRESHOLD = 0.3


def load_reference():
    import glob
    files = glob.glob(f"{REFERENCE_PATH}/**/*.parquet", recursive=True) or \
            glob.glob(f"{REFERENCE_PATH}/*.parquet")
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    return df[["label"]].rename(columns={"label": "label_id"})


def load_current():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT label_id FROM predictions ORDER BY id DESC LIMIT 1000", conn)
    conn.close()
    return df


def main():
    reference = load_reference()
    current = load_current()

    if current.empty:
        print("[drift] No predictions logged yet — skipping drift check.")
        with open(RESULT_PATH, "w") as f:
            json.dump({"drift_detected": False, "reason": "no_predictions"}, f)
        return

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference, current_data=current)

    report_dict = report.as_dict()
    drift_share = report_dict["metrics"][0]["result"]["share_of_drifted_columns"]
    drift_detected = drift_share >= DRIFT_SHARE_THRESHOLD

    result = {
        "drift_detected": drift_detected,
        "drift_share": drift_share,
        "threshold": DRIFT_SHARE_THRESHOLD,
    }
    with open(RESULT_PATH, "w") as f:
        json.dump(result, f, indent=2)

    print(f"[drift] drift_share={drift_share:.3f}  drift_detected={drift_detected}")


if __name__ == "__main__":
    main()
