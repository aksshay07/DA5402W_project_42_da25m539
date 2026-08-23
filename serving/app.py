import json
import os
import sqlite3
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from prometheus_client import Counter
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel

from serving.predict import load_model
from serving.predict import predict as run_predict

DB_PATH = os.getenv("PREDICTION_LOG_DB", "monitoring/predictions.db")
DRIFT_RESULT_PATH = os.getenv("DRIFT_RESULT_PATH", "monitoring/drift_result.json")
DRIFT_HISTORY_PATH = os.getenv("DRIFT_HISTORY_PATH", "monitoring/drift_history.jsonl")

LABEL_NAMES = ["sadness", "joy", "love", "anger", "fear", "surprise"]

# Per-class prediction distribution to check model drift
PREDICTIONS_BY_LABEL = Counter(
    "predictions_by_label_total",
    "Count of predictions grouped by predicted emotion label",
    ["label"],
)


def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT,
            text_hash TEXT,
            label TEXT,
            label_id INTEGER,
            latency_ms REAL
        )
    """)
    conn.commit()
    conn.close()


def log_prediction(text: str, label: str, label_id: int, latency_ms: float):
    import hashlib
    text_hash = hashlib.sha256(text.encode()).hexdigest()[:16]
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT INTO predictions (ts, text_hash, label, label_id, latency_ms) VALUES (?,?,?,?,?)",
        (datetime.now(timezone.utc).isoformat(), text_hash, label, label_id, latency_ms),
    )
    conn.commit()
    conn.close()


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    load_model()
    yield


app = FastAPI(title="Sentiment Analysis API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

Instrumentator().instrument(app).expose(app)


class PredictRequest(BaseModel):
    text: str


class PredictResponse(BaseModel):
    label: str
    label_id: int
    latency_ms: float


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if not req.text.strip():
        raise HTTPException(status_code=422, detail="text must not be empty")

    start = time.perf_counter()
    result = run_predict(req.text)
    latency_ms = (time.perf_counter() - start) * 1000

    log_prediction(req.text, result["label"], result["label_id"], latency_ms)
    PREDICTIONS_BY_LABEL.labels(label=result["label"]).inc()
    logger.info(f"predict | label={result['label']} latency={latency_ms:.1f}ms")

    return PredictResponse(latency_ms=round(latency_ms, 2), **result)


@app.get("/drift-status")
def drift_status():
    """Return the latest drift detection result and recent prediction distribution."""
    # Latest drift result
    drift_result = None
    if os.path.exists(DRIFT_RESULT_PATH):
        with open(DRIFT_RESULT_PATH) as f:
            drift_result = json.load(f)

    # Live prediction distribution from DB (last 200)
    live_distribution = {}
    total_predictions = 0
    try:
        conn = sqlite3.connect(DB_PATH)
        rows = conn.execute(
            "SELECT label_id, COUNT(*) as cnt FROM "
            "(SELECT label_id FROM predictions ORDER BY id DESC LIMIT 200) "
            "GROUP BY label_id"
        ).fetchall()
        conn.close()
        total = sum(r[1] for r in rows)
        total_predictions = total
        for label_id, cnt in rows:
            if 0 <= label_id < len(LABEL_NAMES):
                live_distribution[LABEL_NAMES[label_id]] = round(cnt / max(total, 1), 4)
    except Exception as e:  # noqa: BLE001 - predictions DB may not exist yet; degrade gracefully
        logger.warning(f"Could not read live prediction distribution: {e}")

    return {
        "drift_result": drift_result,
        "live_distribution": live_distribution,
        "total_predictions": total_predictions,
    }


@app.get("/drift-history")
def drift_history(limit: int = 50):
    """Return the last N drift check entries for time-series charting."""
    entries = []
    if os.path.exists(DRIFT_HISTORY_PATH):
        with open(DRIFT_HISTORY_PATH) as f:
            for line in f:
                line = line.strip()
                if line:
                    entries.append(json.loads(line))
    return entries[-limit:]


@app.get("/prediction-counts")
def prediction_counts():
    """Return total prediction count and per-label counts."""
    try:
        conn = sqlite3.connect(DB_PATH)
        rows = conn.execute(
            "SELECT label, COUNT(*) as cnt FROM predictions GROUP BY label"
        ).fetchall()
        total = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
        conn.close()
        return {
            "total": total,
            "by_label": {r[0]: r[1] for r in rows},
        }
    except Exception as e:  # noqa: BLE001 - predictions DB may not exist yet; degrade gracefully
        logger.warning(f"Could not read prediction counts: {e}")
        return {"total": 0, "by_label": {}}
