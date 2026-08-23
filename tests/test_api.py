"""Integration tests for the FastAPI serving endpoint."""
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    with patch("serving.predict.load_model"), \
         patch("serving.predict._model") as mock_model, \
         patch("serving.app.init_db"), \
         patch("serving.app.log_prediction"):

        mock_model.predict.return_value = [1]  # label_id 1 = "joy"

        from serving.app import app
        with TestClient(app) as c:
            yield c


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_predict_valid(client):
    resp = client.post("/predict", json={"text": "I feel amazing today!"})
    assert resp.status_code == 200
    data = resp.json()
    assert "label" in data
    assert "latency_ms" in data


def test_predict_empty_text(client):
    resp = client.post("/predict", json={"text": "   "})
    assert resp.status_code == 422
