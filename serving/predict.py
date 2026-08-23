import os

import mlflow.pyfunc

MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MODEL_NAME = os.getenv("SERVING_MODEL_NAME", "sentiment-distilbert")
MODEL_STAGE = os.getenv("SERVING_MODEL_STAGE", "Production")

LABEL_NAMES = ["sadness", "joy", "love", "anger", "fear", "surprise"]

_model = None


def load_model():
    global _model
    mlflow.set_tracking_uri(MLFLOW_URI)
    model_uri = f"models:/{MODEL_NAME}/{MODEL_STAGE}"
    _model = mlflow.pyfunc.load_model(model_uri)
    print(f"[predict] Loaded model: {model_uri}")


def predict(text: str) -> dict:
    if _model is None:
        load_model()
    import pandas as pd
    result = _model.predict(pd.DataFrame({"text": [text]}))
    label_idx = int(result[0])
    return {"label": LABEL_NAMES[label_idx], "label_id": label_idx}
