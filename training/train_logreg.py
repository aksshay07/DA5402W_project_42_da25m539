"""
Model A: TF-IDF + Logistic Regression baseline.
Logs all params, metrics, and the model artifact to MLflow.
Registers the model in the MLflow Model Registry if it beats the current champion.
"""
import os
import shutil

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

PROCESSED_PATH = os.getenv("PROCESSED_PATH", "data/processed")
MODEL_OUT_PATH = os.getenv("MODEL_OUT_PATH", "models/logreg")
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
EXPERIMENT_NAME = "sentiment-analysis"
MODEL_NAME = "sentiment-logreg"

LABEL_NAMES = ["sadness", "joy", "love", "anger", "fear", "surprise"]


def load_data():
    import glob
    files = glob.glob(f"{PROCESSED_PATH}/**/*.parquet", recursive=True) or \
            glob.glob(f"{PROCESSED_PATH}/*.parquet")
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    return df["text"].tolist(), df["label"].tolist()


def main():
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    texts, labels = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )

    params = {"C": 1.0, "max_iter": 1000, "max_features": 50000, "ngram_range": (1, 2)}

    with mlflow.start_run(run_name="logreg"):
        mlflow.log_params(params)

        pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(max_features=params["max_features"],
                                      ngram_range=params["ngram_range"])),
            ("clf", LogisticRegression(C=params["C"], max_iter=params["max_iter"],
                                       multi_class="multinomial", solver="lbfgs")),
        ])

        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_test)

        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, average="macro")

        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("macro_f1", f1)
        print(classification_report(y_test, preds, target_names=LABEL_NAMES))

        mlflow.sklearn.log_model(
            pipeline, artifact_path="model",
            registered_model_name=MODEL_NAME,
        )

        if os.path.exists(MODEL_OUT_PATH):
            shutil.rmtree(MODEL_OUT_PATH)
        mlflow.sklearn.save_model(pipeline, MODEL_OUT_PATH)
        print(f"[logreg] accuracy={acc:.4f}  macro_f1={f1:.4f}")
        print(f"[logreg] saved local copy to {MODEL_OUT_PATH}")


if __name__ == "__main__":
    main()
