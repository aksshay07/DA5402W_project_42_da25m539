"""
Model B: DistilBERT fine-tuned for 6-class emotion classification.
Logs all params, metrics, and the model artifact to MLflow.
Registers the model in the MLflow Model Registry.


"""
import os
import shutil
import tempfile

import mlflow
import mlflow.pyfunc
import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from transformers import (
    DataCollatorWithPadding,
    DistilBertForSequenceClassification,
    DistilBertTokenizerFast,
    Trainer,
    TrainingArguments,
)

PROCESSED_PATH = os.getenv("PROCESSED_PATH", "data/processed")
MODEL_OUT_PATH = os.getenv("MODEL_OUT_PATH", "models/distilbert")
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
EXPERIMENT_NAME = "sentiment-analysis"
MODEL_NAME = "sentiment-distilbert"
NUM_LABELS = 6


def resolve_training_device() -> str:
    """Resolve auto, cuda, mps, or cpu without changing the default behavior."""
    requested = os.getenv("TRAINING_DEVICE", "auto").strip().lower()
    valid_devices = {"auto", "cpu", "cuda", "mps"}
    if requested not in valid_devices:
        raise ValueError(
            f"TRAINING_DEVICE must be one of {sorted(valid_devices)}, got {requested!r}"
        )

    if requested == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("TRAINING_DEVICE=cuda requested, but CUDA is unavailable")
    if requested == "mps" and (
        not hasattr(torch.backends, "mps") or not torch.backends.mps.is_available()
    ):
        raise RuntimeError("TRAINING_DEVICE=mps requested, but MPS is unavailable")
    return requested


def load_data():
    import glob
    files = glob.glob(f"{PROCESSED_PATH}/**/*.parquet", recursive=True) or \
            glob.glob(f"{PROCESSED_PATH}/*.parquet")
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    return df["text"].tolist(), df["label"].tolist()


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "macro_f1": f1_score(labels, preds, average="macro"),
    }


class DistilBertEmotionModel(mlflow.pyfunc.PythonModel):
    """Bundles the tokenizer with the fine-tuned model so pyfunc serving
    accepts raw text and returns integer class ids, matching the contract
    of the sklearn pipeline logged by train_logreg.py."""

    def load_context(self, context):
        self.tokenizer = DistilBertTokenizerFast.from_pretrained(
            context.artifacts["tokenizer"]
        )
        self.model = DistilBertForSequenceClassification.from_pretrained(
            context.artifacts["model"]
        )
        self.model.eval()

    def predict(self, context, model_input, params=None):
        texts = model_input["text"].tolist() if hasattr(model_input, "columns") \
            else list(model_input)
        encodings = self.tokenizer(
            texts, truncation=True, padding=True, max_length=128, return_tensors="pt"
        )
        with torch.no_grad():
            logits = self.model(**encodings).logits
        return pd.Series(logits.argmax(dim=-1).tolist())


def main():
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    device = resolve_training_device()
    print(f"[distilbert] training device: {device}")

    texts, labels = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )

    params = {
        "epochs": 3,
        "batch_size": int(os.getenv("DISTILBERT_BATCH_SIZE", "32")),
        "lr": 2e-5,
        "max_len": 128,
        "device": device,
    }

    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=params["max_len"])

    train_ds = Dataset.from_dict({"text": X_train, "label": y_train}).map(
        tokenize, batched=True
    )
    test_ds = Dataset.from_dict({"text": X_test, "label": y_test}).map(
        tokenize, batched=True
    )
    # Dynamic padding (pad each batch to its own longest sequence, via the
    # collator below) instead of padding every example to max_len=128 up
    # front — noticeably lighter per batch on CPU.
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased", num_labels=NUM_LABELS
    )

    with tempfile.TemporaryDirectory() as run_tmpdir:
        training_args_kwargs = {
            "output_dir": run_tmpdir,
            "num_train_epochs": params["epochs"],
            "per_device_train_batch_size": params["batch_size"],
            "per_device_eval_batch_size": params["batch_size"],
            "learning_rate": params["lr"],
            "eval_strategy": "epoch",
            "save_strategy": "no",
            "logging_strategy": "epoch",
            # 0 = load/collate in the main process. Airflow's container has
            # limited CPU headroom shared with the rest of the stack, and
            # worker subprocesses add memory/thread overhead not worth it
            # for a dataset this size.
            "dataloader_num_workers": 0,
            "report_to": [],
            "disable_tqdm": True,
        }
        if device == "cpu":
            training_args_kwargs["no_cuda"] = True
        elif device == "mps":
            training_args_kwargs["use_mps_device"] = True

        training_args = TrainingArguments(
            **training_args_kwargs,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=test_ds,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
        )

        with mlflow.start_run(run_name="distilbert"):
            mlflow.log_params(params)

            trainer.train()
            for epoch_metrics in trainer.state.log_history:
                if "loss" in epoch_metrics and "epoch" in epoch_metrics:
                    mlflow.log_metric(
                        "train_loss", epoch_metrics["loss"],
                        step=int(epoch_metrics["epoch"]),
                    )

            eval_metrics = trainer.evaluate()
            acc = eval_metrics["eval_accuracy"]
            f1 = eval_metrics["eval_macro_f1"]
            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("macro_f1", f1)

            with tempfile.TemporaryDirectory() as artifact_tmpdir:
                model_dir = os.path.join(artifact_tmpdir, "hf_model")
                tokenizer_dir = os.path.join(artifact_tmpdir, "hf_tokenizer")
                model.save_pretrained(model_dir)
                tokenizer.save_pretrained(tokenizer_dir)

                artifacts = {"model": model_dir, "tokenizer": tokenizer_dir}

                mlflow.pyfunc.log_model(
                    artifact_path="model",
                    python_model=DistilBertEmotionModel(),
                    artifacts=artifacts,
                    registered_model_name=MODEL_NAME,
                )

                if os.path.exists(MODEL_OUT_PATH):
                    shutil.rmtree(MODEL_OUT_PATH)
                mlflow.pyfunc.save_model(
                    path=MODEL_OUT_PATH,
                    python_model=DistilBertEmotionModel(),
                    artifacts=artifacts,
                )

            print(f"[distilbert] accuracy={acc:.4f}  macro_f1={f1:.4f}")
            print(f"[distilbert] saved local copy to {MODEL_OUT_PATH}")


if __name__ == "__main__":
    main()