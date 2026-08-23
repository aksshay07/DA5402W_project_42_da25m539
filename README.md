# X Posts Sentiment Analysis - MLOps Project

Team 42 project for DA5402W. We are building an end-to-end MLOps pipeline
that classifies text posts into emotions (like joy, anger, sadness, etc.)
using the dair-ai/emotion dataset with technologies - Kafka, Spark, MLflow, Airflow, FastAPI and Prometheus/Grafana

## Team

- Aksshay P DA25M539
- Shayan Sarkar DA25M618
- Subramanian Ganesh DA25M626
- Akash N DA25M536

## What we are planning to use

- **Kafka** - to simulate streaming posts
- **Spark** - for preprocessing the data
- **MLflow** - to track our model experiments
- **DVC** - to version our data and models
- **Airflow** - to automate and schedule the pipeline
- **FastAPI** - to serve the trained model as an API
- **Docker** - to containerize everything
- **Prometheus + Grafana** - for monitoring the API
- **GitHub Actions** - for CI/CD

## Data & Model Storage

We use DVC to version our dataset and models. The actual files are stored
in a shared Google Drive folder (not in Git):
https://drive.google.com/drive/folders/1RiJTIe29SK3OjqTYNKjv1E4qJbCE5Cf9

To pull the data after cloning this repo, run `dvc pull`. You'll need to
be added as a test user on our Google OAuth app first - ping Aksshay for
access and for the client ID/secret (not committed here for security
reasons).

## Setup and Installation

This section walks through everything from an empty folder to a fully
running pipeline. It assumes you have Python 3.12, Docker Desktop, and
git already installed on your machine.

### 1. Clone the repo

```bash
git clone https://github.com/aksshay07/DA5402W_project_42_da25m539.git
cd DA5402W_project_42_da25m539
```

### 2. Set up a Python virtual environment

We use a virtual environment so this project's Python packages don't mix
with other projects on your machine.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

You'll need to run `source .venv/bin/activate` again every time you open
a new terminal window to work on this project.

### 3. Get the data

The dataset (dair-ai/emotion, a set of short text posts each labeled with
an emotion) isn't stored directly in Git - it's tracked with a tool called
DVC and the actual files live in a shared Google Drive folder. This keeps
the Git repo small and lets everyone stay in sync on the same data.

First, download the raw dataset:

```bash
python scripts/download_data.py
python scripts/check_data.py
```

If you'd rather pull the data that's already tracked by DVC (recommended,
since it saves you from re-downloading):

```bash
dvc pull
```

To use `dvc pull`/`dvc push`, you'll need access to our shared Google
Drive folder and a couple of OAuth credentials - ping Aksshay for these,
since they aren't committed to the repo for security reasons. Once you
have them:

```bash
dvc remote modify gdrive-storage gdrive_client_id '<CLIENT_ID>' --local
dvc remote modify gdrive-storage gdrive_client_secret '<CLIENT_SECRET>' --local
```

The first time you run `dvc pull` or `dvc push`, a browser window will
pop up asking you to log in with Google - just approve it.

### 4. Preprocess the data with Spark

This step cleans up the raw text (lowercasing, removing links/junk
characters, etc.) using Apache Spark, and saves the cleaned data to
`data/processed`.

```bash
dvc repro preprocess
```

We use `dvc repro` instead of running the script directly, because it
lets DVC keep track of exactly which version of the code produced which
version of the data - so if `data/raw` or `processing/preprocess.py`
changes later, DVC knows this step needs to be re-run.

### 5. Train the models

We train two different models on the cleaned data and log everything
(metrics, parameters, the trained model itself) to MLflow so we can
compare them later.

First, start MLflow so it's there to receive the logs. You can either run
it directly on your machine:

```bash
mlflow ui --port 5001
```

or run it inside Docker (closer to how it runs in production for this
project):

```bash
docker compose up -d mlflow
```

Then train both models:

```bash
dvc repro train_logreg        # a simple TF-IDF + Logistic Regression model
dvc repro train_distilbert    # a fine-tuned DistilBERT transformer model
```

DistilBERT takes a lot longer to train than Logistic Regression (tens of
minutes on a normal laptop CPU), since it's a much bigger model.

Once both are done, open the MLflow UI at `http://localhost:5001` (or
`5000` if you ran it directly) in your browser to compare their accuracy
and F1 scores.

### 6. Promote the best model to "Production"

MLflow keeps every model version you've ever trained, but the serving API
only loads whichever version is marked as "Production." This is a manual
step - someone looks at the metrics in the MLflow UI and decides which
version is actually good enough to serve.

You can do this from the MLflow UI (Models tab, pick a version, click
"Transition to Production"), or via the API:

```bash
curl -X POST "http://localhost:5001/api/2.0/mlflow/model-versions/transition-stage" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "sentiment-distilbert",
    "version": "1",
    "stage": "Production",
    "archive_existing_versions": true
  }'
```

### 7. Run the API

Once a model is promoted, you can start the FastAPI app that serves
predictions:

```bash
export MLFLOW_TRACKING_URI=http://localhost:5001
uvicorn serving.app:app --host 0.0.0.0 --port 8000
```

Try it out:

```bash
curl http://localhost:8000/health

curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "I am so incredibly happy today, everything is going great!"}'
```

You can also open `http://localhost:8000/docs` in a browser for an
interactive page where you can try the API without typing curl commands.

### 8. Bring up the full pipeline with Docker

Instead of running each piece by hand, you can bring up everything at
once with Docker Compose - this starts Kafka, MLflow, the API, Airflow,
Prometheus, and Grafana all together, wired up to talk to each other:

```bash
docker compose up -d
docker compose ps   # check that everything started correctly
```

To stop everything:

```bash
docker compose down
```

### 9. Kafka streaming (simulating live data)

To try out the streaming part of the pipeline (replaying the dataset as
if it were arriving in real time):

```bash
docker compose up -d zookeeper kafka

python kafka/producer.py         # sends rows into the Kafka topic
python kafka/consumer_spark.py   # reads them back out and cleans them with Spark
```

### 10. Airflow (automating everything)

Airflow lets us schedule and automate the steps above instead of running
them by hand. The first time you set it up:

```bash
docker compose up -d postgres
docker compose up airflow-init     # one-time setup: creates the database and an admin user
docker compose up -d airflow-webserver airflow-scheduler
```

Then open `http://localhost:8080` in a browser and log in with username
`admin` and password `admin`. You'll see three DAGs (Airflow's name for
an automated pipeline):

- **data_ingest_dag** - replays the dataset into Kafka
- **train_pipeline_dag** - preprocesses the data and trains both models
- **retrain_dag** - checks for data drift and retrains if needed

DAGs start out paused - toggle one on, then click the play button to
trigger it manually.

### 11. Monitoring - Prometheus and Grafana

Once the API is up and Prometheus/Grafana are running, you can watch
live metrics (how many predictions were made, how fast, error rates,
etc.):

```bash
docker compose up -d mlflow api prometheus grafana
```

- Prometheus: `http://localhost:9090`
- Grafana: `http://localhost:3000` (username `admin`, password `admin`) -
  open Dashboards -> Sentiment API to see the pre-built dashboard.

### 12. Drift detection

This checks whether the data the API is currently seeing looks
meaningfully different from the data the models were trained on - a
sign that the models might need retraining.

```bash
python monitoring/drift_report.py
```

### 13. Running tests and checks

Before pushing code, you can run the same checks our CI pipeline runs
automatically on GitHub:

```bash
ruff check .          # lint - catches style issues and simple bugs
pytest tests/ -v       # runs our automated tests
```

These also run automatically on every push and pull request via GitHub
Actions (see `.github/workflows/ci.yml`).
