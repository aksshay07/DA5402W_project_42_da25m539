# X Posts Sentiment Analysis - MLOps Project

Team 42 project for DA5402W. We are building an end-to-end MLOps pipeline
that classifies text posts into emotions (like joy, anger, sadness, etc.)
using the dair-ai/emotion dataset with technologies - Kafka, Spark, MLflow, Airflow, FastAPI and Prometheus/Grafana

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

## Status

This project is just getting started. We have set up the basic repo
structure, dependencies, and a docker-compose skeleton for the services
we will need. More components will be added step by step as we build
out the pipeline.

## Team

- Aksshay P DA25M539
- Shayan Sarkar DA25M618
- Subramanian Ganesh DA25M626
- Akash N DA25M536

## Data & Model Storage

We use DVC to version our dataset and models. The actual files are stored
in a shared Google Drive folder (not in Git):
https://drive.google.com/drive/folders/1RiJTIe29SK3OjqTYNKjv1E4qJbCE5Cf9

To pull the data after cloning this repo, run `dvc pull`. You'll need to
be added as a test user on our Google OAuth app first - ping Aksshay for
access and for the client ID/secret (not committed here for security
reasons).

## Setup and Installation
