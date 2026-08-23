from datetime import datetime, timezone

from airflow.operators.bash import BashOperator

from airflow import DAG

default_args = {"owner": "mlops", "retries": 1}

with DAG(
    dag_id="train_pipeline_dag",
    default_args=default_args,
    description="Preprocess data with Spark, train LogReg and DistilBERT, track with MLflow",
    schedule_interval=None,
    start_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
    catchup=False,
    tags=["training", "mlflow", "spark"],
) as dag:

    spark_preprocess = BashOperator(
        task_id="spark_preprocess",
        bash_command="python /opt/airflow/processing/preprocess.py",
        cwd="/opt/airflow",
    )

    train_logreg = BashOperator(
        task_id="train_logreg",
        bash_command="python /opt/airflow/training/train_logreg.py",
        cwd="/opt/airflow",
    )

    train_distilbert = BashOperator(
        task_id="train_distilbert",
        bash_command="python /opt/airflow/training/train_distilbert.py",
        cwd="/opt/airflow",
    )

    spark_preprocess >> train_logreg >> train_distilbert
