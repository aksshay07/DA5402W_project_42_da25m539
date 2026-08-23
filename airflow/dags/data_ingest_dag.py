from datetime import datetime, timezone

from airflow.operators.bash import BashOperator

from airflow import DAG

default_args = {"owner": "mlops", "retries": 1}

with DAG(
    dag_id="data_ingest_dag",
    default_args=default_args,
    description="Replay dair-ai/emotion dataset rows into Kafka topic raw_comments",
    schedule_interval=None,
    start_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
    catchup=False,
    tags=["ingest", "kafka"],
) as dag:

    produce_to_kafka = BashOperator(
        task_id="produce_to_kafka",
        bash_command="python /opt/airflow/kafka/producer.py",
        cwd="/opt/airflow",
    )
