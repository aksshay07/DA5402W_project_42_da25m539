import json
import os
import time
from pathlib import Path

import pandas as pd
from confluent_kafka import Producer

KAFKA_BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:29092")
TOPIC = "raw_comments"
DATA_PATH = os.getenv("DATA_PATH", "data/raw")
DELAY_SECONDS = float(os.getenv("PRODUCER_DELAY", "0.05"))


def delivery_report(err, msg):
    if err:
        print(f"[producer] delivery failed: {err}")


def main():
    producer = Producer({"bootstrap.servers": KAFKA_BOOTSTRAP})
    parquet_files = list(Path(DATA_PATH).glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {DATA_PATH}")

    df = pd.concat([pd.read_parquet(f) for f in parquet_files], ignore_index=True)
    print(f"[producer] Loaded {len(df)} rows. Publishing to topic '{TOPIC}'...")

    for _, row in df.iterrows():
        payload = json.dumps({"text": row["text"], "label": int(row["label"])})
        producer.produce(TOPIC, value=payload.encode("utf-8"), callback=delivery_report)
        producer.poll(0)
        time.sleep(DELAY_SECONDS)

    producer.flush()
    print("[producer] Done.")


if __name__ == "__main__":
    main()
