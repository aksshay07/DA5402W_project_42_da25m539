import os
import sys
from pathlib import Path

if sys.platform.startswith("win"):
    java_home = os.getenv("JAVA_HOME")
    if not java_home or not Path(java_home).exists():
        adoptium = Path(r"C:\Program Files\Eclipse Adoptium")
        if adoptium.exists():
            jdks = list(adoptium.glob("jdk*"))
            if jdks:
                os.environ["JAVA_HOME"] = str(jdks[0])
    hadoop_dir = Path("hadoop").resolve()
    if (hadoop_dir / "bin" / "winutils.exe").exists():
        os.environ["HADOOP_HOME"] = str(hadoop_dir)
        os.environ["PATH"] = str(hadoop_dir / "bin") + os.pathsep + os.environ.get("PATH", "")

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, from_json, lower, regexp_replace, trim
from pyspark.sql.types import IntegerType, StringType, StructField, StructType

KAFKA_BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:29092")
TOPIC = "raw_comments"
OUTPUT_PATH = os.getenv("STREAMED_PATH", "data/streamed")
CHECKPOINT_PATH = "data/.checkpoint"

SCHEMA = StructType([
    StructField("text", StringType()),
    StructField("label", IntegerType()),
])

LABEL_MAP = {0: "sadness", 1: "joy", 2: "love", 3: "anger", 4: "fear", 5: "surprise"}


def create_spark_session():
    return (
        SparkSession.builder
        .appName("SentimentStreamConsumer")
        .config("spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.1")
        .getOrCreate()
    )


def clean_text(df):
    return (
        df
        .withColumn("text", lower(col("text")))
        .withColumn("text", regexp_replace(col("text"), r"http\S+", ""))
        .withColumn("text", regexp_replace(col("text"), r"[^a-z0-9\s]", ""))
        .withColumn("text", trim(col("text")))
        .filter(col("text") != "")
        .dropna(subset=["text", "label"])
    )


def main():
    spark = create_spark_session()
    spark.sparkContext.setLogLevel("WARN")

    raw_stream = (
        spark.readStream
        .format("kafka")
        .option("kafka.bootstrap.servers", KAFKA_BOOTSTRAP)
        .option("subscribe", TOPIC)
        .option("startingOffsets", "earliest")
        .option("failOnDataLoss", "false")
        .load()
    )

    parsed = (
        raw_stream
        .select(from_json(col("value").cast("string"), SCHEMA).alias("data"))
        .select("data.*")
    )

    cleaned = clean_text(parsed)

    query = (
        cleaned.writeStream
        .format("parquet")
        .option("path", OUTPUT_PATH)
        .option("checkpointLocation", CHECKPOINT_PATH)
        .trigger(processingTime="30 seconds")
        .start()
    )

    query.awaitTermination(timeout=300)
    print(f"[consumer] Written processed data to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
