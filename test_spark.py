from pyspark.sql import SparkSession
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent

DATA_FILE = (
    BASE_DIR
    / "data"
    / "online_retail_II.parquet"
)


spark = (
    SparkSession.builder
    .appName("TestRealDataset")
    .master("local[*]")
    .getOrCreate()
)


spark.sparkContext.setLogLevel("WARN")


print("\nLoading Parquet dataset...")


df = spark.read.parquet(
    str(DATA_FILE)
)


print("\nDataset loaded successfully.")


print("\nTotal Rows:")
print(
    f"{df.count():,}"
)


print("\nTotal Columns:")
print(
    len(df.columns)
)


print("\nColumns:")
print(
    df.columns
)


print("\nSchema:")
df.printSchema()


print("\nFirst 5 Rows:")
df.show(
    5,
    truncate=False
)


spark.stop()