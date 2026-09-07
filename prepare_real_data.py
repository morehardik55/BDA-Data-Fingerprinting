"""Prepare the genuine 2022 Amazon purchases data with PySpark."""

from pathlib import Path
import shutil

import pyarrow.parquet as pq
from py4j.protocol import Py4JJavaError
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DateType, DoubleType, StringType, StructField, StructType


BASE_DIR = Path(__file__).resolve().parent
INPUT_FILE = BASE_DIR / "data" / "amazon-purchases.csv"
OUTPUT_FILE = BASE_DIR / "data" / "amazon_purchases_2022.parquet"

SOURCE_COLUMNS = [
    "Order Date", "Purchase Price Per Unit", "Quantity", "Shipping Address State",
    "Title", "ASIN/ISBN (Product Code)", "Category", "Survey ResponseID",
]

SOURCE_SCHEMA = StructType([
    StructField("Order Date", DateType(), True),
    StructField("Purchase Price Per Unit", DoubleType(), True),
    StructField("Quantity", DoubleType(), True),
    StructField("Shipping Address State", StringType(), True),
    StructField("Title", StringType(), True),
    StructField("ASIN/ISBN (Product Code)", StringType(), True),
    StructField("Category", StringType(), True),
    StructField("Survey ResponseID", StringType(), True),
])


def main():
    spark = (
        SparkSession.builder.appName("PrepareAmazonPurchases2022")
        .master("local[*]").config("spark.ui.enabled", "false")
        .config("spark.hadoop.fs.permissions.enabled", "false").getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    try:
        print("Loading Open e-commerce 1.0 Amazon purchases CSV with PySpark...")
        source_df = (
            spark.read.option("header", True).option("dateFormat", "yyyy-MM-dd")
            .option("quote", '"').option("escape", '"').option("multiLine", True)
            .schema(SOURCE_SCHEMA).csv(str(INPUT_FILE))
        )

        if source_df.columns != SOURCE_COLUMNS:
            raise ValueError(f"Unexpected Amazon CSV schema: {source_df.columns}")

        prepared_df = source_df
        for column in [
            "Shipping Address State", "Title", "ASIN/ISBN (Product Code)",
            "Category", "Survey ResponseID",
        ]:
            prepared_df = prepared_df.withColumn(
                column,
                F.when(F.trim(F.col(column)) == "", F.lit(None)).otherwise(F.col(column)),
            )

        invalid_dates = prepared_df.filter(F.col("Order Date").isNull()).count()
        if invalid_dates:
            raise ValueError(f"Found {invalid_dates} null or unparseable Order Date values.")

        amazon_2022_df = prepared_df.filter(F.year(F.col("Order Date")) == 2022)
        validation = amazon_2022_df.agg(
            F.count(F.lit(1)).alias("records"),
            F.min("Order Date").alias("earliest_date"),
            F.max("Order Date").alias("latest_date"),
        ).first()
        if validation["records"] == 0:
            raise ValueError("No genuine 2022 Amazon purchase records were found.")

        print(f"Genuine 2022 records: {validation['records']:,}")
        print(f"2022 source range: {validation['earliest_date']} to {validation['latest_date']}")
        try:
            amazon_2022_df.write.mode("overwrite").parquet(str(OUTPUT_FILE))
        except Py4JJavaError:
            # Spark 4 on this Windows host cannot create local output folders
            # without winutils.exe. The Spark DataFrame remains the ingestion,
            # validation, and filtering engine; Arrow is only a local Parquet
            # writer fallback for this environment.
            if OUTPUT_FILE.exists():
                if OUTPUT_FILE.is_dir():
                    shutil.rmtree(OUTPUT_FILE)
                else:
                    OUTPUT_FILE.unlink()
            pq.write_table(amazon_2022_df.toArrow(), str(OUTPUT_FILE))
            print("Used Arrow Parquet writer fallback because local Windows Spark lacks winutils.exe.")
        print(f"Prepared Parquet saved to: {OUTPUT_FILE}")
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
