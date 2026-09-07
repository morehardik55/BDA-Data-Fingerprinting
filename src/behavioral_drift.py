"""Implementation 2: weekly behavioural drift intelligence with PySpark.

This pipeline intentionally keeps its output separate from Implementation 1.
Each weekly batch is compared only with preceding weekly fingerprints.
"""

from functools import reduce
import json
from pathlib import Path

from pyspark.sql import SparkSession, Window
from pyspark.sql import functions as F


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = BASE_DIR / "data" / "online_retail_II.parquet"
OUTPUT_FILE = BASE_DIR / "fingerprints" / "behavioral_drift.json"

# A week needs several prior observations before a statistical baseline is used.
MIN_HISTORY_WEEKS = 4
ZERO_VARIANCE_EPSILON = 1e-9

FEATURES = [
    "transaction_count",
    "unique_customer_count",
    "unique_product_count",
    "average_quantity",
    "quantity_stddev",
    "average_unit_price",
    "unit_price_stddev",
    "cancellation_rate",
    "missing_value_rate",
    "duplicate_rate",
]


def numeric_or_none(value):
    """Convert Spark numeric values to JSON-safe Python numbers."""
    return None if value is None else float(value)


def build_weekly_fingerprints(dataframe):
    """Create weekly aggregate features using Spark DataFrame operations."""
    source_columns = dataframe.columns
    weekly_data = (
        dataframe
        .filter(F.col("InvoiceDate").isNotNull())
        .withColumn("week_start", F.to_date(F.date_trunc("week", F.col("InvoiceDate"))))
    )

    missing_terms = [
        F.sum(F.when(F.col(column).isNull(), F.lit(1)).otherwise(F.lit(0)))
        for column in source_columns
    ]
    total_missing_expression = reduce(lambda left, right: left + right, missing_terms)

    weekly_features = (
        weekly_data
        .groupBy("week_start")
        .agg(
            F.count(F.lit(1)).alias("transaction_count"),
            F.countDistinct("CustomerID").alias("unique_customer_count"),
            F.countDistinct("StockCode").alias("unique_product_count"),
            F.avg(F.col("Quantity").cast("double")).alias("average_quantity"),
            F.stddev_samp(F.col("Quantity").cast("double")).alias("quantity_stddev"),
            F.avg(F.col("UnitPrice").cast("double")).alias("average_unit_price"),
            F.stddev_samp(F.col("UnitPrice").cast("double")).alias("unit_price_stddev"),
            (
                F.avg(
                    F.when(F.upper(F.col("InvoiceNo")).startswith("C"), F.lit(1.0))
                    .otherwise(F.lit(0.0))
                ) * F.lit(100.0)
            ).alias("cancellation_rate"),
            total_missing_expression.alias("total_missing_values"),
        )
        .withColumn(
            "missing_value_rate",
            F.when(
                F.col("transaction_count") > 0,
                F.col("total_missing_values") / (
                    F.col("transaction_count") * F.lit(len(source_columns))
                ) * F.lit(100.0),
            ).otherwise(F.lit(0.0)),
        )
    )

    # Count distinct complete rows per week in Spark, then derive duplicate rate.
    unique_rows = (
        weekly_data
        .dropDuplicates(["week_start", *source_columns])
        .groupBy("week_start")
        .agg(F.count(F.lit(1)).alias("unique_row_count"))
    )

    return (
        weekly_features
        .join(unique_rows, on="week_start", how="inner")
        .withColumn(
            "duplicate_rate",
            (F.col("transaction_count") - F.col("unique_row_count"))
            / F.col("transaction_count") * F.lit(100.0),
        )
        .drop("total_missing_values", "unique_row_count")
    )


def add_historical_baseline(weekly_features):
    """Use windows to calculate a prior-only mean, standard deviation, and z-score."""
    history_window = (
        Window.orderBy("week_start")
        .rowsBetween(Window.unboundedPreceding, -1)
    )

    result = weekly_features.withColumn(
        "historical_observations", F.count(F.lit(1)).over(history_window)
    )

    for feature in FEATURES:
        result = (
            result
            .withColumn(f"{feature}__baseline_mean", F.avg(feature).over(history_window))
            .withColumn(f"{feature}__baseline_stddev", F.stddev_samp(feature).over(history_window))
        )

        z_score = F.when(
            F.col("historical_observations") < MIN_HISTORY_WEEKS,
            F.lit(None).cast("double"),
        ).when(
            F.col(f"{feature}__baseline_stddev") > ZERO_VARIANCE_EPSILON,
            F.abs(
                (F.col(feature) - F.col(f"{feature}__baseline_mean"))
                / F.col(f"{feature}__baseline_stddev")
            ),
        ).when(
            F.abs(F.col(feature) - F.col(f"{feature}__baseline_mean")) <= ZERO_VARIANCE_EPSILON,
            F.lit(0.0),
        ).otherwise(F.lit(5.0))

        result = result.withColumn(f"{feature}__z_score", z_score)

    # A capped mean absolute z-score prevents one extreme value from obscuring
    # the rest of the behavioural profile. One z-score point costs 20 points.
    capped_distances = [
        F.least(F.coalesce(F.col(f"{feature}__z_score"), F.lit(0.0)), F.lit(5.0))
        for feature in FEATURES
    ]
    mean_distance = reduce(lambda left, right: left + right, capped_distances) / len(FEATURES)

    return (
        result
        .withColumn("behavioral_distance", mean_distance)
        .withColumn(
            "behavioral_similarity_score",
            F.round(F.greatest(F.lit(0.0), F.lit(100.0) - F.col("behavioral_distance") * 20.0), 2),
        )
        .withColumn(
            "drift_status",
            F.when(F.col("historical_observations") < MIN_HISTORY_WEEKS, F.lit("Normal"))
            .when(F.col("behavioral_similarity_score") >= 80.0, F.lit("Normal"))
            .when(F.col("behavioral_similarity_score") >= 55.0, F.lit("Moderate Drift"))
            .otherwise(F.lit("Major Drift")),
        )
    )


def serialise_fingerprints(scored_features):
    """Collect compact weekly results after all profiling and scoring in Spark."""
    records = []
    for row in scored_features.orderBy("week_start").collect():
        values = row.asDict(recursive=True)
        has_baseline = values["historical_observations"] >= MIN_HISTORY_WEEKS
        comparisons = {}

        for feature in FEATURES:
            baseline_mean = numeric_or_none(values[f"{feature}__baseline_mean"])
            baseline_stddev = numeric_or_none(values[f"{feature}__baseline_stddev"])
            z_score = numeric_or_none(values[f"{feature}__z_score"])
            comparisons[feature] = {
                "current": numeric_or_none(values[feature]),
                "baseline_mean": baseline_mean,
                "baseline_stddev": baseline_stddev,
                "z_score": z_score,
            }

        top_features = sorted(
            (
                {
                    "feature": feature,
                    "z_score": comparison["z_score"],
                    "current": comparison["current"],
                    "baseline_mean": comparison["baseline_mean"],
                }
                for feature, comparison in comparisons.items()
                if comparison["z_score"] is not None
            ),
            key=lambda item: item["z_score"],
            reverse=True,
        )[:3]

        records.append({
            "week": values["week_start"].isoformat(),
            "historical_observations": int(values["historical_observations"]),
            "baseline_ready": has_baseline,
            "behavioral_features": {
                feature: numeric_or_none(values[feature]) for feature in FEATURES
            },
            "feature_comparison": comparisons,
            "behavioral_similarity_score": numeric_or_none(values["behavioral_similarity_score"]),
            "drift_status": values["drift_status"],
            "top_deviating_features": top_features,
        })
    return records


def main():
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    spark = (
        SparkSession.builder
        .appName("BehavioralDriftIntelligence")
        .master("local[*]")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    try:
        print("Loading real UCI Online Retail II dataset...")
        source_data = spark.read.parquet(str(DATA_FILE))
        weekly_features = build_weekly_fingerprints(source_data)
        scored_features = add_historical_baseline(weekly_features)
        fingerprints = serialise_fingerprints(scored_features)

        with open(OUTPUT_FILE, "w", encoding="utf-8") as file:
            json.dump(fingerprints, file, indent=4)

        print(f"Generated {len(fingerprints)} weekly behavioural fingerprints.")
        print(f"Saved to: {OUTPUT_FILE}")
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
