"""Implementation 2: Amazon 2022 behavioural drift intelligence with PySpark."""

from functools import reduce
import json
from pathlib import Path

from pyspark.sql import SparkSession, Window
from pyspark.sql import functions as F


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = BASE_DIR / "data" / "amazon_purchases_2022.parquet"
OUTPUT_FILE = BASE_DIR / "fingerprints" / "behavioral_drift.json"
MIN_HISTORY_WEEKS = 4
ZERO_VARIANCE_EPSILON = 1e-9

FEATURES = [
    "purchase_record_count", "survey_response_diversity", "unique_product_code_count",
    "unique_category_count", "unique_shipping_state_count", "average_quantity",
    "quantity_stddev", "average_purchase_price_per_unit", "purchase_price_per_unit_stddev",
    "missing_value_rate", "duplicate_rate",
]


def numeric_or_none(value):
    return None if value is None else float(value)


def source_window_metadata(weekly_df, first_date, last_date):
    first_week = F.to_date(F.date_trunc("week", F.lit(first_date)))
    last_week = F.to_date(F.date_trunc("week", F.lit(last_date)))
    is_first_partial = (F.col("week_start") == first_week) & (F.lit(first_date) > F.col("week_start"))
    is_last_partial = (F.col("week_start") == last_week) & (F.lit(last_date) < F.date_add(F.col("week_start"), 6))
    return (
        weekly_df.withColumn("is_partial_source_window", is_first_partial | is_last_partial)
        .withColumn(
            "source_window_note",
            F.when(
                is_first_partial & is_last_partial,
                F.lit("Partial source window: the 2022 source begins and ends before this calendar week is complete."),
            ).when(
                is_first_partial,
                F.concat(F.lit("Partial source window: genuine 2022 data begins on "), F.date_format(F.lit(first_date), "yyyy-MM-dd"), F.lit(" after this calendar week began.")),
            ).when(
                is_last_partial,
                F.concat(F.lit("Partial source window: genuine 2022 data ends on "), F.date_format(F.lit(last_date), "yyyy-MM-dd"), F.lit(" before this calendar week is complete.")),
            ).otherwise(F.lit(None).cast("string")),
        )
    )


def build_weekly_fingerprints(dataframe):
    source_columns = dataframe.columns
    weekly_data = dataframe.filter(F.col("Order Date").isNotNull()).withColumn(
        "week_start", F.to_date(F.date_trunc("week", F.col("Order Date")))
    )
    source_dates = weekly_data.agg(F.min("Order Date").alias("first"), F.max("Order Date").alias("last")).first()
    missing_terms = [F.sum(F.when(F.col(column).isNull(), 1).otherwise(0)) for column in source_columns]
    total_missing = reduce(lambda left, right: left + right, missing_terms)

    weekly_features = weekly_data.groupBy("week_start").agg(
        F.count(F.lit(1)).alias("purchase_record_count"),
        F.countDistinct("Survey ResponseID").alias("survey_response_diversity"),
        F.countDistinct("ASIN/ISBN (Product Code)").alias("unique_product_code_count"),
        F.countDistinct("Category").alias("unique_category_count"),
        F.countDistinct("Shipping Address State").alias("unique_shipping_state_count"),
        F.avg(F.col("Quantity").cast("double")).alias("average_quantity"),
        F.stddev_samp(F.col("Quantity").cast("double")).alias("quantity_stddev"),
        F.avg(F.col("Purchase Price Per Unit").cast("double")).alias("average_purchase_price_per_unit"),
        F.stddev_samp(F.col("Purchase Price Per Unit").cast("double")).alias("purchase_price_per_unit_stddev"),
        total_missing.alias("total_missing_values"),
    ).withColumn(
        "missing_value_rate",
        F.col("total_missing_values") / (F.col("purchase_record_count") * F.lit(len(source_columns))) * F.lit(100.0),
    )

    unique_rows = weekly_data.dropDuplicates(["week_start", *source_columns]).groupBy("week_start").agg(
        F.count(F.lit(1)).alias("unique_row_count")
    )
    return source_window_metadata(
        weekly_features.join(unique_rows, "week_start").withColumn(
            "duplicate_rate",
            (F.col("purchase_record_count") - F.col("unique_row_count")) / F.col("purchase_record_count") * F.lit(100.0),
        ).drop("total_missing_values", "unique_row_count"),
        source_dates["first"], source_dates["last"],
    )


def add_historical_baseline(weekly_features):
    history = Window.orderBy("week_start").rowsBetween(Window.unboundedPreceding, -1)
    result = weekly_features.withColumn("historical_observations", F.count(F.lit(1)).over(history))
    for feature in FEATURES:
        result = result.withColumn(f"{feature}__baseline_mean", F.avg(feature).over(history)).withColumn(
            f"{feature}__baseline_stddev", F.stddev_samp(feature).over(history)
        )
        z_score = F.when(F.col("historical_observations") < MIN_HISTORY_WEEKS, F.lit(None).cast("double")).when(
            F.col(f"{feature}__baseline_stddev").isNull(), F.lit(None).cast("double")
        ).when(
            F.col(f"{feature}__baseline_stddev") > ZERO_VARIANCE_EPSILON,
            F.abs((F.col(feature) - F.col(f"{feature}__baseline_mean")) / F.col(f"{feature}__baseline_stddev")),
        ).when(
            F.abs(F.col(feature) - F.col(f"{feature}__baseline_mean")) <= ZERO_VARIANCE_EPSILON,
            F.lit(0.0),
        ).otherwise(F.lit(5.0))
        if feature == "purchase_record_count":
            z_score = F.when(F.col("is_partial_source_window"), F.lit(None).cast("double")).otherwise(z_score)
        result = result.withColumn(f"{feature}__z_score", z_score)

    flags = [F.when(F.col(f"{feature}__z_score").isNotNull(), 1).otherwise(0) for feature in FEATURES]
    valid_count = reduce(lambda left, right: left + right, flags)
    capped = [
        F.when(F.col(f"{feature}__z_score").isNotNull(), F.least(F.col(f"{feature}__z_score"), F.lit(5.0))).otherwise(F.lit(0.0))
        for feature in FEATURES
    ]
    distance = reduce(lambda left, right: left + right, capped)
    return result.withColumn("similarity_scored_feature_count", valid_count).withColumn(
        "behavioral_distance", F.when(F.col("similarity_scored_feature_count") > 0, distance / F.col("similarity_scored_feature_count"))
    ).withColumn(
        "behavioral_similarity_score",
        F.when(F.col("historical_observations") < MIN_HISTORY_WEEKS, F.lit(None).cast("double")).when(
            F.col("behavioral_distance").isNull(), F.lit(None).cast("double")
        ).otherwise(F.round(F.greatest(F.lit(0.0), F.lit(100.0) - F.col("behavioral_distance") * 20.0), 2)),
    ).withColumn(
        "drift_status",
        F.when(F.col("historical_observations") < MIN_HISTORY_WEEKS, F.lit("Baseline Building")).when(
            F.col("behavioral_similarity_score").isNull(), F.lit("Insufficient History")
        ).when(F.col("behavioral_similarity_score") >= 80.0, F.lit("Normal")).when(
            F.col("behavioral_similarity_score") >= 55.0, F.lit("Moderate Drift")
        ).otherwise(F.lit("Major Drift")),
    )


def serialise_fingerprints(scored_features):
    fingerprints = []
    for row in scored_features.orderBy("week_start").collect():
        values = row.asDict(recursive=True)
        comparisons = {
            feature: {
                "current": numeric_or_none(values[feature]),
                "baseline_mean": numeric_or_none(values[f"{feature}__baseline_mean"]),
                "baseline_stddev": numeric_or_none(values[f"{feature}__baseline_stddev"]),
                "z_score": numeric_or_none(values[f"{feature}__z_score"]),
            }
            for feature in FEATURES
        }
        top_features = sorted(
            ({"feature": feature, "z_score": item["z_score"], "current": item["current"], "baseline_mean": item["baseline_mean"]}
             for feature, item in comparisons.items() if item["z_score"] is not None),
            key=lambda item: item["z_score"], reverse=True,
        )[:3]
        fingerprints.append({
            "week": values["week_start"].isoformat(),
            "is_partial_source_window": bool(values["is_partial_source_window"]),
            "source_window_note": values["source_window_note"],
            "historical_observations": int(values["historical_observations"]),
            "baseline_ready": values["historical_observations"] >= MIN_HISTORY_WEEKS,
            "similarity_scored_feature_count": int(values["similarity_scored_feature_count"]),
            "similarity_scored_features": [feature for feature, item in comparisons.items() if item["z_score"] is not None],
            "excluded_from_similarity_features": [{
                "feature": "purchase_record_count",
                "reason": "Excluded from similarity scoring because the genuine 2022 source period does not cover this full calendar week.",
            }] if values["is_partial_source_window"] else [],
            "behavioral_features": {feature: numeric_or_none(values[feature]) for feature in FEATURES},
            "feature_comparison": comparisons,
            "behavioral_similarity_score": numeric_or_none(values["behavioral_similarity_score"]),
            "drift_status": values["drift_status"], "top_deviating_features": top_features,
        })
    return fingerprints


def main():
    spark = SparkSession.builder.appName("AmazonBehavioralDriftIntelligence").master("local[*]").getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    try:
        source_data = spark.read.parquet(str(DATA_FILE))
        fingerprints = serialise_fingerprints(add_historical_baseline(build_weekly_fingerprints(source_data)))
        with open(OUTPUT_FILE, "w", encoding="utf-8") as file:
            json.dump(fingerprints, file, indent=4)
        print(f"Generated {len(fingerprints)} Amazon 2022 weekly behavioural fingerprints.")
        print(f"Saved to: {OUTPUT_FILE}")
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
