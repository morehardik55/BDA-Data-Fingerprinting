"""Implementation 1: Amazon 2022 data-health profiling with PySpark."""

import json
from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql import functions as F


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = BASE_DIR / "data" / "amazon_purchases_2022.parquet"
OUTPUT_FILE = BASE_DIR / "fingerprints" / "monthly_fingerprints.json"

MISSING_COLUMN_THRESHOLD = 20.0
DUPLICATE_THRESHOLD = 3.0
GROWTH_THRESHOLD = 50.0
OUTLIER_THRESHOLD = 3.0
NUMERICAL_COLUMNS = ["Quantity", "Purchase Price Per Unit"]


def get_schema(dataframe):
    return {
        field.name: field.dataType.simpleString()
        for field in dataframe.schema.fields
        if field.name != "BatchMonth"
    }


def detect_schema_drift(previous_schema, current_schema):
    if previous_schema is None:
        return {"detected": False, "baseline": True, "added_columns": [], "removed_columns": [], "changed_data_types": []}

    previous_columns, current_columns = set(previous_schema), set(current_schema)
    changed = [
        {"column": column, "previous_type": previous_schema[column], "current_type": current_schema[column]}
        for column in sorted(previous_columns & current_columns)
        if previous_schema[column] != current_schema[column]
    ]
    added, removed = sorted(current_columns - previous_columns), sorted(previous_columns - current_columns)
    return {
        "detected": bool(added or removed or changed), "baseline": False,
        "added_columns": added, "removed_columns": removed, "changed_data_types": changed,
    }


def create_fingerprint(batch_df, month, previous_records=None, previous_schema=None):
    profile_df = batch_df.drop("BatchMonth")
    columns = profile_df.columns
    total_records, total_columns = profile_df.count(), len(columns)

    null_row = profile_df.select(*[
        F.sum(F.when(F.col(column).isNull(), 1).otherwise(0)).alias(column)
        for column in columns
    ]).first().asDict()
    missing_values = {column: int(null_row[column] or 0) for column in columns}
    missing_percentages = {
        column: round(missing_values[column] / total_records * 100, 4) if total_records else 0
        for column in columns
    }
    total_missing = sum(missing_values.values())
    missing_percentage = round(total_missing / (total_records * total_columns) * 100, 4) if total_records and total_columns else 0

    duplicate_records = total_records - profile_df.dropDuplicates().count()
    duplicate_percentage = round(duplicate_records / total_records * 100, 4) if total_records else 0
    growth_rate = round((total_records - previous_records) / previous_records * 100, 4) if previous_records else 0.0

    numerical_statistics, outlier_analysis = {}, {}
    for column in [name for name in NUMERICAL_COLUMNS if name in columns]:
        numeric_df = profile_df.select(F.col(column).cast("double").alias(column)).filter(F.col(column).isNotNull())
        summary = {row["summary"]: row[column] for row in numeric_df.summary("count", "mean", "stddev", "min", "max").collect()}
        as_float = lambda value: float(value) if value is not None else None
        numerical_statistics[column] = {
            "count": int(float(summary.get("count", 0))), "mean": as_float(summary.get("mean")),
            "stddev": as_float(summary.get("stddev")), "min": as_float(summary.get("min")), "max": as_float(summary.get("max")),
        }
        quantiles = numeric_df.approxQuantile(column, [0.25, 0.75], 0.01)
        if len(quantiles) == 2:
            q1, q3 = quantiles
            iqr = q3 - q1
            lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            valid_count = numeric_df.count()
            outlier_count = numeric_df.filter((F.col(column) < lower) | (F.col(column) > upper)).count()
            outlier_analysis[column] = {
                "q1": q1, "q3": q3, "iqr": iqr, "lower_bound": lower, "upper_bound": upper,
                "outlier_count": outlier_count,
                "outlier_percentage": round(outlier_count / valid_count * 100, 4) if valid_count else 0,
            }

    current_schema = get_schema(profile_df)
    schema_drift = detect_schema_drift(previous_schema, current_schema)
    high_missing = {column: value for column, value in missing_percentages.items() if value > MISSING_COLUMN_THRESHOLD}
    anomalies = [f"High missing values in {column}: {value:.2f}%" for column, value in high_missing.items()]
    if duplicate_percentage > DUPLICATE_THRESHOLD:
        anomalies.append(f"High duplicate percentage: {duplicate_percentage:.2f}%")
    if previous_records and abs(growth_rate) > GROWTH_THRESHOLD:
        anomalies.append(f"Abnormal batch-size change: {growth_rate:.2f}%")
    for column, result in outlier_analysis.items():
        if result["outlier_percentage"] > OUTLIER_THRESHOLD:
            anomalies.append(f"High numerical outlier percentage in {column}: {result['outlier_percentage']:.2f}%")
    if schema_drift["detected"]:
        anomalies.append("Schema drift detected.")

    deductions, health_score = [], 100
    for condition, deduction, label in [
        (bool(high_missing), 15, "High missing values"),
        (duplicate_percentage > DUPLICATE_THRESHOLD, 15, "High duplicate percentage"),
        (bool(previous_records and abs(growth_rate) > GROWTH_THRESHOLD), 20, "Abnormal batch-size change"),
        (any(item["outlier_percentage"] > OUTLIER_THRESHOLD for item in outlier_analysis.values()), 15, "High numerical outliers"),
        (schema_drift["detected"], 25, "Schema drift"),
    ]:
        if condition:
            health_score -= deduction
            deductions.append(f"{label}: -{deduction}")
    health_score = max(health_score, 0)
    health_status = "Healthy" if health_score >= 90 else "Warning" if health_score >= 70 else "Critical"

    return {
        "month": month, "total_records": total_records, "total_columns": total_columns,
        "missing_percentage": missing_percentage, "total_missing": total_missing,
        "missing_values_per_column": missing_values, "missing_percentage_per_column": missing_percentages,
        "duplicate_records": duplicate_records, "duplicate_percentage": duplicate_percentage,
        "growth_rate": growth_rate, "numerical_statistics": numerical_statistics,
        "data_quality_anomalies": anomalies, "outlier_analysis": outlier_analysis,
        "schema": current_schema, "schema_drift": schema_drift, "health_score": health_score,
        "health_status": health_status, "severity": health_status, "deductions": deductions,
    }


def create_all_monthly_fingerprints(df):
    """Profile all months with grouped Spark aggregations before collecting 12 results."""
    source_columns = [column for column in df.columns if column != "BatchMonth"]
    missing_aliases = {column: f"missing_{index}" for index, column in enumerate(source_columns)}
    base = df.groupBy("BatchMonth").agg(
        F.count(F.lit(1)).alias("total_records"),
        *[F.sum(F.when(F.col(column).isNull(), 1).otherwise(0)).alias(missing_aliases[column]) for column in source_columns],
        *[expression for column in NUMERICAL_COLUMNS for expression in [
            F.avg(F.col(column)).alias(f"{column}_mean"), F.stddev_samp(F.col(column)).alias(f"{column}_stddev"),
            F.min(F.col(column)).alias(f"{column}_min"), F.max(F.col(column)).alias(f"{column}_max"),
        ]],
        *[F.percentile_approx(F.col(column), F.array(F.lit(0.25), F.lit(0.75)), 10000).alias(f"{column}_quantiles") for column in NUMERICAL_COLUMNS],
    )
    unique_rows = df.dropDuplicates(["BatchMonth", *source_columns]).groupBy("BatchMonth").agg(F.count(F.lit(1)).alias("unique_records"))
    bounds = base.select(
        "BatchMonth",
        *[expression for column in NUMERICAL_COLUMNS for expression in [
            F.col(f"{column}_quantiles")[0].alias(f"{column}_q1"), F.col(f"{column}_quantiles")[1].alias(f"{column}_q3"),
        ]],
    )
    joined = df.join(bounds, "BatchMonth")
    outliers = joined.groupBy("BatchMonth").agg(*[
        F.sum(F.when((F.col(column) < F.col(f"{column}_q1") - 1.5 * (F.col(f"{column}_q3") - F.col(f"{column}_q1"))) | (F.col(column) > F.col(f"{column}_q3") + 1.5 * (F.col(f"{column}_q3") - F.col(f"{column}_q1"))), 1).otherwise(0)).alias(f"{column}_outlier_count")
        for column in NUMERICAL_COLUMNS
    ])
    rows = base.join(unique_rows, "BatchMonth").join(outliers, "BatchMonth").orderBy("BatchMonth").collect()
    schema = get_schema(df)
    fingerprints, previous_records, previous_schema = [], None, None
    for row in rows:
        values = row.asDict(recursive=True)
        total_records = int(values["total_records"])
        missing_values = {column: int(values[missing_aliases[column]] or 0) for column in source_columns}
        missing_percentages = {column: round(value / total_records * 100, 4) if total_records else 0 for column, value in missing_values.items()}
        total_missing = sum(missing_values.values())
        duplicate_records = total_records - int(values["unique_records"])
        duplicate_percentage = round(duplicate_records / total_records * 100, 4) if total_records else 0
        growth_rate = round((total_records - previous_records) / previous_records * 100, 4) if previous_records else 0.0
        numerical_statistics, outlier_analysis = {}, {}
        for column in NUMERICAL_COLUMNS:
            q1, q3 = float(values[f"{column}_quantiles"][0]), float(values[f"{column}_quantiles"][1])
            iqr = q3 - q1
            outlier_count = int(values[f"{column}_outlier_count"] or 0)
            numerical_statistics[column] = {"count": total_records, "mean": float(values[f"{column}_mean"]), "stddev": float(values[f"{column}_stddev"]), "min": float(values[f"{column}_min"]), "max": float(values[f"{column}_max"])}
            outlier_analysis[column] = {"q1": q1, "q3": q3, "iqr": iqr, "lower_bound": q1 - 1.5 * iqr, "upper_bound": q3 + 1.5 * iqr, "outlier_count": outlier_count, "outlier_percentage": round(outlier_count / total_records * 100, 4)}
        schema_drift = detect_schema_drift(previous_schema, schema)
        high_missing = {column: value for column, value in missing_percentages.items() if value > MISSING_COLUMN_THRESHOLD}
        anomalies = [f"High missing values in {column}: {value:.2f}%" for column, value in high_missing.items()]
        if duplicate_percentage > DUPLICATE_THRESHOLD: anomalies.append(f"High duplicate percentage: {duplicate_percentage:.2f}%")
        if previous_records and abs(growth_rate) > GROWTH_THRESHOLD: anomalies.append(f"Abnormal batch-size change: {growth_rate:.2f}%")
        for column, result in outlier_analysis.items():
            if result["outlier_percentage"] > OUTLIER_THRESHOLD: anomalies.append(f"High numerical outlier percentage in {column}: {result['outlier_percentage']:.2f}%")
        if schema_drift["detected"]: anomalies.append("Schema drift detected.")
        deductions, health_score = [], 100
        for condition, deduction, label in [(bool(high_missing), 15, "High missing values"), (duplicate_percentage > DUPLICATE_THRESHOLD, 15, "High duplicate percentage"), (bool(previous_records and abs(growth_rate) > GROWTH_THRESHOLD), 20, "Abnormal batch-size change"), (any(item["outlier_percentage"] > OUTLIER_THRESHOLD for item in outlier_analysis.values()), 15, "High numerical outliers"), (schema_drift["detected"], 25, "Schema drift")]:
            if condition: health_score, deductions = health_score - deduction, deductions + [f"{label}: -{deduction}"]
        health_score = max(health_score, 0)
        health_status = "Healthy" if health_score >= 90 else "Warning" if health_score >= 70 else "Critical"
        fingerprints.append({"month": values["BatchMonth"], "total_records": total_records, "total_columns": len(source_columns), "missing_percentage": round(total_missing / (total_records * len(source_columns)) * 100, 4), "total_missing": total_missing, "missing_values_per_column": missing_values, "missing_percentage_per_column": missing_percentages, "duplicate_records": duplicate_records, "duplicate_percentage": duplicate_percentage, "growth_rate": growth_rate, "numerical_statistics": numerical_statistics, "data_quality_anomalies": anomalies, "outlier_analysis": outlier_analysis, "schema": schema, "schema_drift": schema_drift, "health_score": health_score, "health_status": health_status, "severity": health_status, "deductions": deductions})
        previous_records, previous_schema = total_records, schema
    return fingerprints


def main():
    spark = SparkSession.builder.appName("AmazonDataHealthProfiling").master("local[*]").config("spark.ui.enabled", "false").getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    try:
        df = spark.read.parquet(str(DATA_FILE)).withColumn("BatchMonth", F.date_format(F.col("Order Date"), "yyyy-MM"))
        fingerprints = create_all_monthly_fingerprints(df)
        with open(OUTPUT_FILE, "w", encoding="utf-8") as file: json.dump(fingerprints, file, indent=4, default=str)
        print(f"Saved {len(fingerprints)} Amazon 2022 monthly fingerprints to: {OUTPUT_FILE}")
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
