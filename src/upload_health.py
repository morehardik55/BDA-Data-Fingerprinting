"""Reusable Spark helpers for uploaded CSV data-health analysis."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Iterable

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    DateType,
    DoubleType,
    IntegerType,
    LongType,
    ShortType,
    FloatType,
    TimestampType,
)


NUMERIC_TYPES = (DoubleType, IntegerType, LongType, ShortType, FloatType)
DATE_TYPES = (DateType, TimestampType)


@dataclass
class UploadHealthResult:
    file_name: str
    row_count: int
    column_count: int
    schema_overview: list[dict]
    missing_by_column: list[dict]
    total_missing_values: int
    overall_missing_percentage: float
    duplicate_row_count: int
    duplicate_percentage: float
    numerical_summary: list[dict]
    outlier_analysis: list[dict]
    detected_issues: list[str]
    data_health_score: int
    health_status: str
    date_detection_message: str | None
    has_valid_rows: bool
    has_readable_columns: bool


def create_spark_session(existing_session: SparkSession | None = None) -> SparkSession:
    if existing_session is not None:
        return existing_session
    return (
        SparkSession.builder.appName("UploadedDatasetHealthAnalyzer")
        .master("local[2]")
        .config("spark.ui.enabled", "false")
        .config("spark.driver.host", "127.0.0.1")
        .config("spark.driver.bindAddress", "127.0.0.1")
        .getOrCreate()
    )


def load_uploaded_csv(spark: SparkSession, uploaded_file) -> tuple[DataFrame | None, Path | None, str | None]:
    suffix = Path(uploaded_file.name).suffix.lower()
    if suffix != ".csv":
        return None, None, "Please upload a valid .csv file."

    with NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
        tmp.write(uploaded_file.getbuffer())
        temp_path = Path(tmp.name)

    try:
        df = (
            spark.read.option("header", True)
            .option("inferSchema", True)
            .option("mode", "PERMISSIVE")
            .option("escape", "\"")
            .option("multiLine", True)
            .option("quote", "\"")
            .option("columnNameOfCorruptRecord", "_corrupt_record")
            .csv(str(temp_path))
        )
        # Spark evaluates CSV data lazily, so the temporary file must remain
        # available until the caller finishes every analysis action.
        return df, temp_path, None
    except Exception as exc:
        _remove_temporary_upload(temp_path)
        return None, None, f"Unable to read the CSV file: {exc}"


def _is_numeric_field(data_type) -> bool:
    return isinstance(data_type, NUMERIC_TYPES)


def _is_date_field(data_type) -> bool:
    return isinstance(data_type, DATE_TYPES)


def _safe_float(value):
    return None if value is None else float(value)


def _remove_temporary_upload(path: Path) -> None:
    """Remove an upload when Windows releases Spark's file handle."""
    try:
        path.unlink(missing_ok=True)
    except OSError:
        # Spark can retain a Windows file handle briefly after an action.
        # The OS temp directory will clean up any deferred file instead.
        pass


def detect_schema_overview(df: DataFrame) -> list[dict]:
    return [{"Column Name": field.name, "Detected Data Type": field.dataType.simpleString()} for field in df.schema.fields]


def calculate_missing_metrics(df: DataFrame) -> tuple[list[dict], int, float]:
    row_count = df.count()
    columns = df.columns
    if not columns:
        return [], 0, 0.0

    missing_exprs = [
        F.sum(F.when(F.col(column).isNull() | (F.trim(F.col(column).cast("string")) == ""), 1).otherwise(0)).alias(column)
        for column in columns
    ]
    result = df.select(*missing_exprs).first()
    missing_by_column = []
    total_missing = 0
    for column in columns:
        missing_count = int(result[column] or 0) if result else 0
        total_missing += missing_count
        missing_by_column.append({
            "Column": column,
            "Missing Count": missing_count,
            "Missing %": round((missing_count / row_count) * 100, 2) if row_count else 0.0,
        })
    overall_missing = round((total_missing / (row_count * len(columns))) * 100, 2) if row_count and columns else 0.0
    return missing_by_column, total_missing, overall_missing


def calculate_duplicate_metrics(df: DataFrame) -> tuple[int, float]:
    row_count = df.count()
    duplicate_count = row_count - df.dropDuplicates().count()
    duplicate_percentage = round((duplicate_count / row_count) * 100, 2) if row_count else 0.0
    return duplicate_count, duplicate_percentage


def calculate_numeric_summary(df: DataFrame) -> tuple[list[dict], list[str]]:
    summary_rows = []
    numeric_columns = [field.name for field in df.schema.fields if _is_numeric_field(field.dataType)]
    for column in numeric_columns:
        numeric_df = df.select(F.col(column).cast("double").alias(column)).filter(F.col(column).isNotNull())
        row = numeric_df.summary("count", "mean", "stddev", "min", "max").collect()
        summary_map = {item["summary"]: item[column] for item in row} if row else {}
        summary_rows.append({
            "Column": column,
            "Count": int(float(summary_map.get("count", 0) or 0)),
            "Mean": _safe_float(summary_map.get("mean")),
            "Std Dev": _safe_float(summary_map.get("stddev")),
            "Minimum": _safe_float(summary_map.get("min")),
            "Maximum": _safe_float(summary_map.get("max")),
        })
    return summary_rows, numeric_columns


def calculate_iqr_outliers(df: DataFrame, numeric_columns: Iterable[str]) -> list[dict]:
    outliers = []
    for column in numeric_columns:
        numeric_df = df.select(F.col(column).cast("double").alias(column)).filter(F.col(column).isNotNull())
        valid_count = numeric_df.count()
        if valid_count < 4:
            continue
        quantiles = numeric_df.approxQuantile(column, [0.25, 0.75], 0.01)
        if len(quantiles) != 2:
            continue
        q1, q3 = quantiles
        if q1 is None or q3 is None:
            continue
        iqr = q3 - q1
        if iqr <= 0:
            continue
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outlier_count = numeric_df.filter((F.col(column) < lower_bound) | (F.col(column) > upper_bound)).count()
        outliers.append({
            "Column": column,
            "Q1": round(float(q1), 4),
            "Q3": round(float(q3), 4),
            "Lower Bound": round(float(lower_bound), 4),
            "Upper Bound": round(float(upper_bound), 4),
            "Outlier Count": int(outlier_count),
            "Outlier %": round((outlier_count / valid_count) * 100, 2) if valid_count else 0.0,
        })
    return outliers


def detect_date_column(df: DataFrame) -> str | None:
    candidate_names = [field.name for field in df.schema.fields if _is_date_field(field.dataType)]
    if candidate_names:
        return candidate_names[0]

    candidates = []
    for field in df.schema.fields:
        lowered = field.name.lower()
        if any(token in lowered for token in ("date", "time", "timestamp")):
            candidates.append(field.name)
    return candidates[0] if candidates else None


def calculate_health_score(
    overall_missing_percentage: float,
    duplicate_percentage: float,
    overall_outlier_percentage: float,
    has_valid_rows: bool,
    has_readable_columns: bool,
) -> tuple[int, list[str]]:
    score = 100
    issues = []

    missing_penalty = 0 if overall_missing_percentage <= 1 else 10 if overall_missing_percentage <= 5 else 20 if overall_missing_percentage <= 15 else 30
    duplicate_penalty = 0 if duplicate_percentage <= 1 else 10 if duplicate_percentage <= 5 else 20 if duplicate_percentage <= 15 else 30
    outlier_penalty = 0 if overall_outlier_percentage <= 2 else 5 if overall_outlier_percentage <= 5 else 10 if overall_outlier_percentage <= 10 else 20

    if overall_missing_percentage > 1:
        issues.append(f"High missing-value rate detected: {overall_missing_percentage:.1f}%")
    if duplicate_percentage > 1:
        issues.append(f"Duplicate rows detected: {duplicate_percentage:.1f}%")
    if overall_outlier_percentage > 2:
        issues.append(f"Outliers detected across numeric fields: {overall_outlier_percentage:.1f}%")

    score -= missing_penalty + duplicate_penalty + outlier_penalty
    if not has_readable_columns or not has_valid_rows:
        score -= 10
        issues.append("Dataset usability is limited: readable columns or valid rows are missing.")

    score = max(0, min(100, score))
    if not issues:
        issues.append("No major data-health issues detected.")
    return score, issues


def analyze_uploaded_csv(spark: SparkSession, uploaded_file) -> tuple[UploadHealthResult | None, str | None]:
    df, temp_path, error = load_uploaded_csv(spark, uploaded_file)
    if error:
        return None, error
    if df is None or temp_path is None:
        return None, "Unable to read the CSV file."

    try:
        row_count = df.count()
        column_count = len(df.columns)
        schema_overview = detect_schema_overview(df)
        missing_by_column, total_missing, overall_missing_percentage = calculate_missing_metrics(df)
        duplicate_row_count, duplicate_percentage = calculate_duplicate_metrics(df)
        numerical_summary, numeric_columns = calculate_numeric_summary(df)
        outlier_analysis = calculate_iqr_outliers(df, numeric_columns)
        overall_outlier_percentage = 0.0
        if outlier_analysis:
            total_outliers = sum(item["Outlier Count"] for item in outlier_analysis)
            count_by_column = {item["Column"]: item["Count"] for item in numerical_summary}
            total_valid = sum(int(count_by_column.get(item["Column"], 0)) for item in outlier_analysis)
            overall_outlier_percentage = round((total_outliers / total_valid) * 100, 2) if total_valid else 0.0

        date_column = detect_date_column(df)
        date_detection_message = None
        if date_column:
            date_detection_message = (
                "Time-based data detected. Historical behavioral drift analysis may be possible for this dataset."
                f" Candidate date/time column: {date_column}."
            )

        has_valid_rows = row_count > 0 and column_count > 0
        has_readable_columns = bool(df.columns)
        score, issues = calculate_health_score(
            overall_missing_percentage,
            duplicate_percentage,
            overall_outlier_percentage,
            has_valid_rows,
            has_readable_columns,
        )
        health_status = "HEALTHY" if score >= 80 else "WARNING" if score >= 50 else "CRITICAL"

        return UploadHealthResult(
            file_name=uploaded_file.name,
            row_count=row_count,
            column_count=column_count,
            schema_overview=schema_overview,
            missing_by_column=missing_by_column,
            total_missing_values=total_missing,
            overall_missing_percentage=overall_missing_percentage,
            duplicate_row_count=duplicate_row_count,
            duplicate_percentage=duplicate_percentage,
            numerical_summary=numerical_summary,
            outlier_analysis=outlier_analysis,
            detected_issues=issues,
            data_health_score=score,
            health_status=health_status,
            date_detection_message=date_detection_message,
            has_valid_rows=has_valid_rows,
            has_readable_columns=has_readable_columns,
        ), None
    except Exception as exc:
        return None, f"Unable to analyze the CSV file: {exc}"
    finally:
        _remove_temporary_upload(temp_path)
