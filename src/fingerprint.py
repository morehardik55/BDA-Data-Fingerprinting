from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pathlib import Path
import json


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_FILE = (
    BASE_DIR
    / "data"
    / "online_retail_II.parquet"
)

OUTPUT_DIR = (
    BASE_DIR
    / "fingerprints"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

OUTPUT_FILE = (
    OUTPUT_DIR
    / "monthly_fingerprints.json"
)


# ============================================================
# CONFIGURABLE THRESHOLDS
# ============================================================

MISSING_COLUMN_THRESHOLD = 20.0
DUPLICATE_THRESHOLD = 3.0
GROWTH_THRESHOLD = 50.0
OUTLIER_THRESHOLD = 3.0


# ============================================================
# SPARK SESSION
# ============================================================

spark = (
    SparkSession.builder
    .appName(
        "IntelligentDataFingerprinting"
    )
    .master("local[*]")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")


# ============================================================
# LOAD REAL DATASET
# ============================================================

print("\nLoading real UCI Online Retail II dataset...")

df = spark.read.parquet(
    str(DATA_FILE)
)


# ============================================================
# CREATE REAL MONTHLY BATCHES
#
# IMPORTANT:
# We are NOT creating fake months.
# BatchMonth is derived directly from InvoiceDate.
# ============================================================

df = df.withColumn(
    "BatchMonth",
    F.date_format(
        F.col("InvoiceDate"),
        "yyyy-MM"
    )
)


df = df.filter(
    F.col("BatchMonth").isNotNull()
)


months = [
    row["BatchMonth"]
    for row in (
        df.select("BatchMonth")
        .distinct()
        .orderBy("BatchMonth")
        .collect()
    )
]


print(
    f"Detected {len(months)} real monthly batches."
)

print(
    "Months:",
    months
)


# ============================================================
# HELPER FUNCTION: SCHEMA
# ============================================================

def get_schema(dataframe):

    return {
        field.name:
            field.dataType.simpleString()

        for field
        in dataframe.schema.fields

        if field.name != "BatchMonth"
    }


# ============================================================
# HELPER FUNCTION: SCHEMA DRIFT
# ============================================================

def detect_schema_drift(
    previous_schema,
    current_schema
):

    if previous_schema is None:

        return {
            "detected": False,
            "baseline": True,
            "added_columns": [],
            "removed_columns": [],
            "changed_data_types": []
        }


    previous_columns = set(
        previous_schema.keys()
    )

    current_columns = set(
        current_schema.keys()
    )


    added_columns = sorted(
        current_columns
        - previous_columns
    )


    removed_columns = sorted(
        previous_columns
        - current_columns
    )


    changed_data_types = []


    for column in (
        previous_columns
        & current_columns
    ):

        if (
            previous_schema[column]
            != current_schema[column]
        ):

            changed_data_types.append(
                {
                    "column":
                        column,

                    "previous_type":
                        previous_schema[
                            column
                        ],

                    "current_type":
                        current_schema[
                            column
                        ]
                }
            )


    detected = bool(
        added_columns
        or removed_columns
        or changed_data_types
    )


    return {
        "detected":
            detected,

        "baseline":
            False,

        "added_columns":
            added_columns,

        "removed_columns":
            removed_columns,

        "changed_data_types":
            changed_data_types
    }


# ============================================================
# MAIN FINGERPRINT FUNCTION
# ============================================================

def create_fingerprint(
    batch_df,
    month,
    previous_records=None,
    previous_schema=None
):

    profile_df = batch_df.drop(
        "BatchMonth"
    )


    # ========================================================
    # BASIC PROFILE
    # ========================================================

    total_records = (
        profile_df.count()
    )


    columns = (
        profile_df.columns
    )


    total_columns = len(
        columns
    )


    # ========================================================
    # NULL ANALYSIS
    # ========================================================

    null_expressions = [

        F.sum(
            F.when(
                F.col(column).isNull(),
                1
            ).otherwise(0)
        ).alias(column)

        for column in columns
    ]


    null_row = (
        profile_df
        .select(
            null_expressions
        )
        .collect()[0]
        .asDict()
    )


    missing_values_per_column = {

        column:
            int(
                null_row[column]
                or 0
            )

        for column in columns
    }


    missing_percentage_per_column = {

        column:
            round(
                (
                    missing_values_per_column[
                        column
                    ]
                    / total_records
                    * 100
                )
                if total_records > 0
                else 0,
                4
            )

        for column in columns
    }


    total_missing = sum(
        missing_values_per_column.values()
    )


    total_cells = (
        total_records
        * total_columns
    )


    missing_percentage = round(
        (
            total_missing
            / total_cells
            * 100
        )
        if total_cells > 0
        else 0,
        4
    )


    # ========================================================
    # DUPLICATE ANALYSIS
    # ========================================================

    unique_records = (
        profile_df
        .dropDuplicates()
        .count()
    )


    duplicate_records = (
        total_records
        - unique_records
    )


    duplicate_percentage = round(
        (
            duplicate_records
            / total_records
            * 100
        )
        if total_records > 0
        else 0,
        4
    )


    # ========================================================
    # HISTORICAL GROWTH
    # ========================================================

    if (
        previous_records is not None
        and previous_records > 0
    ):

        growth_rate = round(
            (
                total_records
                - previous_records
            )
            / previous_records
            * 100,
            4
        )

    else:

        growth_rate = 0.0


    # ========================================================
    # NUMERICAL SUMMARY
    # ========================================================

    numerical_columns = [
        column
        for column in [
            "Quantity",
            "UnitPrice"
        ]
        if column in profile_df.columns
    ]


    numerical_statistics = {}


    for column in numerical_columns:

        stats = (
            profile_df
            .select(
                F.col(column)
                .cast("double")
                .alias(column)
            )
            .summary(
                "count",
                "mean",
                "stddev",
                "min",
                "max"
            )
            .collect()
        )


        stat_map = {
            row["summary"]:
                row[column]

            for row in stats
        }


        def safe_float(value):

            try:
                return float(value)

            except (
                TypeError,
                ValueError
            ):
                return None


        numerical_statistics[
            column
        ] = {

            "count":
                int(
                    float(
                        stat_map.get(
                            "count",
                            0
                        )
                    )
                ),

            "mean":
                safe_float(
                    stat_map.get(
                        "mean"
                    )
                ),

            "stddev":
                safe_float(
                    stat_map.get(
                        "stddev"
                    )
                ),

            "min":
                safe_float(
                    stat_map.get(
                        "min"
                    )
                ),

            "max":
                safe_float(
                    stat_map.get(
                        "max"
                    )
                )
        }


    # ========================================================
    # IQR OUTLIER ANALYSIS
    # ========================================================

    outlier_analysis = {}


    for column in numerical_columns:

        numeric_df = (
            profile_df
            .select(
                F.col(column)
                .cast("double")
                .alias(column)
            )
            .filter(
                F.col(column).isNotNull()
            )
        )


        quantiles = (
            numeric_df
            .approxQuantile(
                column,
                [0.25, 0.75],
                0.01
            )
        )


        if len(quantiles) == 2:

            q1 = quantiles[0]
            q3 = quantiles[1]

            iqr = (
                q3 - q1
            )


            lower_bound = (
                q1
                - 1.5 * iqr
            )


            upper_bound = (
                q3
                + 1.5 * iqr
            )


            outlier_count = (
                numeric_df
                .filter(
                    (
                        F.col(column)
                        < lower_bound
                    )
                    |
                    (
                        F.col(column)
                        > upper_bound
                    )
                )
                .count()
            )


            valid_count = (
                numeric_df.count()
            )


            outlier_percentage = round(
                (
                    outlier_count
                    / valid_count
                    * 100
                )
                if valid_count > 0
                else 0,
                4
            )


            outlier_analysis[
                column
            ] = {

                "q1":
                    q1,

                "q3":
                    q3,

                "iqr":
                    iqr,

                "lower_bound":
                    lower_bound,

                "upper_bound":
                    upper_bound,

                "outlier_count":
                    outlier_count,

                "outlier_percentage":
                    outlier_percentage
            }


    # ========================================================
    # SCHEMA
    # ========================================================

    current_schema = (
        get_schema(
            profile_df
        )
    )


    schema_drift = (
        detect_schema_drift(
            previous_schema,
            current_schema
        )
    )


    # ========================================================
    # DATA QUALITY ANOMALIES
    # ========================================================

    anomalies = []


    high_missing_columns = {

        column:
            percentage

        for (
            column,
            percentage
        ) in (
            missing_percentage_per_column.items()
        )

        if (
            percentage
            > MISSING_COLUMN_THRESHOLD
        )
    }


    for (
        column,
        percentage
    ) in high_missing_columns.items():

        anomalies.append(
            f"High missing values in "
            f"{column}: "
            f"{percentage:.2f}%"
        )


    if (
        duplicate_percentage
        > DUPLICATE_THRESHOLD
    ):

        anomalies.append(
            f"High duplicate percentage: "
            f"{duplicate_percentage:.2f}%"
        )


    if (
        previous_records is not None
        and abs(
            growth_rate
        ) > GROWTH_THRESHOLD
    ):

        anomalies.append(
            f"Abnormal batch-size change: "
            f"{growth_rate:.2f}%"
        )


    for (
        column,
        result
    ) in outlier_analysis.items():

        if (
            result[
                "outlier_percentage"
            ]
            > OUTLIER_THRESHOLD
        ):

            anomalies.append(
                f"High numerical outlier "
                f"percentage in {column}: "
                f"{result['outlier_percentage']:.2f}%"
            )


    if (
        schema_drift[
            "detected"
        ]
    ):

        anomalies.append(
            "Schema drift detected."
        )


    # ========================================================
    # DATA HEALTH SCORE
    # ========================================================

    health_score = 100

    deductions = []


    if high_missing_columns:

        health_score -= 15

        deductions.append(
            "High missing values: -15"
        )


    if (
        duplicate_percentage
        > DUPLICATE_THRESHOLD
    ):

        health_score -= 15

        deductions.append(
            "High duplicate percentage: -15"
        )


    if (
        previous_records is not None
        and abs(
            growth_rate
        ) > GROWTH_THRESHOLD
    ):

        health_score -= 20

        deductions.append(
            "Abnormal batch-size change: -20"
        )


    high_outliers = any(

        result[
            "outlier_percentage"
        ]
        > OUTLIER_THRESHOLD

        for result
        in outlier_analysis.values()
    )


    if high_outliers:

        health_score -= 15

        deductions.append(
            "High numerical outliers: -15"
        )


    if (
        schema_drift[
            "detected"
        ]
    ):

        health_score -= 25

        deductions.append(
            "Schema drift: -25"
        )


    health_score = max(
        health_score,
        0
    )


    if health_score >= 90:

        health_status = (
            "Healthy"
        )

    elif health_score >= 70:

        health_status = (
            "Warning"
        )

    else:

        health_status = (
            "Critical"
        )


    # ========================================================
    # RETURN FINGERPRINT
    # ========================================================

    return {

        "month":
            month,

        "total_records":
            total_records,

        "total_columns":
            total_columns,

        "missing_percentage":
            missing_percentage,

        "total_missing":
            total_missing,

        "missing_values_per_column":
            missing_values_per_column,

        "missing_percentage_per_column":
            missing_percentage_per_column,

        "duplicate_records":
            duplicate_records,

        "duplicate_percentage":
            duplicate_percentage,

        "growth_rate":
            growth_rate,

        "numerical_statistics":
            numerical_statistics,

        "data_quality_anomalies":
            anomalies,

        "outlier_analysis":
            outlier_analysis,

        "schema":
            current_schema,

        "schema_drift":
            schema_drift,

        "health_score":
            health_score,

        "health_status":
            health_status,

        "severity":
            health_status,

        "deductions":
            deductions
    }


# ============================================================
# PROCESS ALL REAL MONTHLY BATCHES
# ============================================================

fingerprints = []


previous_records = None

previous_schema = None


for month in months:

    print(
        "\n========================================"
    )

    print(
        f"Processing Month: {month}"
    )

    print(
        "========================================"
    )


    monthly_df = (
        df.filter(
            F.col("BatchMonth")
            == month
        )
    )


    fingerprint = (
        create_fingerprint(
            monthly_df,
            month,
            previous_records,
            previous_schema
        )
    )


    fingerprints.append(
        fingerprint
    )


    print(
        f"Records: "
        f"{fingerprint['total_records']:,}"
    )

    print(
        f"Columns: "
        f"{fingerprint['total_columns']}"
    )

    print(
        f"Missing %: "
        f"{fingerprint['missing_percentage']:.2f}%"
    )

    print(
        f"Duplicates %: "
        f"{fingerprint['duplicate_percentage']:.2f}%"
    )

    print(
        f"Growth: "
        f"{fingerprint['growth_rate']:.2f}%"
    )


    print("\nAnomalies:")

    if (
        fingerprint[
            "data_quality_anomalies"
        ]
    ):

        for anomaly in (
            fingerprint[
                "data_quality_anomalies"
            ]
        ):

            print(
                "-",
                anomaly
            )

    else:

        print(
            "- No configured anomalies detected."
        )


    print("\nOutliers:")

    for (
        column,
        result
    ) in (
        fingerprint[
            "outlier_analysis"
        ].items()
    ):

        print(
            f"- {column}: "
            f"{result['outlier_count']:,} "
            f"({result['outlier_percentage']:.2f}%)"
        )


    print(
        "\nSchema Drift:",
        fingerprint[
            "schema_drift"
        ][
            "detected"
        ]
    )


    print(
        "Health Score:",
        f"{fingerprint['health_score']}/100"
    )


    print(
        "Health Status:",
        fingerprint[
            "health_status"
        ]
    )


    previous_records = (
        fingerprint[
            "total_records"
        ]
    )


    previous_schema = (
        fingerprint[
            "schema"
        ]
    )


# ============================================================
# SAVE FINGERPRINTS
# ============================================================

with open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        fingerprints,
        file,
        indent=4,
        default=str
    )


print(
    "\n========================================"
)

print(
    "Fingerprint generation completed."
)

print(
    "Saved to:"
)

print(
    OUTPUT_FILE
)

print(
    "========================================"
)


spark.stop()