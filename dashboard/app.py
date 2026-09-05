import streamlit as st
import pandas as pd
import json
from pathlib import Path


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Intelligent Data Fingerprinting",
    page_icon="📊",
    layout="wide"
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

FINGERPRINT_FILE = (
    BASE_DIR
    / "fingerprints"
    / "monthly_fingerprints.json"
)


# ============================================================
# LOAD FINGERPRINTS
# ============================================================

@st.cache_data
def load_fingerprints():

    with open(
        FINGERPRINT_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(file)


fingerprints = load_fingerprints()


months = [
    item["month"]
    for item in fingerprints
]


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title(
    "Data Health Monitor"
)


selected_month = st.sidebar.selectbox(
    "Select Transaction Month",
    months,
    index=len(months) - 1
)


selected_data = next(
    item
    for item in fingerprints
    if item["month"] == selected_month
)


st.sidebar.divider()


st.sidebar.subheader(
    "Dataset"
)

st.sidebar.write(
    "UCI Online Retail II"
)

st.sidebar.caption(
    "Real public e-commerce transaction dataset"
)


st.sidebar.divider()


st.sidebar.subheader(
    "Technology Stack"
)

st.sidebar.write(
    "Apache Spark / PySpark"
)

st.sidebar.write(
    "Python"
)

st.sidebar.write(
    "Pandas"
)

st.sidebar.write(
    "Streamlit"
)


# ============================================================
# TITLE
# ============================================================

st.title(
    "Intelligent Data Fingerprinting "
    "for Big Data Health Analysis"
)


st.write(
    "PySpark-based monitoring of real monthly "
    "transaction batches from the UCI Online Retail II dataset."
)


st.info(
    "Each batch is derived from the real InvoiceDate field. "
    "No synthetic monthly transaction data is used."
)


st.divider()


# ============================================================
# OVERVIEW
# ============================================================

st.header(
    f"Batch Overview — {selected_month}"
)


c1, c2, c3, c4 = st.columns(4)


c1.metric(
    "Health Score",
    f"{selected_data['health_score']}/100"
)


c2.metric(
    "Health Status",
    selected_data["health_status"]
)


c3.metric(
    "Records",
    f"{selected_data['total_records']:,}"
)


c4.metric(
    "Columns",
    selected_data["total_columns"]
)


if selected_data["health_status"] == "Healthy":

    st.success(
        "This batch is classified as Healthy."
    )

elif selected_data["health_status"] == "Warning":

    st.warning(
        "This batch is classified as Warning."
    )

else:

    st.error(
        "This batch is classified as Critical."
    )


# ============================================================
# FEATURE 1
# DATA FINGERPRINT
# ============================================================

st.divider()

st.header(
    "1. Automated Data Fingerprinting"
)


f1, f2, f3 = st.columns(3)


f1.metric(
    "Missing %",
    f"{selected_data['missing_percentage']:.2f}%"
)


f2.metric(
    "Duplicate %",
    f"{selected_data['duplicate_percentage']:.2f}%"
)


f3.metric(
    "Growth Rate",
    f"{selected_data['growth_rate']:.2f}%"
)


# ============================================================
# MISSING VALUES
# ============================================================

st.subheader(
    "Missing Values by Column"
)


missing_df = pd.DataFrame(
    [
        {
            "Column": column,
            "Missing Count":
                selected_data[
                    "missing_values_per_column"
                ][column],
            "Missing %":
                percentage
        }

        for column, percentage
        in selected_data[
            "missing_percentage_per_column"
        ].items()
    ]
)


st.dataframe(
    missing_df,
    use_container_width=True,
    hide_index=True
)


st.bar_chart(
    missing_df.set_index(
        "Column"
    )["Missing %"]
)


# ============================================================
# NUMERICAL SUMMARY
# ============================================================

st.subheader(
    "Numerical Summary"
)


numerical_rows = []


for column, stats in selected_data[
    "numerical_statistics"
].items():

    numerical_rows.append(
        {
            "Column":
                column,

            "Count":
                stats["count"],

            "Mean":
                stats["mean"],

            "Std Dev":
                stats["stddev"],

            "Minimum":
                stats["min"],

            "Maximum":
                stats["max"]
        }
    )


numerical_df = pd.DataFrame(
    numerical_rows
)


st.dataframe(
    numerical_df,
    use_container_width=True,
    hide_index=True
)


# ============================================================
# FEATURE 2
# HISTORICAL COMPARISON
# ============================================================

st.divider()

st.header(
    "2. Historical Fingerprint Comparison"
)


history_df = pd.DataFrame(
    [
        {
            "Month":
                item["month"],

            "Records":
                item["total_records"],

            "Growth %":
                item["growth_rate"],

            "Missing %":
                item["missing_percentage"],

            "Duplicate %":
                item["duplicate_percentage"],

            "Health Score":
                item["health_score"]
        }

        for item in fingerprints
    ]
)


st.subheader(
    "Monthly Record Count"
)


st.line_chart(
    history_df.set_index(
        "Month"
    )[["Records"]]
)


st.subheader(
    "Health Score Trend"
)


st.line_chart(
    history_df.set_index(
        "Month"
    )[["Health Score"]]
)


with st.expander(
    "View Historical Fingerprint Table"
):

    st.dataframe(
        history_df,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# FEATURE 3
# DATA QUALITY ANOMALY DETECTION
# ============================================================

st.divider()

st.header(
    "3. Data Quality Anomaly Detection"
)


anomalies = selected_data[
    "data_quality_anomalies"
]


if anomalies:

    for anomaly in anomalies:

        st.warning(
            anomaly
        )

else:

    st.success(
        "No configured data-quality anomalies detected."
    )


st.caption(
    "Thresholds are configurable prototype monitoring rules "
    "and are not universal industry standards."
)


# ============================================================
# FEATURE 4
# IQR OUTLIER DETECTION
# ============================================================

st.divider()

st.header(
    "4. IQR-Based Outlier Detection"
)


outlier_rows = []


for column, values in selected_data[
    "outlier_analysis"
].items():

    outlier_rows.append(
        {
            "Column":
                column,

            "Q1":
                values["q1"],

            "Q3":
                values["q3"],

            "IQR":
                values["iqr"],

            "Lower Bound":
                values["lower_bound"],

            "Upper Bound":
                values["upper_bound"],

            "Outlier Count":
                values["outlier_count"],

            "Outlier %":
                values["outlier_percentage"]
        }
    )


outlier_df = pd.DataFrame(
    outlier_rows
)


st.dataframe(
    outlier_df,
    use_container_width=True,
    hide_index=True
)


st.latex(
    r"IQR = Q_3 - Q_1"
)

st.latex(
    r"Lower\ Bound = Q_1 - 1.5(IQR)"
)

st.latex(
    r"Upper\ Bound = Q_3 + 1.5(IQR)"
)


# ============================================================
# FEATURE 5
# SCHEMA DRIFT
# ============================================================

st.divider()

st.header(
    "5. Schema Drift Detection"
)


schema_drift = selected_data[
    "schema_drift"
]


if schema_drift.get(
    "baseline",
    False
):

    st.info(
        "This is the first historical batch and is used "
        "as the baseline schema."
    )


elif schema_drift[
    "detected"
]:

    st.error(
        "Schema drift detected."
    )


    if schema_drift[
        "added_columns"
    ]:

        st.write(
            "Added Columns:",
            schema_drift[
                "added_columns"
            ]
        )


    if schema_drift[
        "removed_columns"
    ]:

        st.write(
            "Removed Columns:",
            schema_drift[
                "removed_columns"
            ]
        )


    if schema_drift[
        "changed_data_types"
    ]:

        st.write(
            "Changed Data Types:",
            schema_drift[
                "changed_data_types"
            ]
        )


else:

    st.success(
        "No schema drift detected for this historical batch."
    )


with st.expander(
    "View Current Schema"
):

    schema_df = pd.DataFrame(
        [
            {
                "Column":
                    column,

                "Data Type":
                    datatype
            }

            for column, datatype
            in selected_data[
                "schema"
            ].items()
        ]
    )


    st.dataframe(
        schema_df,
        use_container_width=True,
        hide_index=True
    )


st.caption(
    "The source dataset uses a largely stable historical schema. "
    "A separate incoming-batch validation demo can be used "
    "to demonstrate schema-drift handling."
)


# ============================================================
# FEATURE 6
# HEALTH SCORE
# ============================================================

st.divider()

st.header(
    "6. Data Health Score & Severity"
)


h1, h2, h3 = st.columns(3)


h1.metric(
    "Health Score",
    f"{selected_data['health_score']}/100"
)


h2.metric(
    "Health Status",
    selected_data["health_status"]
)


h3.metric(
    "Severity",
    selected_data["severity"]
)


with st.expander(
    "View Health Score Deductions"
):

    deductions = selected_data[
        "deductions"
    ]


    if deductions:

        st.write(
            "**Starting Score: 100**"
        )


        for deduction in deductions:

            st.write(
                f"• {deduction}"
            )


        st.write(
            f"**Final Score: "
            f"{selected_data['health_score']}/100**"
        )

    else:

        st.success(
            "No deductions applied."
        )


# ============================================================
# NEAR-REAL-TIME BATCH MONITORING
# ============================================================

st.divider()

st.header(
    "Near-Real-Time Batch Monitoring"
)


st.code(
"""
New Data Batch Arrives
        |
        v
PySpark Loads the Batch
        |
        v
Generate Data Fingerprint
        |
        v
Compare with Historical Baseline
        |
        v
Detect Missing / Duplicate / Growth Issues
        |
        v
Perform IQR Outlier Analysis
        |
        v
Validate Schema
        |
        v
Calculate Data Health Score
        |
        v
Update Dashboard
"""
)


st.write(
    """
The current implementation is batch-based.
When a new transaction batch becomes available,
the same PySpark pipeline can profile it,
compare it with historical fingerprints,
calculate the updated health score,
and display the new result in Streamlit.
"""
)


# ============================================================
# SOURCE
# ============================================================

st.divider()

st.subheader(
    "Dataset Source"
)


st.write(
    "Online Retail II — "
    "UCI Machine Learning Repository"
)


st.write(
    "Chen, D. (2012). "
    "Online Retail II [Dataset]. "
    "UCI Machine Learning Repository."
)


st.write(
    "DOI: 10.24432/C5CG6D"
)