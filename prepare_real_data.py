import pandas as pd
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

INPUT_FILE = (
    BASE_DIR
    / "data"
    / "online_retail_II.xlsx"
)

OUTPUT_FILE = (
    BASE_DIR
    / "data"
    / "online_retail_II.parquet"
)


# ============================================================
# READ DATASET
# ============================================================

print("Reading Online Retail II dataset...")


sheets = pd.read_excel(
    INPUT_FILE,
    sheet_name=None
)


frames = []


for sheet_name, df in sheets.items():

    print(
        f"Reading sheet: {sheet_name} "
        f"({len(df):,} rows)"
    )


    # ========================================================
    # NORMALIZE COLUMN NAMES BETWEEN BOTH EXCEL SHEETS
    # ========================================================

    rename_map = {

        "Invoice": "InvoiceNo",
        "InvoiceNo": "InvoiceNo",

        "StockCode": "StockCode",

        "Description": "Description",

        "Quantity": "Quantity",

        "InvoiceDate": "InvoiceDate",

        "Price": "UnitPrice",
        "UnitPrice": "UnitPrice",

        "Customer ID": "CustomerID",
        "CustomerID": "CustomerID",

        "Country": "Country"
    }


    df = df.rename(
        columns=rename_map
    )


    required_columns = [

        "InvoiceNo",
        "StockCode",
        "Description",
        "Quantity",
        "InvoiceDate",
        "UnitPrice",
        "CustomerID",
        "Country"

    ]


    df = df[
        required_columns
    ]


    frames.append(
        df
    )


# ============================================================
# COMBINE BOTH PUBLISHED DATASET SHEETS
# ============================================================

data = pd.concat(
    frames,
    ignore_index=True
)


# ============================================================
# STANDARDIZE DATATYPES
# ============================================================

# ------------------------------------------------------------
# InvoiceNo
#
# Some invoices contain cancellation codes such as C489449.
# Therefore InvoiceNo must be treated as an identifier/string,
# not as an integer.
# ------------------------------------------------------------

data["InvoiceNo"] = (
    data["InvoiceNo"]
    .astype("string")
)


# ------------------------------------------------------------
# StockCode
#
# Stock codes are identifiers and may contain both numbers
# and characters.
# ------------------------------------------------------------

data["StockCode"] = (
    data["StockCode"]
    .astype("string")
)


# ------------------------------------------------------------
# Description
# ------------------------------------------------------------

data["Description"] = (
    data["Description"]
    .astype("string")
)


# ------------------------------------------------------------
# CustomerID
#
# Excel may interpret Customer IDs as floats because the
# column contains missing values.
#
# Convert to string and remove the artificial ".0".
# Missing values remain missing.
# ------------------------------------------------------------

data["CustomerID"] = (
    data["CustomerID"]
    .astype("string")
    .str.replace(
        r"\.0$",
        "",
        regex=True
    )
)


# ------------------------------------------------------------
# Country
# ------------------------------------------------------------

data["Country"] = (
    data["Country"]
    .astype("string")
)


# ------------------------------------------------------------
# Quantity
# ------------------------------------------------------------

data["Quantity"] = pd.to_numeric(
    data["Quantity"],
    errors="coerce"
)


# ------------------------------------------------------------
# UnitPrice
# ------------------------------------------------------------

data["UnitPrice"] = pd.to_numeric(
    data["UnitPrice"],
    errors="coerce"
)


# ------------------------------------------------------------
# InvoiceDate
# ------------------------------------------------------------

data["InvoiceDate"] = pd.to_datetime(
    data["InvoiceDate"],
    errors="coerce"
)


# ============================================================
# DATASET INFORMATION
# ============================================================

print("\nDataset loaded.")

print(
    "Rows:",
    f"{len(data):,}"
)

print(
    "Columns:",
    len(data.columns)
)

print(
    "Date Range:",
    data["InvoiceDate"].min(),
    "to",
    data["InvoiceDate"].max()
)


print("\nMissing Values:")

print(
    data.isna().sum()
)


print("\nColumn Data Types:")

print(
    data.dtypes
)


# ============================================================
# IMPORTANT:
#
# DO NOT remove:
#
# - Missing values
# - Duplicate records
# - Negative quantities
# - Unusual prices
# - Cancelled invoices
#
# These are genuine characteristics of the published dataset
# and are useful for Data Health Analysis.
# ============================================================


# ============================================================
# SAVE AS PARQUET
# ============================================================

print(
    "\nSaving dataset as Parquet..."
)


data.to_parquet(
    OUTPUT_FILE,
    index=False,
    engine="pyarrow",
    coerce_timestamps="us",
    allow_truncated_timestamps=True
)


print(
    "\nDataset successfully saved:"
)

print(
    OUTPUT_FILE
)


print(
    "\nNo synthetic transaction records "
    "were generated or added."
)