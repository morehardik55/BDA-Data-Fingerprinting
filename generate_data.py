import pandas as pd
import numpy as np
import os
import random
from datetime import datetime, timedelta

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

random.seed(42)
np.random.seed(42)

os.makedirs("data", exist_ok=True)

products = [
    "Laptop",
    "Smartphone",
    "Headphones",
    "Keyboard",
    "Mouse",
    "Monitor",
    "Smartwatch",
    "Tablet"
]

payment_methods = [
    "UPI",
    "Credit Card",
    "Debit Card",
    "Cash on Delivery"
]

locations = [
    "Mumbai",
    "Delhi",
    "Bangalore",
    "Chennai",
    "Hyderabad",
    "Pune"
]

device_types = [
    "Mobile",
    "Desktop",
    "Tablet"
]


# --------------------------------------------------
# DATASET GENERATOR
# --------------------------------------------------

def generate_dataset(
    month_name,
    year,
    month_number,
    records,
    add_anomalies=False,
    add_schema_drift=False,
    add_outliers=False
):

    start_date = datetime(
        year,
        month_number,
        1
    )

    if month_number == 12:
        next_month = datetime(
            year + 1,
            1,
            1
        )
    else:
        next_month = datetime(
            year,
            month_number + 1,
            1
        )

    number_of_days = (
        next_month - start_date
    ).days

    data = []

    for i in range(records):

        order_date = (
            start_date
            + timedelta(
                days=random.randint(
                    0,
                    number_of_days - 1
                )
            )
        )

        row = {
            "Order_ID":
                f"{month_name[:3].upper()}_{i + 1:06d}",

            "Customer_ID":
                f"CUST_{random.randint(1, 5000):05d}",

            "Product":
                random.choice(products),

            "Quantity":
                random.randint(1, 5),

            "Price":
                random.randint(200, 50000),

            "Order_Date":
                order_date.strftime("%Y-%m-%d"),

            "Payment_Method":
                random.choice(payment_methods),

            "Location":
                random.choice(locations)
        }

        # April contains an extra column
        # to simulate schema drift
        if add_schema_drift:

            row["Device_Type"] = (
                random.choice(device_types)
            )

        data.append(row)

    df = pd.DataFrame(data)

    # --------------------------------------------------
    # ADD APRIL DATA QUALITY PROBLEMS
    # --------------------------------------------------

    if add_anomalies:

        # Missing Payment_Method values
        payment_missing_count = int(
            len(df) * 0.12
        )

        payment_missing_indices = (
            np.random.choice(
                df.index,
                size=payment_missing_count,
                replace=False
            )
        )

        df.loc[
            payment_missing_indices,
            "Payment_Method"
        ] = np.nan

        # Missing Location values
        location_missing_count = int(
            len(df) * 0.07
        )

        location_missing_indices = (
            np.random.choice(
                df.index,
                size=location_missing_count,
                replace=False
            )
        )

        df.loc[
            location_missing_indices,
            "Location"
        ] = np.nan

    # --------------------------------------------------
    # ADD PRICE OUTLIERS
    # --------------------------------------------------

    if add_outliers:

        # Select 50 transactions and give them
        # unusually high prices.
        #
        # Normal price range:
        # ₹200 - ₹50,000
        #
        # Injected abnormal range:
        # ₹150,000 - ₹300,000

        outlier_count = 50

        outlier_indices = (
            np.random.choice(
                df.index,
                size=outlier_count,
                replace=False
            )
        )

        abnormal_prices = (
            np.random.randint(
                150000,
                300001,
                size=outlier_count
            )
        )

        df.loc[
            outlier_indices,
            "Price"
        ] = abnormal_prices

        print(
            f"{month_name}: "
            f"{outlier_count} abnormal "
            f"price values injected."
        )

    # --------------------------------------------------
    # ADD DUPLICATE RECORDS
    # --------------------------------------------------

    if add_anomalies:

        duplicate_count = 1500

        duplicate_rows = df.sample(
            n=duplicate_count,
            random_state=42
        )

        df = pd.concat(
            [
                df,
                duplicate_rows
            ],
            ignore_index=True
        )

    # --------------------------------------------------
    # SAVE DATASET
    # --------------------------------------------------

    file_path = (
        f"data/{month_name.lower()}.csv"
    )

    df.to_csv(
        file_path,
        index=False
    )

    print(
        f"{month_name}: "
        f"{len(df):,} records "
        f"saved to {file_path}"
    )

    print(
        f"Columns: "
        f"{list(df.columns)}"
    )

    print()


# --------------------------------------------------
# GENERATE MONTHLY DATASETS
# --------------------------------------------------

generate_dataset(
    month_name="January",
    year=2026,
    month_number=1,
    records=10000
)

generate_dataset(
    month_name="February",
    year=2026,
    month_number=2,
    records=10500
)

generate_dataset(
    month_name="March",
    year=2026,
    month_number=3,
    records=11000
)

# April deliberately contains:
#
# 1. Abnormal dataset growth
# 2. Missing values
# 3. Duplicate records
# 4. Schema drift
# 5. Abnormal price outliers

generate_dataset(
    month_name="April",
    year=2026,
    month_number=4,
    records=17000,
    add_anomalies=True,
    add_schema_drift=True,
    add_outliers=True
)

print("=" * 60)
print(
    "All monthly datasets generated successfully."
)
print("=" * 60)