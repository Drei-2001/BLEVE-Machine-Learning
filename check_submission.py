import pandas as pd
import numpy as np

# File names
prediction_file = "sample_prediction.csv"

# Expected column names
required_columns = ["ID", "Target Pressure (bar)"]

try:
    # Load prediction file
    df = pd.read_csv(prediction_file)

    print("\n--- Checking Submission File ---\n")

    # 1. Check if file is empty
    if df.empty:
        print("ERROR: The submission file is empty.")
    else:
        print(f"Rows found: {len(df)}")

    # 2. Check required columns
    missing_cols = [col for col in required_columns if col not in df.columns]

    if missing_cols:
        print(f"ERROR: Missing columns: {missing_cols}")
    else:
        print("Column names are correct.")

    # 3. Check for duplicate IDs
    duplicate_ids = df["ID"].duplicated().sum()

    if duplicate_ids > 0:
        print(f"WARNING: Found {duplicate_ids} duplicate ID(s).")
    else:
        print("No duplicate IDs found.")

    # 4. Check for missing values
    missing_values = df.isnull().sum()

    if missing_values.sum() > 0:
        print("\nWARNING: Missing values detected:")
        print(missing_values[missing_values > 0])
    else:
        print("No missing values found.")

    # 5. Check if predictions are numeric
    if not pd.api.types.is_numeric_dtype(df["Target Pressure (bar)"]):
        print("ERROR: Target Pressure column is not numeric.")
    else:
        print("Predictions are numeric.")

    # 6. Check for invalid prediction values
    invalid_predictions = np.isinf(df["Target Pressure (bar)"]).sum()

    if invalid_predictions > 0:
        print(f"WARNING: Found {invalid_predictions} infinite values.")
    else:
        print("No infinite values found.")

    # 7. Check for negative pressures
    negative_pressures = (df["Target Pressure (bar)"] < 0).sum()

    if negative_pressures > 0:
        print(f"WARNING: Found {negative_pressures} negative pressure values.")
    else:
        print("No negative pressure values found.")

    # 8. Basic statistics
    print("\n--- Prediction Statistics ---")
    stats = df["Target Pressure (bar)"].describe()

    # Remove quartiles
    stats = stats.drop(["25%", "50%", "75%"])

    print(stats)
    print("\nSubmission file check complete.")

except FileNotFoundError:
    print(f"ERROR: '{prediction_file}' was not found.")

except Exception as e:
    print(f"Unexpected error: {e}")