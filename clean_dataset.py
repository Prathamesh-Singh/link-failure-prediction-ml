
import pandas as pd
from pathlib import Path

data_folder = Path("data")
cleaned_folder = data_folder / "cleaned"
cleaned_folder.mkdir(parents=True, exist_ok=True)

files = [
    "HardFailure_dataset.csv",
    "SoftFailure_dataset.csv"
]

for filename in files:
    input_path = data_folder / filename
    df = pd.read_csv(input_path)

    print("\n" + "=" * 55)
    print("Cleaning:", filename)

    original_rows = len(df)

    # Convert blank failure labels to 0 and keep failure labels as 1.
    df["Failure"] = df["Failure"].fillna(0).astype(int)

    # Ensure timestamps are stored as integers.
    df["Timestamp"] = df["Timestamp"].astype("int64")

    # Keep rows with valid identifiers and timestamps.
    df = df.dropna(subset=["Timestamp", "Type", "ID"])

    # Save a separate cleaned copy; never overwrite the original.
    output_name = filename.replace(".csv", "_cleaned.csv")
    output_path = cleaned_folder / output_name
    df.to_csv(output_path, index=False)

    print("Original rows:", original_rows)
    print("Cleaned rows:", len(df))
    print("Rows removed:", original_rows - len(df))
    print("Remaining missing values:")
    print(df.isna().sum())
    print("Failure counts:")
    print(df["Failure"].value_counts().sort_index())
    print("Saved to:", output_path)