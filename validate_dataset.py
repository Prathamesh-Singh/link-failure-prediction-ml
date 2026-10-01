
import pandas as pd
from pathlib import Path

cleaned_folder = Path("data") / "cleaned"

files = [
    "HardFailure_dataset_cleaned.csv",
    "SoftFailure_dataset_cleaned.csv"
]

for filename in files:
    path = cleaned_folder / filename
    df = pd.read_csv(path)

    print("\n" + "=" * 55)
    print("VALIDATING:", filename)

    print("Total rows:", len(df))
    print("Total columns:", len(df.columns))
    print("Columns:", list(df.columns))

    print("\nMissing values:")
    print(df.isna().sum())

    print("\nFailure label counts:")
    print(df["Failure"].value_counts().sort_index())

    print("\nDuplicate rows:", df.duplicated().sum())

    print("\nInvalid failure labels:",
          (~df["Failure"].isin([0, 1])).sum())

    print("\nRows by equipment type:")
    print(df["Type"].value_counts())

    print("\nRows by equipment ID:")
    print(df["ID"].value_counts())

    print("\nTimestamp range:")
    print("Minimum:", df["Timestamp"].min())
    print("Maximum:", df["Timestamp"].max())

    expected_measurements = (
        df["Type"].eq("Infrastructure")
        & df["BER"].isna()
        & df["OSNR"].isna()
        & df["InputPower"].notna()
        & df["OutputPower"].notna()
    ) | (
        df["Type"].eq("Devices")
        & df["BER"].notna()
        & df["OSNR"].notna()
        & df["InputPower"].isna()
        & df["OutputPower"].isna()
    )

    print("\nRows matching expected measurement pattern:",
          expected_measurements.sum())
    print("Rows not matching expected measurement pattern:",
          (~expected_measurements).sum())