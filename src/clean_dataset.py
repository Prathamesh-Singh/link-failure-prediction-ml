import pandas as pd
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
input_file = (
    project_root
    / "data"
    / "features"
    / "combined_reduced_features.csv"
)

output_folder = project_root / "data" / "cleaned_features"

output_folder.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(input_file)

print("Original rows:", len(df))

df["Failure"] = pd.to_numeric(
    df["Failure"],
    errors="coerce"
)

df = df[df["Failure"].isin([0, 1])]

df["Failure"] = df["Failure"].astype(int)

df = df.drop_duplicates()

df["TimeWindow"] = pd.to_datetime(
    df["TimeWindow"],
    errors="coerce"
)

df = df.dropna(subset=["TimeWindow"])

df = df.sort_values("TimeWindow").reset_index(drop=True)

df["DeviceDataMissing"] = df["BER"].isna().astype(int)

print("\nMissing values:")
print(df.isna().sum())

print("\nFailure counts:")
print(
    df["Failure"]
    .value_counts()
    .sort_index()
)

print("\nFailure counts where device data is missing:")
print(
    df[df["DeviceDataMissing"] == 1]["Failure"]
    .value_counts()
    .sort_index()
)

output_file = (
    output_folder
    / "combined_features_cleaned.csv"
)

df.to_csv(
    output_file,
    index=False
)

print("\nCleaned rows:", len(df))
print("Saved:", output_file)