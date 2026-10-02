import pandas as pd
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent

input_file = (
    project_root
    / "data"
    / "cleaned_features"
    / "combined_features_cleaned.csv"
)

output_folder = project_root / "data" / "split"
output_folder.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(input_file)

df["TimeWindow"] = pd.to_datetime(
    df["TimeWindow"],
    errors="coerce"
)

df = df.dropna(subset=["TimeWindow"])

df = df.sort_values("TimeWindow").reset_index(drop=True)

unique_windows = df["TimeWindow"].drop_duplicates().sort_values()

if len(unique_windows) < 2:
    raise ValueError(
        "Not enough distinct TimeWindow values to create train/test split."
    )

split_index = int(len(unique_windows) * 0.75)

split_index = max(
    1,
    min(split_index, len(unique_windows) - 1)
)

split_time = unique_windows.iloc[split_index]

train = df[df["TimeWindow"] < split_time].copy()
test = df[df["TimeWindow"] >= split_time].copy()

train.to_csv(
    output_folder / "train.csv",
    index=False
)

test.to_csv(
    output_folder / "test.csv",
    index=False
)

print("Total rows:", len(df))
print("Training rows:", len(train))
print("Testing rows:", len(test))

print("\nTraining failure counts:")
print(train["Failure"].value_counts().sort_index())

print("\nTesting failure counts:")
print(test["Failure"].value_counts().sort_index())

overlap = set(train["TimeWindow"]) & set(test["TimeWindow"])

print("\nTimeWindow overlap:", len(overlap))