
import pandas as pd
from pathlib import Path

folder = Path("reduced")

for filename in ["combined_reduced.csv", "train.csv", "test.csv"]:
    df = pd.read_csv(folder / filename)

    print("\n" + "=" * 45)
    print("FILE:", filename)
    print("Rows:", len(df))
    print("Missing values in Failure:", df["Failure"].isna().sum())
    print("Invalid labels:", (~df["Failure"].isin([0, 1])).sum())
    print("Duplicate rows:", df.duplicated().sum())
    print("Failure counts:")
    print(df["Failure"].value_counts().sort_index())

train = pd.read_csv(folder / "train.csv")
test = pd.read_csv(folder / "test.csv")

print("\nTrain/test timestamp overlap:",
      len(set(train["Timestamp"]) & set(test["Timestamp"])))

print("\nValidation completed.")
