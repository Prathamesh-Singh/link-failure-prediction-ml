
import pandas as pd
import numpy as np
from pathlib import Path

input_folder = Path("cleaned")
output_folder = Path("reduced")
output_folder.mkdir(parents=True, exist_ok=True)

# Combine the hard- and soft-failure datasets.
hard = pd.read_csv(input_folder / "HardFailure_dataset_cleaned.csv")
soft = pd.read_csv(input_folder / "SoftFailure_dataset_cleaned.csv")

df = pd.concat([hard, soft], ignore_index=True)
df = df.sort_values(["Timestamp", "ID"]).reset_index(drop=True)

# Select a reproducible sample of 6,000 rows, preserving
# the approximate proportion of failure and normal labels.
sampled_parts = []

for label in [0, 1]:
    group = df[df["Failure"] == label]
    proportion = len(group) / len(df)
    target = round(6000 * proportion)
    target = min(target, len(group))

    sampled_parts.append(
        group.sample(n=target, random_state=42)
    )

reduced = pd.concat(sampled_parts, ignore_index=True)
reduced = reduced.sort_values(["Timestamp", "ID"]).reset_index(drop=True)

# Split by timestamp groups so the same timestamp cannot
# appear in both training and testing data.
timestamps = np.array(sorted(reduced["Timestamp"].unique()))
rng = np.random.default_rng(42)
rng.shuffle(timestamps)

target_test_rows = round(len(reduced) * 0.25)
test_times = set()
test_rows = 0

for timestamp in timestamps:
    if test_rows >= target_test_rows:
        break
    test_times.add(timestamp)
    test_rows += int((reduced["Timestamp"] == timestamp).sum())

test = reduced[reduced["Timestamp"].isin(test_times)].copy()
train = reduced[~reduced["Timestamp"].isin(test_times)].copy()

# Save the reduced dataset and the train/test splits.
reduced.to_csv(output_folder / "combined_reduced.csv", index=False)
train.to_csv(output_folder / "train.csv", index=False)
test.to_csv(output_folder / "test.csv", index=False)

print("Original combined rows:", len(df))
print("Reduced rows:", len(reduced))
print("Training rows:", len(train))
print("Testing rows:", len(test))

print("\nFailure counts in reduced dataset:")
print(reduced["Failure"].value_counts().sort_index())

print("\nFailure counts in training data:")
print(train["Failure"].value_counts().sort_index())

print("\nFailure counts in testing data:")
print(test["Failure"].value_counts().sort_index())

print("\nTimestamp overlap between train and test:",
      len(set(train["Timestamp"]) & set(test["Timestamp"])))

print("\nFiles saved in:", output_folder)