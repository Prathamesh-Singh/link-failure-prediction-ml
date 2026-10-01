
import pandas as pd
from pathlib import Path

cleaned_folder = Path("data") / "cleaned"
prepared_folder = Path("data") / "prepared"
prepared_folder.mkdir(parents=True, exist_ok=True)

datasets = [
    "HardFailure_dataset_cleaned.csv",
    "SoftFailure_dataset_cleaned.csv"
]

for filename in datasets:
    df = pd.read_csv(cleaned_folder / filename)

    prefix = filename.replace("_cleaned.csv", "")

    # Infrastructure uses amplifier power measurements.
    infrastructure = df[df["Type"] == "Infrastructure"][
        ["Timestamp", "ID", "InputPower", "OutputPower", "Failure"]
    ].copy()

    # Devices use BER and OSNR measurements.
    devices = df[df["Type"] == "Devices"][
        ["Timestamp", "ID", "BER", "OSNR", "Failure"]
    ].copy()

    infrastructure.to_csv(
        prepared_folder / f"{prefix}_infrastructure.csv",
        index=False
    )

    devices.to_csv(
        prepared_folder / f"{prefix}_devices.csv",
        index=False
    )

    print("\nDataset:", prefix)
    print("Infrastructure rows:", len(infrastructure))
    print("Device rows:", len(devices))
    print("Infrastructure missing values:")
    print(infrastructure.isna().sum())
    print("Device missing values:")
    print(devices.isna().sum())

print("\nPrepared datasets saved successfully.")