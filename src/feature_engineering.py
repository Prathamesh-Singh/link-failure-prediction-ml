import pandas as pd
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent

input_folder = project_root / "data" / "reduced"
output_folder = project_root / "data" / "features"

output_folder.mkdir(parents=True , exist_ok=True)

def create_features(filename):
    df=pd.read_csv(input_folder/filename)
    print("Processing the file:",filename)
    print("Orignal rows:",len(df))
    
    df["Datetime"] = pd.to_datetime(df["Timestamp"], unit="s")
    df["TimeWindow"] = df["Datetime"].dt.floor("15s")
    
    infrastructure = df[df["Type"] == "Infrastructure"].copy()

    infrastructure["AmplifierGain"] = (
        infrastructure["OutputPower"]
        - infrastructure["InputPower"]
    )
    
    infrastructure_features = (
    infrastructure
    .groupby("TimeWindow")
    .agg(
        AmplifierGain=("AmplifierGain", "mean")
        )
        .reset_index()
    )
    
    devices = df[df["Type"] == "Devices"].copy()
    device_features = (
    devices
    .groupby("TimeWindow")
    .agg(
        BER=("BER", "mean"),
        OSNR=("OSNR", "mean")
        )
        .reset_index()
    )
    
    failure_labels = (
    df
    .groupby("TimeWindow")
    .agg(
        Failure=("Failure", "max")
        )
        .reset_index()
    )
    
    features = infrastructure_features.merge(
    device_features,
    on="TimeWindow",
    how="left"
    )
    
    features = features.merge(
    failure_labels,
    on="TimeWindow",
    how="left"
    )   
    
    print("\nMissing values after merge:")
    print(features.isna().sum())
    print("Final feature rows:", len(features))
    print(features.head())
    
    output_name = filename.replace(".csv", "_features.csv")

    features.to_csv(
    output_folder / output_name,
    index=False
    )
    print("Saved:", output_folder / output_name)
    print()
    
create_features("combined_reduced.csv")

    