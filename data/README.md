# Dataset Documentation

## 1. Dataset Source

The datasets are from the public Optical Failure Dataset repository:

https://github.com/Network-And-Services/optical-failure-dataset

The dataset was collected from an optical-network testbed. It contains measurements from optical devices and amplifiers under normal and emulated failure conditions.

**Note:** This is a public dataset selected for our project, not the original dataset used in the assigned research paper.

## 2. Original Datasets

* `HardFailure_dataset.csv`
* `SoftFailure_dataset.csv`

## 3. Cleaned Datasets

The cleaned datasets are stored in the `cleaned/` folder.

* `HardFailure_dataset_cleaned.csv` — 65,733 rows
* `SoftFailure_dataset_cleaned.csv` — 53,697 rows

Cleaning steps:

* Checked missing values and duplicate rows.
* Preserved the original timestamps and equipment identifiers.
* Converted missing `Failure` labels to 0 and retained failure labels as 1.
* Preserved expected missing measurement values because different equipment types record different measurements.
* Kept the original CSV files unchanged.

**Label assumption:** A blank `Failure` value is treated as 0 (no recorded failure condition). This assumption should be documented and verified against the dataset description before modelling.

## 4. Prepared Datasets

The prepared datasets are stored in the `prepared/` folder.

| File                                     |   Rows | Measurements            |
| ---------------------------------------- | -----: | ----------------------- |
| `HardFailure_dataset_infrastructure.csv` | 43,822 | InputPower, OutputPower |
| `HardFailure_dataset_devices.csv`        | 21,911 | BER, OSNR               |
| `SoftFailure_dataset_infrastructure.csv` | 35,798 | InputPower, OutputPower |
| `SoftFailure_dataset_devices.csv`        | 17,899 | BER, OSNR               |

Each prepared file retains the timestamp, equipment ID, relevant measurement columns, and failure label.

## 5. Important Modelling Note

The `Failure` column describes recorded failure conditions. A model trained directly on these labels would initially perform failure-condition classification, not necessarily future failure prediction. Future prediction requires a clearly defined prediction horizon and appropriately shifted labels.

## 6. Validation Results

* No duplicate rows were found in the original datasets.
* No missing timestamps, equipment types, or equipment IDs were found.
* All cleaned failure labels are 0 or 1.
* All rows match the expected measurement pattern for their equipment type.
