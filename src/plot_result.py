

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)


ROOT = Path(__file__).resolve().parent.parent
DATA = pd.read_csv(ROOT / "data" / "split" / "test.csv")
MODELS = ROOT / "models"
OUTPUT = ROOT / "results" / "plots"
OUTPUT.mkdir(parents=True, exist_ok=True)

# Remove confusion-matrix images from the earlier version of this script.
for old_name in (
    "confusion_matrix_logistic.png",
    "confusion_matrix_svm.png",
    "confusion_matrix_lda.png",
):
    old_file = OUTPUT / old_name
    if old_file.exists():
        old_file.unlink()

features = ["AmplifierGain", "BER", "OSNR", "DeviceDataMissing"]
X = DATA[features]
y = DATA["Failure"]

imputer = joblib.load(MODELS / "imputer.pkl")
scaler = joblib.load(MODELS / "scaler.pkl")
X = scaler.transform(imputer.transform(X))

model_files = {
    "Logistic Regression": "logistic_model.pkl",
    "Linear SVM": "svm_model.pkl",
    "LDA": "lda_model.pkl",
}

results = []

for name, filename in model_files.items():
    model = joblib.load(MODELS / filename)
    predictions = model.predict(X)

    ConfusionMatrixDisplay.from_predictions(
        y,
        predictions,
        labels=[0, 1],
        display_labels=["Normal", "Failure"],
        cmap="Blues",
        values_format="d",
    )
    plt.title(f"{name} Confusion Matrix")
    plt.tight_layout()
    plt.savefig(OUTPUT / f"{name.lower().replace(' ', '_')}_confusion_matrix.png", dpi=200)
    plt.close()

    results.append(
        {
            "Model": name,
            "Accuracy": accuracy_score(y, predictions),
            "Precision": precision_score(y, predictions, zero_division=0),
            "Recall": recall_score(y, predictions, zero_division=0),
            "F1 score": f1_score(y, predictions, zero_division=0),
        }
    )

metrics = pd.DataFrame(results).set_index("Model")
metrics.plot(kind="bar", ylim=(0, 1), figsize=(9, 5), rot=0)
plt.title("Model Performance")
plt.ylabel("Score")
plt.tight_layout()
plt.savefig(OUTPUT / "model_metrics_comparison.png", dpi=200)
plt.close()

print(f"Plots saved in: {OUTPUT}")
print(metrics.round(3))
