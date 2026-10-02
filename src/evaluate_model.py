import pandas as pd
import joblib
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

project_root = Path(__file__).resolve().parent.parent

test_file = project_root / "data" / "split" / "test.csv"
models_folder = project_root / "models"

df = pd.read_csv(test_file)

print("Testing rows:", len(df))
print(df.head())

feature_columns = [
    "AmplifierGain",
    "BER",
    "OSNR",
    "DeviceDataMissing"
]

X_test = df[feature_columns].copy()
y_test = df["Failure"]

imputer = joblib.load(
    models_folder / "imputer.pkl"
)

scaler = joblib.load(
    models_folder / "scaler.pkl"
)

X_test_imputed = imputer.transform(X_test)

X_test_scaled = scaler.transform(X_test_imputed)

logistic_model = joblib.load(
    models_folder / "logistic_model.pkl"
)

svm_model = joblib.load(
    models_folder / "svm_model.pkl"
)

lda_model = joblib.load(
    models_folder / "lda_model.pkl"
)

logistic_pred = logistic_model.predict(X_test_scaled)
svm_pred = svm_model.predict(X_test_scaled)
lda_pred = lda_model.predict(X_test_scaled)  

def evaluate_model(name,y_true,y_pred):
    accuracy=accuracy_score(y_true,y_pred)
    precision=precision_score(y_true, y_pred, zero_division=0)
    recall=recall_score(y_true,y_pred,zero_division=0)
    f1=f1_score(y_true, y_pred,zero_division=0)
    
    print("\n", name)
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)

    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
    
evaluate_model(
    "Logistic Regression",
    y_test,
    logistic_pred
)

evaluate_model(
    "Linear SVM",
    y_test,
    svm_pred
)

evaluate_model(
    "LDA",
    y_test,
    lda_pred
)