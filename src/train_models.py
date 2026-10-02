import pandas as pd
from pathlib import Path
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import joblib

project_root = Path(__file__).resolve().parent.parent

train_file = project_root / "data" / "split" / "train.csv"

df = pd.read_csv(train_file)

print("Training rows:", len(df))
print(df.head())

feature_columns = [
    "AmplifierGain",
    "BER",
    "OSNR",
    "DeviceDataMissing"
]

X = df[feature_columns].copy()
y = df["Failure"]

print(X.head())
print(y.head())

if X["BER"].isna().all():
    raise ValueError("BER is completely missing in the training data.")

if X["OSNR"].isna().all():
    raise ValueError("OSNR is completely missing in the training data.")

if X["AmplifierGain"].isna().all():
    raise ValueError("AmplifierGain is completely missing in the training data.")

if X["DeviceDataMissing"].isna().all():
    raise ValueError("DeviceDataMissing is completely missing in the training data.")

imputer = SimpleImputer(strategy="median")

X_imputed = imputer.fit_transform(X)

print("Missing values after imputation:", pd.isna(X_imputed).sum())

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X_imputed)

print(X_scaled[:5])

#Logistic Regression
logistic_model = LogisticRegression(
    class_weight="balanced",
    random_state=42
)

logistic_model.fit(X_scaled, y)
print("Logistic Regression trained successfully")

# SVM
svm_model = SVC(
    kernel="linear",
    class_weight="balanced",
    random_state=42
)

svm_model.fit(X_scaled, y)
print("Linear SVM trained successfully")

# LDA
lda_model = LinearDiscriminantAnalysis()

lda_model.fit(X_scaled, y)

print("LDA model trained successfully")



models_folder = project_root / "models"
models_folder.mkdir(parents=True, exist_ok=True)

joblib.dump(
    logistic_model,
    models_folder / "logistic_model.pkl"
)

joblib.dump(
    svm_model,
    models_folder / "svm_model.pkl"
)

joblib.dump(
    lda_model,
    models_folder / "lda_model.pkl"
)

joblib.dump(
    imputer,
    models_folder / "imputer.pkl"
)

joblib.dump(
    scaler,
    models_folder / "scaler.pkl"
)

print("Models and preprocessing objects saved successfully")