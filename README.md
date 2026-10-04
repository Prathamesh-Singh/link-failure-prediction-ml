# Link Failure Prediction Using Machine Learning

## 1. Project Overview

This project explores the use of supervised machine learning techniques for link failure prediction and classification in network systems.

The project is inspired by the research paper **“Link Failure Prediction and Localization in Cloud Scale Networks using Supervised Learning”** by Zahra Bakhtiari.

The objective is to use available network measurements to classify recorded link failure conditions using machine learning models.

## 2. Dataset

The project uses the Optical Failure Dataset available on GitHub:

https://github.com/Network-And-Services/optical-failure-dataset

The dataset contains network measurements and failure labels. The model training pipeline uses the following features:

* **AmplifierGain** – amplifier gain measurement
* **BER** – Bit Error Rate
* **OSNR** – Optical Signal-to-Noise Ratio
* **DeviceDataMissing** – indicates missing device measurement data
* **Failure** – target label for classification

Missing feature values are handled using median imputation.

## 3. Machine Learning Models

The following supervised learning models are implemented:

1. **Logistic Regression**
2. **Linear Support Vector Machine (SVM)**
3. **Linear Discriminant Analysis (LDA)**

The features are standardized using `StandardScaler` before model training.

## 4. Project Structure

```text
link-failure-prediction-ml/
├── data/
│   ├── README.md
│   └── split/
│       ├── train.csv
│       └── test.csv
├── src/
│   ├── clean_dataset.py
│   ├── split_dataset.py
│   ├── feature_engineering.py
│   ├── train_models.py
│   └── evaluate_model.py
├── models/
│   ├── logistic_model.pkl
│   ├── svm_model.pkl
│   ├── lda_model.pkl
│   ├── imputer.pkl
│   └── scaler.pkl
└── README.md
```

*Note: The structure above describes the intended organization of the project. Some generated files may not be included in the GitHub repository.*

## 5. Technologies Used

* Python
* Pandas
* Scikit-learn
* Joblib
* Git and GitHub

## 6. Installation

Install the required Python libraries using:

```bash
python -m pip install pandas scikit-learn joblib
```

## 7. Running the Project

Run the following commands from the project root directory.

### Train the models

```bash
python src/train_models.py
```

This script trains the Logistic Regression, Linear SVM and LDA models and saves the trained models and preprocessing objects in the `models/` directory.

### Evaluate the models

```bash
python src/evaluate_model.py
```

This script evaluates the trained models using the test dataset and reports classification metrics, including accuracy, precision, recall, F1-score and confusion matrix.

The training and evaluation scripts expect their prepared input datasets to be available in the paths specified in the scripts.

## 8. Evaluation Metrics

The models are evaluated using:

* **Accuracy:** Overall proportion of correct predictions.
* **Precision:** Proportion of predicted failures that are actual failures.
* **Recall:** Proportion of actual failures detected by the model.
* **F1-score:** Harmonic mean of precision and recall.
* **Confusion Matrix:** Shows correct and incorrect predictions for each class.

Since failure labels may be imbalanced, precision, recall and F1-score should be considered alongside accuracy.

## 9. Limitations

* Model performance depends on the quality and representativeness of the available dataset.
* Missing network measurements may affect classification performance.
* The current task classifies recorded failure labels; it should not automatically be interpreted as forecasting future failures.
* Further validation is required before applying the models to real-world network operations.

## 10. Conclusion

This project demonstrates a supervised machine learning workflow for classifying network link failure conditions using Logistic Regression, Linear SVM and Linear Discriminant Analysis. It includes data preparation, feature preprocessing, model training and evaluation.
