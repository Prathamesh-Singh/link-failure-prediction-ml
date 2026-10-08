# Link Failure Prediction Using Machine Learning

## 1. Project Overview

This project explores the use of supervised machine learning techniques for link failure prediction and classification in network systems.

The project is inspired by the research paper **“Link Failure Prediction and Localization in Cloud Scale Networks using Supervised Learning”** by Zahra Bakhtiari. We use a related public optical-failure dataset because the paper's original dataset is not available to us. Its measurements and features differ, so our results are not a direct reproduction of the paper.

The objective is to use available network measurements to classify recorded link failure conditions using machine learning models. The current project classifies a failure in the same time window; it does not yet predict a future failure.

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

The paper uses Gaussian Discriminant Analysis (GDA); this project implements scikit-learn's Linear Discriminant Analysis (LDA) as its discriminant-analysis model.

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
│   ├── evaluate_model.py
│   └── plot_result.py
├── models/                 # Generated model files (ignored by Git)
├── results/plots/          # Generated evaluation plots
└── README.md
```

*Prepared train/test CSV files are in `data/split/`. Model files and plots are generated when the scripts run.*

## 5. Technologies Used

* Python
* Pandas
* Scikit-learn
* Joblib
* Matplotlib
* Git and GitHub

## 6. Installation

Install the required Python libraries using:

```bash
python -m pip install pandas scikit-learn joblib matplotlib
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

### Generate plots

```bash
python src/plot_result.py
```

This saves one confusion-matrix image per model and a model-metrics comparison chart in `results/plots/`.

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
