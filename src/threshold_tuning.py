
import os
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

# Import custom classes before loading saved files
from feature_engineering import FeatureEngineer, IQRCapper


# 1. Load dataset
data = pd.read_csv("data/raw/Churn_Modelling.csv")
data = data.drop_duplicates()

data = data.drop(
    columns=["RowNumber", "CustomerId", "Surname"],
    errors="ignore"
)

X = data.drop("Exited", axis=1)
y = data["Exited"]


# 2. Create training and test datasets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# 3. Load saved preprocessing pipeline and best model
original_preprocessor = joblib.load("models/preprocessor.pkl")
original_model = joblib.load("models/best_model.pkl")


# 4. Create a separate validation dataset
X_fit, X_val, y_fit, y_val = train_test_split(
    X_train,
    y_train,
    test_size=0.20,
    random_state=42,
    stratify=y_train
)


# 5. Fit preprocessing only on the fitting portion
validation_preprocessor = clone(original_preprocessor)

X_fit_processed = validation_preprocessor.fit_transform(X_fit)
X_val_processed = validation_preprocessor.transform(X_val)


# 6. Train a validation model
validation_model = clone(original_model)
validation_model.fit(X_fit_processed, y_fit)

val_probabilities = validation_model.predict_proba(
    X_val_processed
)[:, 1]


# 7. Find the best decision threshold using validation data
best_threshold = 0.5
best_f1 = -1

print("\n========== THRESHOLD COMPARISON ==========")

for threshold in np.arange(0.20, 0.71, 0.02):

    predictions = (val_probabilities >= threshold).astype(int)

    accuracy = accuracy_score(y_val, predictions)
    precision = precision_score(
        y_val, predictions, zero_division=0
    )
    recall = recall_score(
        y_val, predictions, zero_division=0
    )
    f1 = f1_score(
        y_val, predictions, zero_division=0
    )

    print(
        f"Threshold: {threshold:.2f} | "
        f"Accuracy: {accuracy * 100:.2f}% | "
        f"Precision: {precision:.4f} | "
        f"Recall: {recall:.4f} | "
        f"F1: {f1:.4f}"
    )

    if f1 > best_f1:
        best_f1 = f1
        best_threshold = float(threshold)


print("\nBest validation threshold:", best_threshold)
print("Best validation F1-score:", round(best_f1, 4))


# 8. Refit preprocessing and model on all training data
final_preprocessor = clone(original_preprocessor)

X_train_processed = final_preprocessor.fit_transform(X_train)
X_test_processed = final_preprocessor.transform(X_test)

final_model = clone(original_model)
final_model.fit(X_train_processed, y_train)


# 9. Evaluate once on test data using selected threshold
test_probabilities = final_model.predict_proba(
    X_test_processed
)[:, 1]

test_predictions = (
    test_probabilities >= best_threshold
).astype(int)


print("\n========== FINAL TEST RESULTS ==========")

print(
    "Accuracy:",
    round(accuracy_score(y_test, test_predictions) * 100, 2),
    "%"
)

print(
    "Precision:",
    round(precision_score(y_test, test_predictions, zero_division=0), 4)
)

print(
    "Recall:",
    round(recall_score(y_test, test_predictions, zero_division=0), 4)
)

print(
    "F1 Score:",
    round(f1_score(y_test, test_predictions, zero_division=0), 4)
)

print(
    "ROC-AUC:",
    round(roc_auc_score(y_test, test_probabilities), 4)
)


print("\n========== CLASSIFICATION REPORT ==========")

print(
    classification_report(
        y_test,
        test_predictions,
        target_names=["Stayed", "Exited"],
        zero_division=0
    )
)

print("\n========== CONFUSION MATRIX ==========")
print(confusion_matrix(y_test, test_predictions))


# 10. Save final model, preprocessing pipeline and threshold
os.makedirs("models", exist_ok=True)

joblib.dump(final_model, "models/best_model.pkl")
joblib.dump(final_preprocessor, "models/preprocessor.pkl")

with open("models/threshold.json", "w") as file:
    json.dump({"threshold": best_threshold}, file, indent=4)

print("\nAll files saved successfully!")
print("Model: models/best_model.pkl")
print("Preprocessor: models/preprocessor.pkl")
print("Threshold: models/threshold.json")
