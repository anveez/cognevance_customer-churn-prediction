
import os
import json
import joblib
import pandas as pd
import numpy as np

from xgboost import XGBClassifier

from sklearn.base import clone
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)

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


# 2. Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# 3. Load existing preprocessing pipeline
original_preprocessor = joblib.load("models/preprocessor.pkl")


# 4. Split training data into training and validation
X_fit, X_val, y_fit, y_val = train_test_split(
    X_train,
    y_train,
    test_size=0.20,
    random_state=42,
    stratify=y_train
)


# 5. Fit preprocessing only on the fitting subset
validation_preprocessor = clone(original_preprocessor)

X_fit_processed = validation_preprocessor.fit_transform(X_fit)
X_val_processed = validation_preprocessor.transform(X_val)


# 6. Calculate class imbalance ratio
scale_weight = (y_fit == 0).sum() / (y_fit == 1).sum()


# 7. Create XGBoost with the best parameters found earlier
validation_model = XGBClassifier(
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=42,
    n_jobs=-1,
    tree_method="hist",
    subsample=0.9,
    n_estimators=300,
    min_child_weight=1,
    max_depth=6,
    learning_rate=0.03,
    gamma=0.3,
    colsample_bytree=0.7,
    scale_pos_weight=scale_weight
)


# 8. Train validation model
print("\n========== XGBOOST VALIDATION TRAINING ==========")

validation_model.fit(
    X_fit_processed,
    y_fit
)


# 9. Get validation probabilities
val_probabilities = validation_model.predict_proba(
    X_val_processed
)[:, 1]


# 10. Find the best threshold
print("\n========== THRESHOLD COMPARISON ==========")

best_threshold = 0.50
best_f1 = 0

for threshold in np.arange(0.20, 0.701, 0.02):

    predictions = (
        val_probabilities >= threshold
    ).astype(int)

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
        best_threshold = round(float(threshold), 2)


print("\n========== BEST THRESHOLD ==========")
print("Best Threshold:", best_threshold)
print("Validation F1 Score:", round(best_f1, 4))


# 11. Retrain preprocessing on complete training data
final_preprocessor = clone(original_preprocessor)

X_train_processed = final_preprocessor.fit_transform(X_train)
X_test_processed = final_preprocessor.transform(X_test)


# 12. Create final XGBoost model
final_scale_weight = (
    (y_train == 0).sum() / (y_train == 1).sum()
)

final_model = XGBClassifier(
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=42,
    n_jobs=-1,
    tree_method="hist",
    subsample=0.9,
    n_estimators=300,
    min_child_weight=1,
    max_depth=6,
    learning_rate=0.03,
    gamma=0.3,
    colsample_bytree=0.7,
    scale_pos_weight=final_scale_weight
)


# 13. Train final model
print("\n========== FINAL XGBOOST TRAINING ==========")

final_model.fit(
    X_train_processed,
    y_train
)


# 14. Predict test probabilities
test_probabilities = final_model.predict_proba(
    X_test_processed
)[:, 1]

test_predictions = (
    test_probabilities >= best_threshold
).astype(int)


# 15. Evaluate final model
print("\n========== FINAL XGBOOST RESULTS ==========")

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


# 16. Classification report
print("\n========== CLASSIFICATION REPORT ==========")

print(
    classification_report(
        y_test,
        test_predictions,
        target_names=["Stayed", "Exited"],
        zero_division=0
    )
)


# 17. Confusion matrix
print("\n========== CONFUSION MATRIX ==========")

print(confusion_matrix(y_test, test_predictions))


# 18. Save final model and preprocessing separately
os.makedirs("models", exist_ok=True)

joblib.dump(
    final_model,
    "models/xgboost_model.pkl"
)

joblib.dump(
    final_preprocessor,
    "models/xgboost_preprocessor.pkl"
)

with open("models/xgboost_threshold.json", "w") as file:
    json.dump(
        {"threshold": best_threshold},
        file,
        indent=4
    )

print("\nXGBoost threshold optimization completed!")
print("Model: models/xgboost_model.pkl")
print("Preprocessor: models/xgboost_preprocessor.pkl")
print("Threshold: models/xgboost_threshold.json")
