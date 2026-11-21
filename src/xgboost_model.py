
import os
import joblib
import pandas as pd

from xgboost import XGBClassifier

from sklearn.model_selection import (
    train_test_split,
    RandomizedSearchCV
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)

# Import custom classes before loading saved preprocessing
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


# 2. Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# 3. Load preprocessing pipeline
preprocessor = joblib.load("models/preprocessor.pkl")

X_train_processed = preprocessor.transform(X_train)
X_test_processed = preprocessor.transform(X_test)


# 4. Calculate class imbalance ratio
negative_count = (y_train == 0).sum()
positive_count = (y_train == 1).sum()

scale_weight = negative_count / positive_count


# 5. Define XGBoost model
xgb_model = XGBClassifier(
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=42,
    n_jobs=-1,
    tree_method="hist",
    scale_pos_weight=scale_weight
)


# 6. Define hyperparameter combinations
parameters = {
    "n_estimators": [100, 200, 300, 400],
    "max_depth": [2, 3, 4, 5, 6],
    "learning_rate": [0.01, 0.03, 0.05, 0.1],
    "subsample": [0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
    "min_child_weight": [1, 3, 5, 7],
    "gamma": [0, 0.1, 0.2, 0.3]
}


# 7. Hyperparameter tuning
print("\n========== XGBOOST TRAINING ==========")

search = RandomizedSearchCV(
    estimator=xgb_model,
    param_distributions=parameters,
    n_iter=20,
    scoring="f1",
    cv=3,
    random_state=42,
    n_jobs=-1,
    verbose=1
)

search.fit(X_train_processed, y_train)


# 8. Get best XGBoost model
best_xgb_model = search.best_estimator_

print("\nBest parameters:")
print(search.best_params_)


# 9. Make predictions
predictions = best_xgb_model.predict(X_test_processed)

probabilities = best_xgb_model.predict_proba(
    X_test_processed
)[:, 1]


# 10. Evaluate model
accuracy = accuracy_score(y_test, predictions)
precision = precision_score(
    y_test, predictions, zero_division=0
)
recall = recall_score(
    y_test, predictions, zero_division=0
)
f1 = f1_score(
    y_test, predictions, zero_division=0
)
roc_auc = roc_auc_score(y_test, probabilities)


print("\n========== XGBOOST RESULTS ==========")

print("Accuracy:", round(accuracy * 100, 2), "%")
print("Precision:", round(precision, 4))
print("Recall:", round(recall, 4))
print("F1 Score:", round(f1, 4))
print("ROC-AUC:", round(roc_auc, 4))


# 11. Classification report
print("\n========== CLASSIFICATION REPORT ==========")

print(
    classification_report(
        y_test,
        predictions,
        target_names=["Stayed", "Exited"],
        zero_division=0
    )
)


# 12. Confusion matrix
print("\n========== CONFUSION MATRIX ==========")
print(confusion_matrix(y_test, predictions))


# 13. Save XGBoost candidate separately
os.makedirs("models", exist_ok=True)

joblib.dump(
    best_xgb_model,
    "models/xgboost_model.pkl"
)

print("\nXGBoost model saved successfully!")
print("Location: models/xgboost_model.pkl")
