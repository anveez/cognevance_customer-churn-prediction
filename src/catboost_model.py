
import os
import json
import joblib
import numpy as np
import pandas as pd

from catboost import CatBoostClassifier
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

os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)

DATA_PATH = "data/raw/Churn_Modelling.csv"

print("\n========== IMPROVED CATBOOST EXPERIMENT ==========")

# --------------------------------------------------
# 1. LOAD DATA
# --------------------------------------------------

df = pd.read_csv(DATA_PATH)
df.drop_duplicates(inplace=True)

X = df.drop(
    columns=["Exited", "RowNumber", "CustomerId", "Surname"],
    errors="ignore"
).copy()

y = df["Exited"].copy()

# --------------------------------------------------
# 2. FEATURE ENGINEERING
# --------------------------------------------------

print("\nCreating additional features...")

X["BalanceSalaryRatio"] = (
    X["Balance"] / (X["EstimatedSalary"] + 1)
)

X["BalancePerProduct"] = (
    X["Balance"] / X["NumOfProducts"].replace(0, 1)
)

X["SalaryPerProduct"] = (
    X["EstimatedSalary"] / X["NumOfProducts"].replace(0, 1)
)

X["AgeTenureRatio"] = (
    X["Age"] / (X["Tenure"] + 1)
)

X["CustomerValue"] = (
    X["Balance"] * X["IsActiveMember"]
)

X["ProductsActive"] = (
    X["NumOfProducts"] * X["IsActiveMember"]
)

X["AgeGroup"] = pd.cut(
    X["Age"],
    bins=[0, 25, 35, 45, 55, 100],
    labels=["Young", "Adult", "Middle", "Senior", "Older"]
).astype(str)

X["TenureGroup"] = pd.cut(
    X["Tenure"],
    bins=[-1, 2, 5, 8, 10],
    labels=["New", "Medium", "Long", "VeryLong"]
).astype(str)

X.replace([np.inf, -np.inf], np.nan, inplace=True)

categorical_features = list(
    X.select_dtypes(include=["object", "category"]).columns
)

for column in categorical_features:
    X[column] = X[column].fillna("Unknown").astype(str)

for column in X.select_dtypes(include=np.number).columns:
    X[column] = X[column].fillna(X[column].median())

print("Total features:", X.shape[1])
print("Categorical features:", categorical_features)

# --------------------------------------------------
# 3. DATA SPLIT
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

X_fit, X_val, y_fit, y_val = train_test_split(
    X_train,
    y_train,
    test_size=0.20,
    random_state=42,
    stratify=y_train
)

print("\nFit:", len(X_fit))
print("Validation:", len(X_val))
print("Test:", len(X_test))

# --------------------------------------------------
# 4. CATBOOST EXPERIMENTS
# --------------------------------------------------

print("\n========== TRAINING MULTIPLE CONFIGURATIONS ==========")

configs = [
    {"depth": 4, "learning_rate": 0.03, "l2_leaf_reg": 5},
    {"depth": 5, "learning_rate": 0.03, "l2_leaf_reg": 7},
    {"depth": 6, "learning_rate": 0.03, "l2_leaf_reg": 7},
    {"depth": 7, "learning_rate": 0.025, "l2_leaf_reg": 9},
    {"depth": 8, "learning_rate": 0.02, "l2_leaf_reg": 9},
    {"depth": 6, "learning_rate": 0.05, "l2_leaf_reg": 5},
    {"depth": 7, "learning_rate": 0.04, "l2_leaf_reg": 5},
    {"depth": 5, "learning_rate": 0.05, "l2_leaf_reg": 3}
]

best_model = None
best_config = None
best_auc = -1

search_results = []

for i, config in enumerate(configs, start=1):

    print(f"\nConfiguration {i}/{len(configs)}:", config)

    model = CatBoostClassifier(
        iterations=2000,
        depth=config["depth"],
        learning_rate=config["learning_rate"],
        l2_leaf_reg=config["l2_leaf_reg"],
        loss_function="Logloss",
        eval_metric="AUC",
        random_seed=42,
        verbose=False,
        allow_writing_files=False
    )

    model.fit(
        X_fit,
        y_fit,
        cat_features=categorical_features,
        eval_set=(X_val, y_val),
        early_stopping_rounds=150,
        verbose=False
    )

    probabilities = model.predict_proba(X_val)[:, 1]

    auc = roc_auc_score(y_val, probabilities)

    search_results.append({
        "depth": config["depth"],
        "learning_rate": config["learning_rate"],
        "l2_leaf_reg": config["l2_leaf_reg"],
        "validation_auc": auc,
        "best_iteration": model.best_iteration_
    })

    print("Validation AUC:", round(auc, 5))
    print("Best iteration:", model.best_iteration_)

    if auc > best_auc:
        best_auc = auc
        best_model = model
        best_config = config

pd.DataFrame(search_results).to_csv(
    "results/catboost_native_search.csv",
    index=False
)

print("\nBest configuration:", best_config)
print("Best validation AUC:", best_auc)

# --------------------------------------------------
# 5. THRESHOLD OPTIMIZATION
# --------------------------------------------------

print("\n========== THRESHOLD OPTIMIZATION ==========")

val_probabilities = best_model.predict_proba(X_val)[:, 1]

threshold_results = []

for threshold in np.arange(0.20, 0.81, 0.01):

    predictions = (
        val_probabilities >= threshold
    ).astype(int)

    threshold_results.append({
        "threshold": round(float(threshold), 2),
        "accuracy": accuracy_score(y_val, predictions),
        "precision": precision_score(
            y_val, predictions, zero_division=0
        ),
        "recall": recall_score(
            y_val, predictions, zero_division=0
        ),
        "f1": f1_score(
            y_val, predictions, zero_division=0
        )
    })

threshold_df = pd.DataFrame(threshold_results)

# Prefer a threshold with at least 50% churn recall.
eligible = threshold_df[threshold_df["recall"] >= 0.50]

if not eligible.empty:
    threshold_df = eligible

threshold_df = threshold_df.sort_values(
    ["accuracy", "f1"],
    ascending=[False, False]
)

best_threshold = float(threshold_df.iloc[0]["threshold"])

print(threshold_df.head(10).to_string(index=False))
print("\nSelected threshold:", best_threshold)

# --------------------------------------------------
# 6. FINAL TRAINING
# --------------------------------------------------

print("\n========== FINAL TRAINING ==========")

final_iterations = max(100, best_model.best_iteration_ + 1)

final_model = CatBoostClassifier(
    iterations=final_iterations,
    depth=best_config["depth"],
    learning_rate=best_config["learning_rate"],
    l2_leaf_reg=best_config["l2_leaf_reg"],
    loss_function="Logloss",
    eval_metric="AUC",
    random_seed=42,
    verbose=False,
    allow_writing_files=False
)

final_model.fit(
    X_train,
    y_train,
    cat_features=categorical_features,
    verbose=False
)

# --------------------------------------------------
# 7. TEST EVALUATION
# --------------------------------------------------

print("\n========== TEST RESULTS ==========")

test_probabilities = final_model.predict_proba(X_test)[:, 1]

test_predictions = (
    test_probabilities >= best_threshold
).astype(int)

accuracy = accuracy_score(y_test, test_predictions)
precision = precision_score(y_test, test_predictions, zero_division=0)
recall = recall_score(y_test, test_predictions, zero_division=0)
f1 = f1_score(y_test, test_predictions, zero_division=0)
auc = roc_auc_score(y_test, test_probabilities)

print("Threshold:", best_threshold)
print(f"Accuracy:  {accuracy * 100:.2f}%")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")
print(f"ROC-AUC:   {auc:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, test_predictions))

print("Confusion Matrix:")
print(confusion_matrix(y_test, test_predictions))

# --------------------------------------------------
# 8. SAVE SEPARATELY
# --------------------------------------------------

joblib.dump(
    final_model,
    "models/catboost_native_model.pkl"
)

with open("models/catboost_native_config.json", "w") as f:
    json.dump({
        "model": "CatBoost Native Features",
        "threshold": best_threshold,
        "validation_auc": float(best_auc),
        "parameters": best_config,
        "categorical_features": categorical_features
    }, f, indent=4)

pd.DataFrame([{
    "Model": "CatBoost Native Features",
    "Threshold": best_threshold,
    "Accuracy": accuracy,
    "Precision": precision,
    "Recall": recall,
    "F1 Score": f1,
    "ROC-AUC": auc
}]).to_csv(
    "results/catboost_native_results.csv",
    index=False
)

print("\nExperiment completed successfully!")
print("Saved: models/catboost_native_model.pkl")
print("Saved: models/catboost_native_config.json")
print("Saved: results/catboost_native_results.csv")
