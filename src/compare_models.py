
import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.base import clone
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier
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

from xgboost import XGBClassifier

# Import custom preprocessing classes
from feature_engineering import FeatureEngineer, IQRCapper


# --------------------------------------------------
# 1. LOAD DATASET
# --------------------------------------------------

print("\n========== LOADING DATASET ==========")

data = pd.read_csv("data/raw/Churn_Modelling.csv")

data = data.drop_duplicates()

data = data.drop(
    columns=["RowNumber", "CustomerId", "Surname"],
    errors="ignore"
)

X = data.drop("Exited", axis=1)
y = data["Exited"]

print("Dataset shape:", data.shape)
print("Features:", X.shape[1])
print("Target distribution:")
print(y.value_counts())


# --------------------------------------------------
# 2. SPLIT DATASET
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Create validation split from training data
X_fit, X_val, y_fit, y_val = train_test_split(
    X_train,
    y_train,
    test_size=0.20,
    random_state=42,
    stratify=y_train
)

print("\n========== DATA SPLITTING ==========")

print("Training subset:", X_fit.shape)
print("Validation subset:", X_val.shape)
print("Final training data:", X_train.shape)
print("Testing data:", X_test.shape)


# --------------------------------------------------
# 3. LOAD PREPROCESSOR
# --------------------------------------------------

print("\n========== PREPROCESSING ==========")

original_preprocessor = joblib.load(
    "models/preprocessor.pkl"
)

# Fit preprocessing only on fitting subset
validation_preprocessor = clone(original_preprocessor)

X_fit_processed = validation_preprocessor.fit_transform(
    X_fit
)

X_val_processed = validation_preprocessor.transform(
    X_val
)

print("Validation preprocessing completed!")


# --------------------------------------------------
# 4. DEFINE MODEL CREATION FUNCTION
# --------------------------------------------------

def create_models(y_values):

    scale_weight = (
        (y_values == 0).sum() /
        (y_values == 1).sum()
    )

    models = {

        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            random_state=42
        ),

        "Decision Tree": DecisionTreeClassifier(
            random_state=42
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            min_samples_split=5,
            min_samples_leaf=1,
            max_features="log2",
            max_depth=None,
            random_state=42,
            n_jobs=-1
        ),

        "Gradient Boosting": GradientBoostingClassifier(
            subsample=0.8,
            n_estimators=150,
            min_samples_split=2,
            min_samples_leaf=4,
            max_depth=5,
            learning_rate=0.05,
            random_state=42
        ),

        "XGBoost": XGBClassifier(
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
    }

    return models


# --------------------------------------------------
# 5. VALIDATION AND THRESHOLD OPTIMIZATION
# --------------------------------------------------

print("\n========== MODEL VALIDATION ==========")

validation_models = create_models(y_fit)

best_thresholds = {}
validation_scores = {}

for name, model in validation_models.items():

    print(f"\nTraining {name}...")

    model.fit(
        X_fit_processed,
        y_fit
    )

    probabilities = model.predict_proba(
        X_val_processed
    )[:, 1]

    best_f1 = -1
    best_threshold = 0.50

    for threshold in np.arange(0.20, 0.701, 0.02):

        predictions = (
            probabilities >= threshold
        ).astype(int)

        score = f1_score(
            y_val,
            predictions,
            zero_division=0
        )

        if score > best_f1:

            best_f1 = score

            best_threshold = round(
                float(threshold),
                2
            )

    best_thresholds[name] = best_threshold

    validation_scores[name] = best_f1

    print("Best threshold:", best_threshold)

    print(
        "Validation F1:",
        round(best_f1, 4)
    )


# --------------------------------------------------
# 6. DISPLAY VALIDATION COMPARISON
# --------------------------------------------------

print("\n========== VALIDATION MODEL COMPARISON ==========")

validation_comparison = pd.DataFrame({

    "Model": list(validation_scores.keys()),

    "Validation F1": list(validation_scores.values()),

    "Threshold": [
        best_thresholds[name]
        for name in validation_scores
    ]

})

validation_comparison = validation_comparison.sort_values(
    by="Validation F1",
    ascending=False
)

print(validation_comparison.to_string(index=False))


# --------------------------------------------------
# 7. FIT PREPROCESSOR ON COMPLETE TRAINING DATA
# --------------------------------------------------

print("\n========== FINAL PREPROCESSING ==========")

final_preprocessor = clone(original_preprocessor)

X_train_processed = final_preprocessor.fit_transform(
    X_train
)

X_test_processed = final_preprocessor.transform(
    X_test
)

print("Final preprocessing completed!")


# --------------------------------------------------
# 8. TRAIN AND EVALUATE ALL MODELS
# --------------------------------------------------

print("\n========== FINAL MODEL TRAINING ==========")

final_models = create_models(y_train)

results = []

for name, model in final_models.items():

    print(f"\nEvaluating {name}...")

    model.fit(
        X_train_processed,
        y_train
    )

    probabilities = model.predict_proba(
        X_test_processed
    )[:, 1]

    threshold = best_thresholds[name]

    predictions = (
        probabilities >= threshold
    ).astype(int)

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0
    )

    roc_auc = roc_auc_score(
        y_test,
        probabilities
    )

    results.append({

        "Model": name,

        "Threshold": threshold,

        "Accuracy": accuracy,

        "Precision": precision,

        "Recall": recall,

        "F1 Score": f1,

        "ROC-AUC": roc_auc

    })

    print("Threshold:", threshold)

    print(
        "Accuracy:",
        round(accuracy * 100, 2),
        "%"
    )

    print("Precision:", round(precision, 4))

    print("Recall:", round(recall, 4))

    print("F1 Score:", round(f1, 4))

    print("ROC-AUC:", round(roc_auc, 4))

    print("\nClassification Report:")

    print(
        classification_report(
            y_test,
            predictions,
            target_names=["Stayed", "Exited"],
            zero_division=0
        )
    )

    print("Confusion Matrix:")

    print(confusion_matrix(y_test, predictions))


# --------------------------------------------------
# 9. CREATE FINAL COMPARISON TABLE
# --------------------------------------------------

comparison = pd.DataFrame(results)

comparison = comparison.sort_values(
    by="F1 Score",
    ascending=False
)

print("\n========== FINAL MODEL COMPARISON ==========")

print(
    comparison.to_string(
        index=False,
        formatters={

            "Threshold": "{:.2f}".format,

            "Accuracy": "{:.4f}".format,

            "Precision": "{:.4f}".format,

            "Recall": "{:.4f}".format,

            "F1 Score": "{:.4f}".format,

            "ROC-AUC": "{:.4f}".format

        }
    )
)


# --------------------------------------------------
# 10. SAVE COMPARISON RESULTS
# --------------------------------------------------

os.makedirs("results", exist_ok=True)

os.makedirs("models", exist_ok=True)

comparison.to_csv(
    "results/model_comparison.csv",
    index=False
)

validation_comparison.to_csv(
    "results/validation_comparison.csv",
    index=False
)

print("\nComparison CSV files saved successfully!")


# --------------------------------------------------
# 11. GENERATE COMPARISON CHART
# --------------------------------------------------

metrics = [

    "Accuracy",

    "Precision",

    "Recall",

    "F1 Score",

    "ROC-AUC"

]

chart_data = (
    comparison.set_index("Model")[metrics] * 100
)

ax = chart_data.plot(
    kind="bar",
    figsize=(13, 7),
    width=0.8
)

plt.title(
    "Customer Churn Prediction - Model Comparison",
    fontsize=15
)

plt.xlabel("Machine Learning Models")

plt.ylabel("Score (%)")

plt.ylim(0, 100)

plt.xticks(
    rotation=20,
    ha="right"
)

plt.legend(
    title="Evaluation Metrics"
)

plt.grid(
    axis="y",
    linestyle="--",
    alpha=0.4
)

plt.tight_layout()

plt.savefig(
    "results/model_comparison.png",
    dpi=300
)

plt.show()

print("\nComparison chart saved successfully!")


# --------------------------------------------------
# 12. SELECT BEST MODEL USING VALIDATION F1
# --------------------------------------------------

print("\n========== BEST MODEL SELECTION ==========")

best_model_name = max(
    validation_scores,
    key=validation_scores.get
)

best_model_object = final_models[best_model_name]

best_threshold = best_thresholds[best_model_name]

print("Selected Model:", best_model_name)

print(
    "Validation F1 Score:",
    round(validation_scores[best_model_name], 4)
)

print("Selected Threshold:", best_threshold)


# --------------------------------------------------
# 13. SAVE BEST MODEL AND PREPROCESSOR
# --------------------------------------------------

joblib.dump(
    best_model_object,
    "models/best_model.pkl"
)

joblib.dump(
    final_preprocessor,
    "models/best_preprocessor.pkl"
)

with open(
    "models/best_threshold.json",
    "w"
) as file:

    json.dump(
        {
            "model": best_model_name,

            "threshold": best_threshold,

            "validation_f1": float(
                validation_scores[best_model_name]
            )

        },
        file,
        indent=4
    )


# --------------------------------------------------
# 14. DISPLAY SELECTED MODEL TEST RESULTS
# --------------------------------------------------

best_test_result = comparison[
    comparison["Model"] == best_model_name
].iloc[0]

print("\n========== SELECTED MODEL TEST RESULTS ==========")

print("Model:", best_model_name)

print(
    "Test Accuracy:",
    round(best_test_result["Accuracy"] * 100, 2),
    "%"
)

print(
    "Test Precision:",
    round(best_test_result["Precision"], 4)
)

print(
    "Test Recall:",
    round(best_test_result["Recall"], 4)
)

print(
    "Test F1 Score:",
    round(best_test_result["F1 Score"], 4)
)

print(
    "Test ROC-AUC:",
    round(best_test_result["ROC-AUC"], 4)
)

print("\n========== SAVED FILES ==========")

print("models/best_model.pkl")

print("models/best_preprocessor.pkl")

print("models/best_threshold.json")

print("results/model_comparison.csv")

print("results/validation_comparison.csv")

print("results/model_comparison.png")

print("\nMODEL COMPARISON COMPLETED SUCCESSFULLY!")
