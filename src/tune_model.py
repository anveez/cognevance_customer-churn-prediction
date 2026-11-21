
import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report
)

# Import custom preprocessing classes before loading the pipeline
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


# 4. Define models and parameter combinations
searches = {
    "Random Forest": (
        RandomForestClassifier(random_state=42, n_jobs=-1),
        {
            "n_estimators": [100, 200, 300, 400],
            "max_depth": [None, 5, 10, 15, 20],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "max_features": ["sqrt", "log2"]
        }
    ),

    "Gradient Boosting": (
        GradientBoostingClassifier(random_state=42),
        {
            "n_estimators": [100, 150, 200, 300],
            "learning_rate": [0.01, 0.03, 0.05, 0.1],
            "max_depth": [2, 3, 4, 5],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "subsample": [0.8, 0.9, 1.0]
        }
    )
}


# 5. Tune and evaluate
results = []

best_model = None
best_model_name = ""
best_f1 = -1

os.makedirs("models", exist_ok=True)

for name, (model, parameters) in searches.items():

    print("\nTuning:", name)

    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=parameters,
        n_iter=20,
        scoring="f1",
        cv=3,
        random_state=42,
        n_jobs=-1,
        verbose=1
    )

    search.fit(X_train_processed, y_train)

    tuned_model = search.best_estimator_

    predictions = tuned_model.predict(X_test_processed)
    probabilities = tuned_model.predict_proba(X_test_processed)[:, 1]

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, zero_division=0)
    recall = recall_score(y_test, predictions, zero_division=0)
    f1 = f1_score(y_test, predictions, zero_division=0)
    roc_auc = roc_auc_score(y_test, probabilities)

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "ROC-AUC": roc_auc
    })

    print("\nBest parameters:", search.best_params_)
    print("Accuracy:", round(accuracy * 100, 2), "%")
    print("Precision:", round(precision, 4))
    print("Recall:", round(recall, 4))
    print("F1 Score:", round(f1, 4))
    print("ROC-AUC:", round(roc_auc, 4))

    # Select the model with the best F1 score
    if f1 > best_f1:
        best_f1 = f1
        best_model = tuned_model
        best_model_name = name


# 6. Display comparison
results_df = pd.DataFrame(results)

print("\n========== TUNED MODEL COMPARISON ==========")
print(results_df.to_string(index=False))


# 7. Save best tuned model
joblib.dump(best_model, "models/best_model.pkl")

print("\n========== BEST TUNED MODEL ==========")
print("Best Model:", best_model_name)
print("Best F1 Score:", round(best_f1, 4))
print("Saved to: models/best_model.pkl")


# 8. Detailed evaluation
final_predictions = best_model.predict(X_test_processed)

print("\n========== CLASSIFICATION REPORT ==========")

print(classification_report(
    y_test,
    final_predictions,
    target_names=["Stayed", "Exited"],
    zero_division=0
))
