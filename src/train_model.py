from feature_engineering import FeatureEngineer, IQRCapper
import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)


# 1. Load dataset
data = pd.read_csv("data/raw/Churn_Modelling.csv")

data = data.drop_duplicates()

data = data.drop(
    columns=["RowNumber", "CustomerId", "Surname"],
    errors="ignore"
)

X = data.drop("Exited", axis=1)
y = data["Exited"]


# 2. Split dataset (same split used during preprocessing)
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# 3. Load saved preprocessing pipeline
preprocessor = joblib.load("models/preprocessor.pkl")

X_train_processed = preprocessor.transform(X_train)
X_test_processed = preprocessor.transform(X_test)


# 4. Define Machine Learning models
models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        random_state=42
    ),

    "Decision Tree": DecisionTreeClassifier(
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    ),

    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        random_state=42
    )
}


# 5. Train and evaluate models
results = []

os.makedirs("models", exist_ok=True)

best_model = None
best_accuracy = 0
best_model_name = ""

for name, model in models.items():

    print("\nTraining:", name)

    model.fit(X_train_processed, y_train)

    predictions = model.predict(X_test_processed)
    probabilities = model.predict_proba(X_test_processed)[:, 1]

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

    print("Accuracy:", round(accuracy * 100, 2), "%")
    print("Precision:", round(precision, 4))
    print("Recall:", round(recall, 4))
    print("F1 Score:", round(f1, 4))
    print("ROC-AUC:", round(roc_auc, 4))

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = model
        best_model_name = name


# 6. Display comparison
results_df = pd.DataFrame(results)

print("\n========== MODEL COMPARISON ==========")
print(results_df.to_string(index=False))


# 7. Save the best model
joblib.dump(best_model, "models/best_model.pkl")

print("\n========== BEST MODEL ==========")
print("Best Model:", best_model_name)
print("Best Accuracy:", round(best_accuracy * 100, 2), "%")
print("Saved to: models/best_model.pkl")


# 8. Detailed evaluation of best model
best_predictions = best_model.predict(X_test_processed)

print("\n========== CLASSIFICATION REPORT ==========")
print(classification_report(
    y_test,
    best_predictions,
    target_names=["Stayed", "Exited"],
    zero_division=0
))

print("\n========== CONFUSION MATRIX ==========")
print(confusion_matrix(y_test, best_predictions))
