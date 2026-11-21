
import os
import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split

from feature_engineering import FeatureEngineer, IQRCapper


data = pd.read_csv("data/raw/Churn_Modelling.csv")

data = data.drop_duplicates()

data = data.drop(
    columns=["RowNumber", "CustomerId", "Surname"],
    errors="ignore"
)

X = data.drop("Exited", axis=1)
y = data["Exited"]


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


continuous_features = [
    "CreditScore",
    "Age",
    "Balance",
    "EstimatedSalary",
    "balance_to_salary_ratio"
]

other_numeric_features = [
    "Tenure",
    "NumOfProducts",
    "HasCrCard",
    "IsActiveMember",
    "products_per_tenure"
]

categorical_features = [
    "Geography",
    "Gender",
    "tenure_group"
]


continuous_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("outlier_capping", IQRCapper()),
    ("scaler", StandardScaler())
])

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])


preprocessor = Pipeline([
    ("feature_engineering", FeatureEngineer()),
    ("column_transformer", ColumnTransformer([
        ("continuous", continuous_pipeline, continuous_features),
        ("numeric", numeric_pipeline, other_numeric_features),
        ("categorical", categorical_pipeline, categorical_features)
    ]))
])


X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)


os.makedirs("models", exist_ok=True)

joblib.dump(preprocessor, "models/preprocessor.pkl")


print("PREPROCESSING COMPLETED SUCCESSFULLY!")
print("-------------------------------------")
print("Original dataset shape:", data.shape)
print("Training data shape:", X_train_processed.shape)
print("Testing data shape:", X_test_processed.shape)
print("Preprocessor saved successfully!")
