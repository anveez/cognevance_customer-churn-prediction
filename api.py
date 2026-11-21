
import sys
import json
import joblib
import pandas as pd

from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Project paths
BASE_DIR = Path(__file__).parent

# Import custom preprocessing classes
sys.path.insert(0, str(BASE_DIR / "src"))

from feature_engineering import FeatureEngineer, IQRCapper

# Load trained model and preprocessor
model = joblib.load(BASE_DIR / "models" / "catboost_model.pkl")

preprocessor = joblib.load(
    BASE_DIR / "models" / "catboost_preprocessor.pkl"
)

# Load prediction threshold
with open(BASE_DIR / "models" / "catboost_threshold.json", "r") as file:
    threshold = json.load(file)["threshold"]

# Create FastAPI application
app = FastAPI(
    title="Customer Churn Prediction API",
    description="Predict whether a bank customer will leave",
    version="1.0"
)


# Customer input details
class CustomerData(BaseModel):
    CreditScore: int = Field(ge=300, le=900)
    Geography: str
    Gender: str
    Age: int = Field(ge=18, le=100)
    Tenure: int = Field(ge=0, le=10)
    Balance: float = Field(ge=0)
    NumOfProducts: int = Field(ge=1, le=4)
    HasCrCard: int = Field(ge=0, le=1)
    IsActiveMember: int = Field(ge=0, le=1)
    EstimatedSalary: float = Field(ge=0)


# Home route
@app.get("/")
def home():
    return {
        "message": "Customer Churn Prediction API is running"
    }


# Health check
@app.get("/health")
def health():
    return {"status": "healthy"}


# Prediction route
@app.post("/predict")
def predict(customer: CustomerData):

    try:
        customer_df = pd.DataFrame([customer.model_dump()])

        processed_data = preprocessor.transform(customer_df)

        probabilities = model.predict_proba(processed_data)

        churn_probability = float(probabilities[0][1])

        prediction = int(churn_probability >= threshold)

        return {
            "prediction": prediction,
            "result": "Churn" if prediction == 1 else "Stay",
            "churn_probability": round(churn_probability, 4),
            "threshold": threshold
        }

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=str(error)
        )
