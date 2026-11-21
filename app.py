
import streamlit as st
import pandas as pd
import plotly.express as px
import urllib.request
import urllib.error
import json
from pathlib import Path
import plotly.graph_objects as go

import sys
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score


# ---------------- PAGE CONFIG ----------------

BASE_DIR = Path(__file__).parent

st.set_page_config(
    page_title="ChurnAI | Analytics",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------- CUSTOM GREEN UI ----------------

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background:
        radial-gradient(ellipse at 50% 0%, #103d2a 0%, transparent 48%),
        linear-gradient(135deg, #030907 0%, #07130e 55%, #020806 100%);
    color: #e6f4ec;
}

/* Main content */
.block-container {
    max-width: 1450px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #071810, #030b07);
    border-right: 1px solid #173c29;
}

section[data-testid="stSidebar"] * {
    color: #d7eee0;
}

/* Headings */
h1, h2, h3 {
    color: #edfff4 !important;
    font-weight: 700 !important;
}

/* Metric cards */
div[data-testid="stMetric"] {
    background: linear-gradient(145deg, #102b1e, #091a12);
    border: 1px solid #24583b;
    border-radius: 16px;
    padding: 22px;
    box-shadow: 0 0 25px rgba(25,230,140,0.07);
}

div[data-testid="stMetric"] label {
    color: #8eaf9b !important;
}

div[data-testid="stMetric"] [data-testid="stMetricValue"] {
    color: #29e994 !important;
    font-weight: 800;
}

/* Input fields */
.stTextInput input,
.stNumberInput input,
.stSelectbox div[data-baseweb="select"] {
    background-color: #0b1e14 !important;
    color: #e6f4ec !important;
    border-color: #28543b !important;
    border-radius: 10px !important;
}

/* Buttons */
.stButton button,
.stFormSubmitButton button {
    background: linear-gradient(90deg, #10b96c, #20e58c) !important;
    color: #031109 !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 800 !important;
    padding: 0.65rem 1.2rem !important;
    transition: 0.2s ease;
}

.stButton button:hover,
.stFormSubmitButton button:hover {
    box-shadow: 0 0 22px rgba(25,230,140,0.45);
    transform: translateY(-2px);
}

/* Form container */
div[data-testid="stForm"] {
    background: linear-gradient(145deg, #10251a, #09150e);
    border: 1px solid #24583b;
    border-radius: 18px;
    padding: 24px;
}

/* Divider */
hr {
    border-color: #1b3c29 !important;
}

/* Dataframe */
div[data-testid="stDataFrame"] {
    border: 1px solid #24583b;
    border-radius: 12px;
}

/* Hide Streamlit branding */
#MainMenu, footer {
    visibility: hidden;
}

</style>
""", unsafe_allow_html=True)


# ---------------- DATA ----------------

@st.cache_data
def load_dataset():
    path = BASE_DIR / "data" / "raw" / "Churn_Modelling.csv"
    return pd.read_csv(path)


def check_api():
    try:
        request = urllib.request.Request(
            "http://127.0.0.1:8000/health"
        )
        response = urllib.request.urlopen(request, timeout=3)
        return response.status == 200
    except Exception:
        return False


try:
    data = load_dataset()
    total_customers = len(data)
except Exception:
    data = None
    total_customers = 0


api_connected = check_api()


# ---------------- SIDEBAR ----------------

with st.sidebar:
    st.markdown(
        "<h2 style='color:#27e994;'>◈ ChurnAI</h2>",
        unsafe_allow_html=True
    )

    st.caption("CUSTOMER INTELLIGENCE PLATFORM")

    st.divider()

    st.markdown("### Navigation")
    page = st.radio(
        "Go to",
        ["Overview", "Churn Prediction"],
        label_visibility="collapsed"
    )

    st.divider()

    if api_connected:
        st.success("● API Connected")
    else:
        st.error("● API Disconnected")

    st.caption("Powered by CatBoost + FastAPI")




# ---------------- HERO HEADER ----------------

st.markdown("""
<div style="
    background: linear-gradient(120deg, #10412a, #05160e);
    border: 1px solid #24583b;
    border-radius: 22px;
    padding: 30px;
    margin-bottom: 25px;
">
    <div style="color:#29e994; font-size:12px;
                font-weight:700; letter-spacing:3px;">
        AI POWERED ANALYTICS
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown(
    "<h1 style='color:#edfff4; font-size:36px;'>Customer Churn Intelligence</h1>",
    unsafe_allow_html=True
)

st.markdown(
    "<p style='color:#9ab8a5; font-size:15px;'>Understand customer behaviour. Identify churn risks. Make smarter retention decisions.</p>",
    unsafe_allow_html=True
)


# ---------------- OVERVIEW PAGE ----------------

if page == "Overview":

    st.markdown("## Performance Overview")
    st.caption("Machine learning model performance and customer statistics")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric("Model Accuracy", "87.20%", "CatBoost")

    with c2:
        st.metric("ROC-AUC Score", "87.18%", "CatBoost")

    with c3:
        st.metric("Total Customers", f"{total_customers:,}")

    st.write("")

    # Model comparison
    st.markdown("## Model Comparison")

    comparison_path = BASE_DIR / "results" / "model_comparison.csv"

    if comparison_path.exists():

        comparison = pd.read_csv(comparison_path)

        model_col = next(
            (c for c in comparison.columns
             if c.lower() in ["model", "model_name", "algorithm"]),
            None
        )

        accuracy_col = next(
            (c for c in comparison.columns
             if "accuracy" in c.lower()),
            None
        )

        if model_col and accuracy_col:

            chart_data = comparison[[model_col, accuracy_col]].copy()
            chart_data.columns = ["Model", "Accuracy"]

            if chart_data["Accuracy"].max() <= 1:
                chart_data["Accuracy"] *= 100

            fig = px.bar(
                chart_data,
                x="Model",
                y="Accuracy",
                text="Accuracy"
            )

            fig.update_traces(
                marker_color="#19E68C",
                marker_line_color="#8AFFC2",
                marker_line_width=1,
                texttemplate="%{text:.2f}%",
                textposition="outside"
            )

            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="#d9f5e4",
                yaxis_title="Accuracy (%)",
                xaxis_title="",
                yaxis_range=[0, 100],
                margin=dict(t=35, b=20, l=10, r=10)
            )

            st.plotly_chart(fig, use_container_width=True)

        else:
            st.warning("Could not identify model and accuracy columns in the CSV.")

    else:
        st.info("Model comparison CSV is not available.")

# ---------------- ROC CURVE ----------------

st.markdown("## CatBoost ROC Curve")

try:
    # Import custom preprocessing classes
    sys.path.insert(0, str(BASE_DIR / "src"))
    from feature_engineering import FeatureEngineer, IQRCapper

    @st.cache_resource
    def load_churn_model():
        model = joblib.load(
            BASE_DIR / "models" / "catboost_model.pkl"
        )

        preprocessor = joblib.load(
            BASE_DIR / "models" / "catboost_preprocessor.pkl"
        )

        return model, preprocessor

    model, preprocessor = load_churn_model()

    # Prepare the original dataset
    X = data.drop(
        columns=["Exited", "RowNumber", "CustomerId", "Surname"]
    )

    y = data["Exited"]

    # Recreate the project's original test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    # Apply saved preprocessing
    X_test_processed = preprocessor.transform(X_test)

    # Calculate churn probabilities
    probabilities = model.predict_proba(X_test_processed)[:, 1]

    # ROC values
    fpr, tpr, thresholds = roc_curve(
        y_test,
        probabilities
    )

    auc_score = roc_auc_score(
        y_test,
        probabilities
    )

    # Draw ROC curve
    roc_fig = go.Figure()

    roc_fig.add_trace(go.Scatter(
        x=fpr,
        y=tpr,
        mode="lines",
        name=f"CatBoost (AUC = {auc_score:.3f})",
        line=dict(color="#19E68C", width=3),
        fill="tozeroy",
        fillcolor="rgba(25,230,140,0.10)"
    ))

    roc_fig.add_trace(go.Scatter(
        x=[0, 1],
        y=[0, 1],
        mode="lines",
        name="Random Classifier",
        line=dict(color="#A0A0A0", dash="dash")
    ))

    roc_fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#E6F4EC",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        title=f"ROC Curve | AUC: {auc_score:.4f}"
    )

    st.plotly_chart(
        roc_fig,
        use_container_width=True
    )

    st.caption(
        "Exploratory evaluation using the project's existing 20% test split."
    )

    # ---------------- FEATURE IMPORTANCE ----------------

    st.markdown("## Feature Importance")

    importances = model.feature_importances_

    try:
        feature_names = preprocessor.get_feature_names_out()

        if len(feature_names) != len(importances):
            raise ValueError("Feature name count mismatch")

    except Exception:
        feature_names = [
            f"Feature {i+1}" for i in range(len(importances))
        ]

    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    })

    importance_df = importance_df.sort_values(
        by="Importance",
        ascending=True
    ).tail(10)

    importance_fig = px.bar(
        importance_df,
        x="Importance",
        y="Feature",
        orientation="h",
        text="Importance"
    )

    importance_fig.update_traces(
        marker_color="#19E68C",
        texttemplate="%{text:.2f}",
        textposition="outside"
    )

    importance_fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#E6F4EC",
        title="Top 10 Important Features",
        xaxis_title="Importance Score",
        yaxis_title="",
        margin=dict(l=20, r=40, t=50, b=20)
    )

    st.plotly_chart(
        importance_fig,
        use_container_width=True
    )

except Exception as error:
    st.error(f"Could not load model analytics: {error}")


    # Dataset preview
    st.markdown("## Customer Dataset")

    if data is not None:
        st.caption("A preview of the customer records used in this project.")
        st.dataframe(data.head(8), use_container_width=True)

    # Churn distribution
    if data is not None and "Exited" in data.columns:

        st.markdown("## Customer Churn Distribution")

        churn_counts = data["Exited"].value_counts().reset_index()
        churn_counts.columns = ["Customer Status", "Count"]
        churn_counts["Customer Status"] = churn_counts[
            "Customer Status"
        ].map({0: "Stayed", 1: "Churned"})

        fig = px.pie(
            churn_counts,
            names="Customer Status",
            values="Count",
            hole=0.65,
            color="Customer Status",
            color_discrete_map={
                "Stayed": "#19E68C",
                "Churned": "#E0A34A"
            }
        )

        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="#e6f4ec"
        )

        st.plotly_chart(fig, use_container_width=True)


# ---------------- PREDICTION PAGE ----------------

else:

    st.markdown("## Predict Customer Churn")

    st.caption(
        "Enter the customer information below to estimate their churn risk."
    )

    with st.form("customer_form"):

        st.markdown("### Customer Information")

        col1, col2 = st.columns(2)

        with col1:

            credit_score = st.number_input(
                "Credit Score", 300, 900, 650
            )

            geography = st.selectbox(
                "Geography", ["France", "Germany", "Spain"]
            )

            gender = st.selectbox(
                "Gender", ["Male", "Female"]
            )

            age = st.number_input(
                "Age", 18, 100, 35
            )

            tenure = st.number_input(
                "Tenure (years)", 0, 10, 5
            )

        with col2:

            balance = st.number_input(
                "Account Balance",
                min_value=0.0,
                value=50000.0
            )

            products = st.number_input(
                "Number of Products", 1, 4, 1
            )

            credit_card = st.selectbox(
                "Has Credit Card?",
                [1, 0],
                format_func=lambda x: "Yes" if x == 1 else "No"
            )

            active_member = st.selectbox(
                "Active Member?",
                [1, 0],
                format_func=lambda x: "Yes" if x == 1 else "No"
            )

            salary = st.number_input(
                "Estimated Salary",
                min_value=0.0,
                value=50000.0
            )

        submitted = st.form_submit_button(
            "◈  Analyse Customer Risk",
            use_container_width=True
        )

    if submitted:

        customer = {
            "CreditScore": credit_score,
            "Geography": geography,
            "Gender": gender,
            "Age": age,
            "Tenure": tenure,
            "Balance": balance,
            "NumOfProducts": products,
            "HasCrCard": credit_card,
            "IsActiveMember": active_member,
            "EstimatedSalary": salary
        }

        try:

            request = urllib.request.Request(
                "http://127.0.0.1:8000/predict",
                data=json.dumps(customer).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST"
            )

            response = urllib.request.urlopen(request, timeout=10)
            result = json.loads(response.read().decode("utf-8"))

            probability = result["churn_probability"] * 100

            st.markdown("## Prediction Analysis")

            r1, r2 = st.columns(2)

            with r1:
                st.metric(
                    "Churn Probability",
                    f"{probability:.2f}%"
                )

            with r2:
                st.metric(
                    "Prediction Threshold",
                    f"{result['threshold'] * 100:.2f}%"
                )

            st.progress(float(result["churn_probability"]))

            if result["prediction"] == 1:

                st.error(
                    "⚠️ HIGH CHURN RISK — This customer is predicted to leave."
                )

            else:

                st.success(
                    "✓ LOW CHURN RISK — This customer is predicted to stay."
                )

        except urllib.error.URLError:
            st.error("Cannot connect to FastAPI. Please start the backend.")

        except Exception as error:
            st.error(f"Prediction failed: {error}")


# ---------------- FOOTER ----------------

st.divider()

st.markdown("""
<div style="text-align:center; color:#668b75; padding:15px;">
    CHURNAI &nbsp; • &nbsp; CUSTOMER INTELLIGENCE PLATFORM
    <br><br>
    <span style="font-size:12px;">
        Built with Python, CatBoost, FastAPI and Streamlit
    </span>
</div>
""", unsafe_allow_html=True)
