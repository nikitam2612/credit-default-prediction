
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "models"
    / "random_forest_model.joblib"
)

st.set_page_config(
    page_title="Credit Default Risk Predictor",
    page_icon="📊",
    layout="wide",
)

def load_sample_customer():
    sample_values = {
        "input_LIMIT_BAL": 50000,
        "input_SEX": 1,
        "input_AGE": 30,
        "input_EDUCATION": 2,
        "input_MARRIAGE": 2,
        "input_PAY_0": 0,
        "input_PAY_2": 0,
        "input_PAY_3": 0,
        "input_PAY_4": 0,
        "input_PAY_5": 0,
        "input_PAY_6": 0,
        "input_BILL_AMT1": 15000,
        "input_BILL_AMT2": 12000,
        "input_BILL_AMT3": 10000,
        "input_BILL_AMT4": 8000,
        "input_BILL_AMT5": 6000,
        "input_BILL_AMT6": 5000,
        "input_PAY_AMT1": 3000,
        "input_PAY_AMT2": 3000,
        "input_PAY_AMT3": 2000,
        "input_PAY_AMT4": 2000,
        "input_PAY_AMT5": 1500,
        "input_PAY_AMT6": 1000,
    }

    for key, value in sample_values.items():
        st.session_state[key] = value

        
if st.button("Load Sample Customer"):
    load_sample_customer()


st.title("Credit Default Risk Predictor")
st.caption("Interactive machine learning portfolio project")

st.warning(
    "Educational demonstration only. Predictions are not validated "
    "for real-world lending or credit approval decisions."
)

if not MODEL_PATH.exists():
    st.error(
        "Model file not found. From the project root, run "
        "`python src/tune_threshold.py` first."
    )
    st.stop()

@st.cache_resource
def load_model_artifact(path):
    return joblib.load(path)

artifact = load_model_artifact(str(MODEL_PATH))
model = artifact["model"]
threshold = float(artifact["threshold"])
feature_names = artifact["feature_names"]

st.write(
    f"Model: **Random Forest** | "
    f"Classification threshold: **{threshold:.2f}**"
)

st.subheader("Model Performance")

metric_cols = st.columns(4)

with metric_cols[0]:
    st.metric("Test Accuracy", "78.53%")

with metric_cols[1]:
    st.metric("Test Precision", "51.42%")

with metric_cols[2]:
    st.metric("Test Recall", "53.28%")

with metric_cols[3]:
    st.metric("Test F1 Score", "52.33%")

st.caption(
    "Metrics are from the held-out test set evaluated at the "
    "selected classification threshold of 0.45. Results are "
    "specific to this dataset and split."
)


st.markdown(
    "Enter the customer's information below. "
    "The input fields follow the feature structure used to train the model."
)

with st.form("prediction_form"):
    st.subheader("1. Customer Profile")

    col1, col2, col3 = st.columns(3)

    
with col1:
    limit_bal = st.number_input(
        "Credit limit (NT dollars)",
        min_value=0,
        max_value=10_000_000,
        step=10_000,
        key="input_LIMIT_BAL",
    )


    sex = st.selectbox(
    "SEX — dataset code",
    options=[1, 2],
    format_func=lambda x: {
        1: "1 — Male (dataset code)",
        2: "2 — Female (dataset code)",
    }[x],
    key="input_SEX",
)

    with col3:
        age = st.number_input(
    "Age",
    min_value=18,
    max_value=100,
    step=1,
    key="input_AGE",
)

    col1, col2 = st.columns(2)

    with col1:
        education = st.selectbox(
    "EDUCATION — dataset code",
    options=[1, 2, 3, 4, 0, 5, 6],
    format_func=lambda x: {
        0: "0 — Other / unknown code",
        1: "1 — Graduate school",
        2: "2 — University",
        3: "3 — High school",
        4: "4 — Other",
        5: "5 — Unknown code",
        6: "6 — Unknown code",
    }[x],
    key="input_EDUCATION",
)

    with col2:
        marriage = st.selectbox(
    "MARRIAGE — dataset code",
    options=[1, 2, 3, 0],
    format_func=lambda x: {
        0: "0 — Unknown code",
        1: "1 — Married",
        2: "2 — Single",
        3: "3 — Other",
    }[x],
    key="input_MARRIAGE",
)

    st.divider()
    st.subheader("2. Repayment Status")

    st.caption(
        "Use the dataset's numeric codes. Negative values and zero "
        "are meaningful categories; positive values represent "
        "reported repayment delay levels."
    )
    



    repayment_options = list(range(-2, 9))
    repayment_labels = {
        -2: "-2 — Dataset code",
        -1: "-1 — Paid duly / dataset code",
        0: "0 — Revolving credit / dataset code",
        1: "1 — Payment delay of 1 month",
        2: "2 — Payment delay of 2 months",
        3: "3 — Payment delay of 3 months",
        4: "4 — Payment delay of 4 months",
        5: "5 — Payment delay of 5 months",
        6: "6 — Payment delay of 6 months",
        7: "7 — Payment delay of 7 months",
        8: "8 — Payment delay of 8 months",
    }

    pay_cols = st.columns(3)
    repayment_values = {}

    for i, feature in enumerate(
        ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
    ):
        with pay_cols[i % 3]:
            repayment_values[feature] = st.selectbox(
                feature,
                options=repayment_options,
                index=repayment_options.index(0),
                format_func=lambda x: repayment_labels[x],
                key=f"input_{feature}",
            )

    st.divider()
    st.subheader("3. Recent Bill Amounts")

    st.caption(
        "Enter amounts in NT dollars, following the dataset's "
        "monthly feature order."
    )

    bill_cols = st.columns(3)
    bill_values = {}

    for i, feature in enumerate(
        [
            "BILL_AMT1",
            "BILL_AMT2",
            "BILL_AMT3",
            "BILL_AMT4",
            "BILL_AMT5",
            "BILL_AMT6",
        ]
    ):
        with bill_cols[i % 3]:
            bill_values[feature] = st.number_input(
                feature,
                min_value=-10_000_000,
                max_value=10_000_000,
                value=0,
                step=1_000,
                key=f"input_{feature}",
            )

    st.divider()
    st.subheader("4. Previous Payment Amounts")

    payment_cols = st.columns(3)
    payment_values = {}

    for i, feature in enumerate(
        [
            "PAY_AMT1",
            "PAY_AMT2",
            "PAY_AMT3",
            "PAY_AMT4",
            "PAY_AMT5",
            "PAY_AMT6",
        ]
    ):
        with payment_cols[i % 3]:
            payment_values[feature] = st.number_input(
                feature,
                min_value=0,
                max_value=10_000_000,
                value=0,
                step=1_000,
                key=f"input_{feature}",
            )

    submitted = st.form_submit_button(
        "Predict Default Risk",
        type="primary",
        use_container_width=True,
    )

if submitted:
    input_values = {
        "LIMIT_BAL": limit_bal,
        "SEX": sex,
        "EDUCATION": education,
        "MARRIAGE": marriage,
        "AGE": age,
        **repayment_values,
        **bill_values,
        **payment_values,
    }

    input_df = pd.DataFrame(
        [[input_values[name] for name in feature_names]],
        columns=feature_names,
    )

    probability = float(model.predict_proba(input_df)[0, 1])
    predicted_class = int(probability >= threshold)

    st.divider()
    st.subheader("Prediction Result")

    result_col1, result_col2 = st.columns(2)

    with result_col1:
        st.metric(
            "Model-estimated default probability",
            f"{probability:.1%}",
        )
        st.progress(min(max(probability, 0.0), 1.0))

    with result_col2:
        if predicted_class == 1:
            st.error(
                f"Predicted class: DEFAULT (1) at threshold {threshold:.2f}"
            )
        else:
            st.success(
                f"Predicted class: NO DEFAULT (0) at threshold {threshold:.2f}"
            )

    st.caption(
        "The classification uses the validation-selected threshold "
        f"of {threshold:.2f}. The probability is a model output and "
        "has not necessarily been calibrated to real-world risk."
    )

    with st.expander("View submitted model inputs"):
        st.dataframe(input_df, use_container_width=True)

st.divider()
st.caption(
    "Portfolio project using the UCI Default of Credit Card Clients dataset. "
    "This tool is for educational demonstration only."
)
