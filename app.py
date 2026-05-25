import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load("customer_churn_model.pkl")

# Load feature columns
model_features = joblib.load("model_features.pkl")

# App title
st.title("📊 Customer Churn Predictor")

st.write(
    "Predict whether a customer is likely to churn."
)

# -----------------------------
# USER INPUTS
# -----------------------------

tenure = st.slider(
    "Tenure (months)",
    0,
    72,
    12
)

monthly_charges = st.slider(
    "Monthly Charges",
    0,
    150,
    70
)

total_charges = st.number_input(
    "Total Charges",
    value=1000.0
)

contract = st.selectbox(
    "Contract Type",
    [
        "Month-to-month",
        "One year",
        "Two year"
    ]
)

internet_service = st.selectbox(
    "Internet Service",
    [
        "DSL",
        "Fiber optic",
        "No"
    ]
)

paperless_billing = st.selectbox(
    "Paperless Billing",
    [
        "Yes",
        "No"
    ]
)

# -----------------------------
# CREATE INPUT DATAFRAME
# -----------------------------

input_data = pd.DataFrame({
    "tenure": [tenure],
    "MonthlyCharges": [monthly_charges],
    "TotalCharges": [total_charges]
})

# -----------------------------
# MANUAL ENCODING
# -----------------------------

# Initialize all model columns with 0
for col in model_features:
    if col not in input_data.columns:
        input_data[col] = 0

# Contract encoding
if contract == "One year":
    input_data["Contract_One year"] = 1

elif contract == "Two year":
    input_data["Contract_Two year"] = 1

# Internet service encoding
if internet_service == "Fiber optic":
    input_data["InternetService_Fiber optic"] = 1

elif internet_service == "No":
    input_data["InternetService_No"] = 1

# Paperless billing encoding
if paperless_billing == "Yes":
    input_data["PaperlessBilling_Yes"] = 1

# Reorder columns
input_data = input_data[model_features]

# -----------------------------
# PREDICTION
# -----------------------------

if st.button("Predict Churn"):

    prediction = model.predict(input_data)[0]

    probability = model.predict_proba(input_data)[0][1]

    st.subheader("Prediction Result")

    if prediction == 1:

        st.error(
            f"⚠️ Customer likely to churn "
            f"({probability:.2%} probability)"
        )

    else:

        st.success(
            f"✅ Customer likely to stay "
            f"({1 - probability:.2%} confidence)"
        )

    st.write("Churn Probability:", f"{probability:.2%}")

