import sys
import pandas as pd
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

sys.path.append("src")

from data_preprocessing import preprocess_data


st.title("Customer Churn Prediction App")

st.write(
    "Enter customer information to estimate churn probability."
)

# Load and prepare data
df = pd.read_csv(
    "data/raw/archive/WA_Fn-UseC_-Telco-Customer-Churn.csv"
)

X, y = preprocess_data(df)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)


# User inputs
tenure = st.slider(
    "Customer tenure (months)",
    0,
    72,
    12
)

monthly_charges = st.number_input(
    "Monthly Charges",
    min_value=0.0,
    value=50.0
)

total_charges = st.number_input(
    "Total Charges",
    min_value=0.0,
    value=500.0
)

internet_service = st.selectbox(
    "Internet Service",
    ["DSL", "Fiber optic", "No"]
)

contract = st.selectbox(
    "Contract Type",
    ["Month-to-month", "One year", "Two year"]
)


if st.button("Predict Churn"):

    input_data = pd.DataFrame(
        columns=X.columns
    )

    input_data.loc[0] = 0

    input_data.loc[0, "tenure"] = tenure
    input_data.loc[0, "MonthlyCharges"] = monthly_charges
    input_data.loc[0, "TotalCharges"] = total_charges

    if internet_service == "Fiber optic" and "InternetService_Fiber optic" in input_data.columns:
        input_data.loc[0, "InternetService_Fiber optic"] = 1

    if internet_service == "No" and "InternetService_No" in input_data.columns:
        input_data.loc[0, "InternetService_No"] = 1

    if contract == "One year" and "Contract_One year" in input_data.columns:
        input_data.loc[0, "Contract_One year"] = 1

    if contract == "Two year" and "Contract_Two year" in input_data.columns:
        input_data.loc[0, "Contract_Two year"] = 1

    input_scaled = scaler.transform(input_data)

    churn_probability = model.predict_proba(input_scaled)[0][1]

    st.subheader("Prediction Result")

    st.write(
        f"Churn probability: {churn_probability:.2%}"
    )

    if churn_probability >= 0.5:
        st.error("High risk: Customer is likely to churn.")
    else:
        st.success("Low risk: Customer is likely to stay.")