import os
import pickle
import pandas as pd
import streamlit as st
from sklearn.linear_model import LinearRegression
from xgboost import XGBClassifier

st.set_page_config(page_title="Customer Prediction App", page_icon="📊", layout="centered")
st.title("📊 Customer Prediction App")
st.caption("Predict Repeat Purchase and Future Revenue")

REPEAT_FEATURES = [
    "total_orders",
    "total_price",
    "average_price",
    "average_freight",
    "average_review",
    "average_installments",
    "total_payment",
    "average_photos",
    "average_weight",
    "approval_time",
    "delivery_time",
    "delivery_delay",
    "shipping_delay",
    "review_response",
    "purchase_year",
    "purchase_month",
    "purchase_hour",
]

REVENUE_FEATURES = [
    "total_orders",
    "last_purchase_year",
    "last_purchase_month",
    "first_purchase_year",
    "customer_tenure_days",
    "average_installments",
    "average_review",
    "past_revenue",
    "avg_order_value",
]


@st.cache_resource
def load_repeat_model():
    """Load saved XGBoost model or train a new one."""
    if os.path.exists("xgb_model.pkl"):
        with open("xgb_model.pkl", "rb") as file:
            return pickle.load(file)

    df = pd.read_csv("customer_df.csv")
    df = df.fillna(df.median(numeric_only=True))
    X = df[REPEAT_FEATURES]
    y = df["Repeat_Purchase"]

    model = XGBClassifier(eval_metric="logloss", random_state=42)
    model.fit(X, y)
    return model


@st.cache_resource
def load_revenue_model():
    """Load saved Linear Regression model or train a new one."""
    if os.path.exists("lr_model.pkl"):
        with open("lr_model.pkl", "rb") as file:
            return pickle.load(file)

    df = pd.read_csv("revenue_dataset.csv")
    X = df[REVENUE_FEATURES]
    y = df["future_revenue"]

    model = LinearRegression()
    model.fit(X, y)
    return model


try:
    repeat_model = load_repeat_model()
    revenue_model = load_revenue_model()
except FileNotFoundError as error:
    st.error(f"Required CSV file not found: {error}")
    st.stop()
except Exception as error:
    st.error(f"Could not load models: {error}")
    st.stop()

option = st.sidebar.radio(
    "Choose prediction type:",
    ["Repeat Purchase", "Future Revenue"],
)

if option == "Repeat Purchase":
    st.header("🔄 Repeat Purchase Prediction")
    st.write("Will the customer buy again within 90 days?")

    col1, col2 = st.columns(2)

    with col1:
        total_orders = st.number_input("Total Orders", min_value=1, value=1)
        total_price = st.number_input("Total Price", min_value=0.0, value=100.0)
        average_price = st.number_input("Average Price", min_value=0.0, value=100.0)
        average_freight = st.number_input("Average Freight", min_value=0.0, value=10.0)
        average_review = st.number_input("Average Review (1-5)", min_value=1.0, max_value=5.0, value=4.0)
        average_installments = st.number_input("Average Installments", min_value=1.0, value=3.0)
        total_payment = st.number_input("Total Payment", min_value=0.0, value=110.0)
        average_photos = st.number_input("Average Photos", min_value=0.0, value=2.0)
        average_weight = st.number_input("Average Weight (g)", min_value=0.0, value=500.0)

    with col2:
        approval_time = st.number_input("Approval Time (hours)", min_value=0.0, value=1.0)
        delivery_time = st.number_input("Delivery Time (days)", min_value=0.0, value=7.0)
        delivery_delay = st.number_input("Delivery Delay (days)", value=-2.0)
        shipping_delay = st.number_input("Shipping Delay (days)", value=-1.0)
        review_response = st.number_input("Review Response (days)", min_value=0.0, value=2.0)
        purchase_year = st.number_input("Purchase Year", min_value=2016, max_value=2020, value=2018)
        purchase_month = st.number_input("Purchase Month", min_value=1, max_value=12, value=6)
        purchase_hour = st.number_input("Purchase Hour", min_value=0, max_value=23, value=12)

    if st.button("Predict Repeat Purchase", type="primary"):
        input_data = pd.DataFrame(
            [[
                total_orders, total_price, average_price, average_freight,
                average_review, average_installments, total_payment,
                average_photos, average_weight, approval_time, delivery_time,
                delivery_delay, shipping_delay, review_response,
                purchase_year, purchase_month, purchase_hour,
            ]],
            columns=REPEAT_FEATURES,
        )

        prediction = repeat_model.predict(input_data)[0]
        probability = repeat_model.predict_proba(input_data)[0][1]

        if prediction == 1:
            st.success(f"✅ Likely to repeat purchase (confidence: {probability:.0%})")
        else:
            st.warning(f"❌ Unlikely to repeat purchase (confidence: {1 - probability:.0%})")

else:
    st.header("💰 Future Revenue Prediction")
    st.write("Predict how much revenue the customer will generate next.")

    col1, col2 = st.columns(2)

    with col1:
        total_orders = st.number_input("Total Orders", min_value=1, value=2)
        last_purchase_year = st.number_input("Last Purchase Year", min_value=2016, max_value=2020, value=2018)
        last_purchase_month = st.number_input("Last Purchase Month", min_value=1, max_value=12, value=6)
        first_purchase_year = st.number_input("First Purchase Year", min_value=2016, max_value=2020, value=2017)
        customer_tenure_days = st.number_input("Customer Tenure (days)", min_value=0, value=180)

    with col2:
        average_installments = st.number_input("Average Installments", min_value=1.0, value=3.0)
        average_review = st.number_input("Average Review (1-5)", min_value=1.0, max_value=5.0, value=4.0)
        past_revenue = st.number_input("Past Revenue", min_value=0.0, value=200.0)
        avg_order_value = st.number_input("Average Order Value", min_value=0.0, value=100.0)

    if st.button("Predict Future Revenue", type="primary"):
        input_data = pd.DataFrame(
            [[
                total_orders,
                last_purchase_year,
                last_purchase_month,
                first_purchase_year,
                customer_tenure_days,
                average_installments,
                average_review,
                past_revenue,
                avg_order_value,
            ]],
            columns=REVENUE_FEATURES,
        )

        prediction = revenue_model.predict(input_data)[0]
        prediction = max(prediction, 0)

        st.success(f"💰 Predicted Future Revenue: ₹ {prediction:,.2f}")
