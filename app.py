import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model
model = joblib.load("restaurant_revenue_model.pkl")

st.title("Restaurant Revenue Predictor")

st.markdown("Enter restaurant details to predict estimated revenue.")

seating = st.number_input("Seating Capacity", min_value=10, max_value=500, step=10)
meal_price = st.number_input("Average Meal Price ($)", min_value=5.0, max_value=100.0, step=1.0)
marketing_budget = st.number_input("Marketing Budget ($)", min_value=0.0, step=100.0)
followers = st.number_input("Social Media Followers", min_value=0)

input_df = pd.DataFrame([{
    'Seating Capacity': seating,
    'Average Meal Price': meal_price,
    'Marketing Budget': marketing_budget,
    'Social Media Followers': followers,
}])

if st.button("Predict Revenue"):
    pred_log = model.predict(input_df)[0]
    pred_rev = np.expm1(pred_log)
    st.success(f"Estimated Revenue: ${pred_rev:,.2f}")
