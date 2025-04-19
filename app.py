import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import gdown
import urllib.request

MODEL_PATH = 'restaurant_revenue_model.pkl'
FILE_ID = '1igwi6zBQK6MnN-1ACExmTcN-EvhoN0PZ'
MODEL_URL = f'https://drive.google.com/uc?export=download&id={FILE_ID}'

if not os.path.exists(MODEL_PATH):
    gdown.download(MODEL_URL, MODEL_PATH, quiet=False, fuzzy=True)

model = joblib.load(MODEL_PATH)

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



