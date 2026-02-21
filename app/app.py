import streamlit as st
import pandas as pd
import joblib

st.title("Credit Risk Dashboard")

model = joblib.load("../models/pd_model.pkl")
scaler = joblib.load("../models/scaler.pkl")

age = st.slider("Age", 18, 75, 30)
credit = st.number_input("Credit Amount", 500, 20000, 5000)
duration = st.slider("Duration", 6, 72, 24)

if st.button("Predict Risk"):
    df = pd.DataFrame({
        "Age":[age],
        "Credit amount":[credit],
        "Duration":[duration]
    })

    df_scaled = scaler.transform(df)
    prob = model.predict_proba(df_scaled)[0][1]

    st.write(f"Default Probability: {prob:.2f}")
