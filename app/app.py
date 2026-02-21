import streamlit as st
import pandas as pd
import joblib

st.title("Credit Risk PD Model")

# load model
model = joblib.load("pd_model.pkl")
scaler = joblib.load("scaler.pkl")

st.write("Enter customer details")

age = st.slider("Age", 18, 75, 30)
credit = st.number_input("Credit Amount", 500, 20000, 5000)
duration = st.slider("Duration", 6, 72, 24)

if st.button("Predict Default Risk"):

    df = pd.DataFrame({
        "Age":[age],
        "Credit amount":[credit],
        "Duration":[duration]
    })

    df_scaled = scaler.transform(df)
    prob = model.predict_proba(df_scaled)[0][1]

    st.success(f"Default Probability: {prob:.2f}")
