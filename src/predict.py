import joblib
import pandas as pd

model = joblib.load("models/pd_model.pkl")
scaler = joblib.load("models/scaler.pkl")

def predict(data):
    data_scaled = scaler.transform(data)
    pd_prob = model.predict_proba(data_scaled)[:,1]
    return pd_prob
