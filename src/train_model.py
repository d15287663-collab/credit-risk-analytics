import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import joblib

from data_preprocessing import load_data, preprocess
from feature_engineering import create_target, encode_features

# Load data
df = load_data("data/german_credit_data.csv")
df = preprocess(df)
df = create_target(df)

X = df.drop("Risk", axis=1)
X = encode_features(X)
y = df["Risk"]

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Train model
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# Save
joblib.dump(model, "models/pd_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")

print("Model trained and saved")
