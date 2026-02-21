import pandas as pd

def create_target(df):
    # Create risk target
    df["Risk"] = (df["Credit amount"] > 5000).astype(int)
    return df

def encode_features(df):
    df = pd.get_dummies(df, drop_first=True)
    return df
