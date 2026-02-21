import pandas as pd

def load_data(path):
    df = pd.read_csv(path)
    df.drop(columns=["Unnamed: 0"], inplace=True)
    return df

def preprocess(df):
    df = df.copy()

    df["Saving accounts"].fillna("no_info", inplace=True)
    df["Checking account"].fillna("no_info", inplace=True)

    return df
