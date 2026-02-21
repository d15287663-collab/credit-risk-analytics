# Credit Risk Analytics Project

This project predicts the Probability of Default (PD) for loan applicants using machine learning.

## Dataset
German Credit Dataset (1000 customers)

## Features
- Age
- Sex
- Job
- Housing
- Credit amount
- Duration
- Purpose
- Saving accounts
- Checking account

## Model
Logistic Regression (PD Model)

## How to Run

### Install requirements
pip install -r requirements.txt

### Train model
python src/train_model.py

### Run dashboard
streamlit run app/app.py
