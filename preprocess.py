# preprocess.py

import pandas as pd

def load_diabetes_data(path):
    df = pd.read_csv(path).dropna()
    return df

def load_heart_data(path):
    df = pd.read_csv(path).dropna()
    return df

def load_kidney_data(path):
    df = pd.read_csv(path).dropna()
    if 'id' in df.columns:
        df = df.drop(columns=['id'])
    if 'classification' in df.columns:
        df['classification'] = df['classification'].replace({'ckd': 1, 'notckd': 0})
    numeric_cols = ["sg", "al", "su", "bgr", "bu", "sc", "hemo", "pcv", "wc", "rc", "classification"]
    return df[numeric_cols].dropna()

def load_stroke_data(path):
    df = pd.read_csv(path)
    df = df.dropna()

    # Encode categorical columns
    df['gender'] = df['gender'].replace({'Male': 1, 'Female': 0, 'Other': 2})
    df['ever_married'] = df['ever_married'].replace({'Yes': 1, 'No': 0})
    df['Residence_type'] = df['Residence_type'].replace({'Urban': 1, 'Rural': 0})
    df['work_type'] = df['work_type'].replace({'Private': 0, 'Self-employed': 1, 'Govt_job': 2, 'children': 3, 'Never_worked': 4})
    df['smoking_status'] = df['smoking_status'].replace({'never smoked': 0, 'formerly smoked': 1, 'smokes': 2, 'Unknown': 3})

    features = ['gender', 'age', 'hypertension', 'heart_disease', 'ever_married',
                'work_type', 'Residence_type', 'avg_glucose_level', 'bmi', 'smoking_status', 'stroke']

    return df[features].dropna()
