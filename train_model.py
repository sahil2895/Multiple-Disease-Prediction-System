# train_model.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
from preprocess import (
    load_diabetes_data,
    load_heart_data,
    load_kidney_data,
    load_stroke_data
)

# ---------- Diabetes ----------
df_diabetes = load_diabetes_data('data/diabetes.csv')
X_dia = df_diabetes.drop('Outcome', axis=1)
y_dia = df_diabetes['Outcome']
X_train, X_test, y_train, y_test = train_test_split(X_dia, y_dia, test_size=0.2, random_state=42)

model_diabetes = RandomForestClassifier(random_state=42)
model_diabetes.fit(X_train, y_train)
joblib.dump(model_diabetes, 'models/diabetes_model.pkl')

# ---------- Heart ----------
df_heart = load_heart_data('data/heart.csv')
features_heart = ["age", "sex", "cp", "trestbps", "chol", "fbs",
                  "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"]
X_heart = df_heart[features_heart]
y_heart = df_heart["target"]
X_train, X_test, y_train, y_test = train_test_split(X_heart, y_heart, test_size=0.2, random_state=42)

model_heart = RandomForestClassifier(random_state=42)
model_heart.fit(X_train, y_train)
joblib.dump(model_heart, 'models/heart_model.pkl')

# ---------- Kidney ----------
df_kidney = load_kidney_data('data/kidney_disease.csv')
X_kidney = df_kidney.drop('classification', axis=1)
y_kidney = df_kidney['classification']
X_train, X_test, y_train, y_test = train_test_split(X_kidney, y_kidney, test_size=0.2, random_state=42)

model_kidney = RandomForestClassifier(random_state=42)
model_kidney.fit(X_train, y_train)
joblib.dump(model_kidney, 'models/kidney_model.pkl')

# ---------- Stroke ----------
df_stroke = load_stroke_data('data/healthcare-dataset-stroke-data.csv')
X_stroke = df_stroke.drop('stroke', axis=1)
y_stroke = df_stroke['stroke']
X_train, X_test, y_train, y_test = train_test_split(X_stroke, y_stroke, test_size=0.2, random_state=42)

model_stroke = RandomForestClassifier(random_state=42)
model_stroke.fit(X_train, y_train)
joblib.dump(model_stroke, 'models/stroke_model.pkl')

print("✅ All models trained and saved successfully.")
