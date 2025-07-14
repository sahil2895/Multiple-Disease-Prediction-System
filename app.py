# app.py

import streamlit as st
import pandas as pd
import joblib

# Load models
diabetes_model = joblib.load("models/diabetes_model.pkl")
heart_model = joblib.load("models/heart_model.pkl")
kidney_model = joblib.load("models/kidney_model.pkl")
stroke_model = joblib.load("models/stroke_model.pkl")

# ---------- Styling ----------
st.markdown("""
    <style>
    html, body, [class*="css"] {
        font-family: 'Segoe UI', sans-serif;
        background-color: #f4f6f8;
        color: #333;
    }
    .main-title {
        text-align: center;
        font-size: 36px;
        color: #004d4d;
        margin-bottom: 10px;
    }
    .stButton>button {
        background-color: #008080;
        color: white;
        font-size: 16px;
        padding: 8px 20px;
        border-radius: 10px;
        border: none;
        margin-top: 10px;
    }
    .stButton>button:hover {
        background-color: #006666;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">🩺 Multiple Disease Prediction System</div>', unsafe_allow_html=True)

disease = st.selectbox("Select a disease to predict:", ["Diabetes", "Heart Disease", "Kidney Disease", "Stroke"])

# ---------- Diabetes ----------
if disease == "Diabetes":
    st.subheader("🩸 Diabetes Inputs")

    pregnancies = st.number_input("Pregnancies", 0, 20)
    glucose = st.number_input("Glucose", 0, 200)
    blood_pressure = st.number_input("Blood Pressure", 0, 150)
    skin_thickness = st.number_input("Skin Thickness", 0, 100)
    insulin = st.number_input("Insulin", 0, 900)
    bmi = st.number_input("BMI", 0.0, 70.0)
    diabetes_pedigree = st.number_input("Diabetes Pedigree Function", 0.0, 2.5)
    age = st.number_input("Age", 1, 120)

    if st.button("🔍 Predict Diabetes"):
        input_df = pd.DataFrame([[pregnancies, glucose, blood_pressure, skin_thickness,
                                  insulin, bmi, diabetes_pedigree, age]],
                                columns=["Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
                                         "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"])
        prediction = diabetes_model.predict(input_df)[0]
        st.success("✅ Not Diabetic" if prediction == 0 else "⚠️ Diabetic")

# ---------- Heart Disease ----------
elif disease == "Heart Disease":
    st.subheader("❤️ Heart Inputs")

    age = st.number_input("Age", 1, 120)
    sex = st.selectbox("Sex", ["Male", "Female"])
    cp = st.number_input("Chest Pain Type (0-3)", 0, 3)
    trestbps = st.number_input("Resting Blood Pressure", 0, 200)
    chol = st.number_input("Cholesterol", 0, 600)
    fbs = st.selectbox("Fasting Blood Sugar > 120?", ["Yes", "No"])
    restecg = st.number_input("Resting ECG (0-2)", 0, 2)
    thalach = st.number_input("Max Heart Rate", 0, 250)
    exang = st.selectbox("Exercise Induced Angina?", ["Yes", "No"])
    oldpeak = st.number_input("Oldpeak", 0.0, 10.0)
    slope = st.number_input("Slope (0-2)", 0, 2)
    ca = st.number_input("Number of Major Vessels (0-3)", 0, 3)
    thal = st.number_input("Thal (1=Normal, 2=Fixed, 3=Reversible)", 1, 3)

    if st.button("🔍 Predict Heart Disease"):
        input_df = pd.DataFrame([[
            age, 1 if sex == "Male" else 0, cp, trestbps, chol,
            1 if fbs == "Yes" else 0, restecg, thalach,
            1 if exang == "Yes" else 0, oldpeak, slope, ca, thal
        ]], columns=[
            "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
            "thalach", "exang", "oldpeak", "slope", "ca", "thal"
        ])
        prediction = heart_model.predict(input_df)[0]
        st.success("✅ No Heart Disease" if prediction == 0 else "⚠️ Risk of Heart Disease")

# ---------- Kidney ----------
elif disease == "Kidney Disease":
    st.subheader("🧫 Kidney Inputs")

    sg = st.number_input("Specific Gravity", 1.000, 1.030, step=0.001)
    al = st.number_input("Albumin", 0, 5)
    su = st.number_input("Sugar", 0, 5)
    bgr = st.number_input("Blood Glucose Random", 0, 500)
    bu = st.number_input("Blood Urea", 0, 200)
    sc = st.number_input("Serum Creatinine", 0.0, 15.0)
    hemo = st.number_input("Hemoglobin", 3.0, 17.5)
    pcv = st.number_input("Packed Cell Volume", 0, 55)
    wc = st.number_input("WBC Count", 0, 18000)
    rc = st.number_input("RBC Count", 0.0, 8.0)

    if st.button("🔍 Predict Kidney Disease"):
        input_df = pd.DataFrame([[sg, al, su, bgr, bu, sc, hemo, pcv, wc, rc]],
                                columns=["sg", "al", "su", "bgr", "bu", "sc", "hemo", "pcv", "wc", "rc"])
        prediction = kidney_model.predict(input_df)[0]
        st.success("✅ No Kidney Disease" if prediction == 0 else "⚠️ Kidney Disease Detected")

# ---------- Stroke ----------
elif disease == "Stroke":
    st.subheader("🧠 Stroke Inputs")

    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    age = st.number_input("Age", 1, 120)
    hypertension = st.selectbox("Hypertension", ["Yes", "No"])
    heart_disease = st.selectbox("Heart Disease", ["Yes", "No"])
    ever_married = st.selectbox("Ever Married", ["Yes", "No"])
    work_type = st.selectbox("Work Type", ["Private", "Self-employed", "Govt_job", "children", "Never_worked"])
    residence_type = st.selectbox("Residence Type", ["Urban", "Rural"])
    avg_glucose = st.number_input("Avg Glucose Level", 0.0, 300.0)
    bmi = st.number_input("BMI", 10.0, 60.0)
    smoking_status = st.selectbox("Smoking Status", ["never smoked", "formerly smoked", "smokes", "Unknown"])

    if st.button("🔍 Predict Stroke Risk"):
        input_df = pd.DataFrame([[
            1 if gender == "Male" else 0 if gender == "Female" else 2,
            age,
            1 if hypertension == "Yes" else 0,
            1 if heart_disease == "Yes" else 0,
            1 if ever_married == "Yes" else 0,
            {"Private": 0, "Self-employed": 1, "Govt_job": 2, "children": 3, "Never_worked": 4}[work_type],
            1 if residence_type == "Urban" else 0,
            avg_glucose,
            bmi,
            {"never smoked": 0, "formerly smoked": 1, "smokes": 2, "Unknown": 3}[smoking_status]
        ]], columns=[
            'gender', 'age', 'hypertension', 'heart_disease', 'ever_married',
            'work_type', 'Residence_type', 'avg_glucose_level', 'bmi', 'smoking_status'
        ])

        prediction = stroke_model.predict(input_df)[0]
        st.success("✅ Low Risk of Stroke" if prediction == 0 else "⚠️ High Risk of Stroke")
