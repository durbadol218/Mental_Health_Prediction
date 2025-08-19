import streamlit as st
import pandas as pd
import joblib
import pickle
import numpy as np

# Load model, scaler, training columns, and feature defaults
model = joblib.load("models/model.pkl")
scaler = joblib.load("models/scaler.pkl")
with open("models/columns.pkl", "rb") as f:
    X_train_columns = pickle.load(f)
with open("models/feature_defaults.pkl", "rb") as f:
    feature_defaults = pickle.load(f)

st.set_page_config(page_title="Mental Health Prediction", layout="centered")
st.title("🧠 Mental Health Prediction App")
st.write("Provide your details below to predict your mental health condition.")

# User input
age = st.number_input("Age", min_value=10, max_value=100, value=25)
gender = st.selectbox("Gender", ["Male", "Female", "Other"])
work_pressure = st.slider("Work Pressure (1 = Very Low, 10 = Very High)", 1, 10, 5)
sleep_duration = st.slider("Sleep Duration (hours)", 0, 12, 7)
diet_quality = st.selectbox("Diet Quality", ["Poor", "Average", "Good"])
study_satisfaction = st.slider("Study Satisfaction (1 = Very Low, 10 = Very High)", 1, 10, 5)
financial_stress = st.slider("Financial Stress (1 = Very Low, 10 = Very High)", 1, 10, 5)
family_history = st.selectbox("Family History of Mental Illness", ["Yes", "No"])

# Mapping categorical inputs
gender_map = {"Male": 0, "Female": 1, "Other": 2}
diet_map = {"Poor": 0, "Average": 1, "Good": 2}
family_map = {"Yes": 1, "No": 0}

# Create DataFrame with all training columns
input_df = pd.DataFrame(columns=X_train_columns)

# Fill user inputs for known features
known_features = {
    'Age': age,
    'Gender': gender_map[gender],
    'Work Pressure': work_pressure,
    'Sleep Duration': sleep_duration,
    'Diet Quality': diet_map[diet_quality],
    'Study Satisfaction': study_satisfaction,
    'Financial Stress': financial_stress,
    'Family History': family_map[family_history]
}

for col, val in known_features.items():
    if col in input_df.columns:
        input_df.loc[0, col] = val

# Fill missing features with training median/mode
for col in input_df.columns:
    if pd.isna(input_df[col]).any():
        input_df[col].fillna(feature_defaults[col], inplace=True)

# Scale input
input_scaled = scaler.transform(input_df)

# Predict button
if st.button("Predict Mental Health Status"):
    prediction = model.predict(input_scaled)[0]

    if prediction == 0:
        st.success("✅ Prediction: No Mental Health Issues Detected")
    else:
        st.error("⚠️ Prediction: At Risk of Mental Health Issues")

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(input_scaled)[0][1]
        st.write(f"**Prediction Probability (Risk Level):** {proba:.2f}")
