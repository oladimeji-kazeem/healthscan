import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap

# Load your model
model = joblib.load("hypertension_model.pkl")

# Title
st.title("Hypertension Prediction App")

# Sidebar inputs
st.sidebar.header("Patient Information")
age = st.sidebar.number_input("Age", min_value=1, max_value=120, value=30)
bp_history = st.sidebar.selectbox("Blood Pressure History", ["Normal", "Pre-hypertension", "Hypertension"])
medication = st.sidebar.selectbox("Medication", ["Yes", "No"])
exercise = st.sidebar.selectbox("Exercise Level", ["Low", "Medium", "High"])
smoking = st.sidebar.selectbox("Smoking Status", ["Non-smoker", "Former", "Current"])

# Preprocess input (convert categorical to numbers if your model needs)
def preprocess_input(bp_history, medication, exercise, smoking, age):
    mapping = {
        "Normal": 0,
        "Pre-hypertension": 1,
        "Hypertension": 2,
        "Yes": 1,
        "No": 0,
        "Low": 0,
        "Medium": 1,
        "High": 2,
        "Non-smoker": 0,
        "Former": 1,
        "Current": 2
    }
    return pd.DataFrame([[
        mapping[bp_history],
        mapping[medication],
        mapping[exercise],
        mapping[smoking],
        float(age)  # make sure age is float
    ]], columns=["BP_History", "Medication", "Exercise_Level", "Smoking_Status", "Age"])

input_df = preprocess_input(bp_history, medication, exercise, smoking, age)

# Predict
prediction = model.predict(input_df.values)[0]
probability = model.predict_proba(input_df.values)[0][1]

st.write(f"**Prediction:** {'Hypertension' if prediction == 1 else 'No Hypertension'}")
st.write(f"**Probability:** {probability:.2f}")

# Optional: SHAP explanation
explainer = shap.Explainer(model, input_df)
shap_values = explainer(input_df)
st.subheader("Feature Importance")
shap.summary_plot(shap_values, input_df, show=False)
st.pyplot(bbox_inches='tight')
