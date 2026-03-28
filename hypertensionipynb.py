import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

# -------------------------------
# Load trained model
# -------------------------------
model = joblib.load("hypertension_model.pkl")  # make sure this file is in the same folder

# -------------------------------
# App title
# -------------------------------
st.title("Hypertension Prediction App")
st.write("Enter patient details to predict the likelihood of hypertension.")

# -------------------------------
# Sidebar for inputs
# -------------------------------
st.sidebar.header("Patient Information")

age = st.sidebar.number_input("Age", min_value=1, max_value=120, value=30)
bp_history = st.sidebar.selectbox("Blood Pressure History", ["Normal", "Pre-hypertension", "Hypertension"])
medication = st.sidebar.selectbox("Medication", ["Yes", "No"])
exercise = st.sidebar.selectbox("Exercise Level", ["Low", "Medium", "High"])
smoking = st.sidebar.selectbox("Smoking Status", ["Non-smoker", "Former", "Current"])

# -------------------------------
# Preprocessing function
# -------------------------------
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
    # Ensure all values are numeric
    return pd.DataFrame([[
        float(mapping[bp_history]),
        float(mapping[medication]),
        float(mapping[exercise]),
        float(mapping[smoking]),
        float(age)
    ]], columns=["BP_History", "Medication", "Exercise_Level", "Smoking_Status", "Age"])

# -------------------------------
# Get processed input
# -------------------------------
input_df = preprocess_input(bp_history, medication, exercise, smoking, age)

# -------------------------------
# Prediction
# -------------------------------
prediction = model.predict(input_df.values)[0]              # use .values to convert to numpy array
probability = model.predict_proba(input_df.values)[0][1]   # probability of hypertension

st.subheader("Prediction Results")
st.write(f"**Prediction:** {'Hypertension' if prediction == 1 else 'No Hypertension'}")
st.write(f"**Probability:** {probability:.2f}")

# -------------------------------
# SHAP Feature Explanation
# -------------------------------
try:
    explainer = shap.Explainer(model, input_df.values)
    shap_values = explainer(input_df.values)
    
    st.subheader("Feature Importance (SHAP Values)")
    plt.figure()
    shap.summary_plot(shap_values, input_df, show=False)
    st.pyplot(bbox_inches='tight')
except Exception as e:
    st.write("SHAP explanation not available:", e)
