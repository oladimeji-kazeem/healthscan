import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the trained model
# Ensure 'hypertension_model.pkl' is in the same directory 
model = joblib.load("hypertension_model.pkl")

def main():
    st.set_page_config(page_title="Hypertension Risk Predictor", layout="centered")
    
    st.title("🩺 Hypertension Risk Assessment")
    st.markdown("""
    Develop and evaluate machine learning models that can predict the likelihood 
    of hypertension based on demographic, lifestyle, and clinical factors. 
    """)

    st.sidebar.header("User Input Features")

    # Define input fields based on the research dataset features [cite: 7500, 7501, 7502, 7503]
    age = st.sidebar.number_input("Age", min_value=1, max_value=120, value=30)
    salt_intake = st.sidebar.slider("Daily Salt Intake (grams)", 0.0, 20.0, 5.0)
    stress_score = st.sidebar.slider("Stress Score (1-10)", 1, 10, 5)
    
    # Categorical Inputs [cite: 7504, 7508]
    bp_history = st.sidebar.selectbox("Blood Pressure History", ("Normal", "High", "Low"))
    smoking_status = st.sidebar.selectbox("Smoking Status", ("Non-smoker", "Former", "Current"))
    exercise_level = st.sidebar.selectbox("Exercise Level", ("Low", "Medium", "High"))

    # Mapping inputs for the model (matching the LabelEncoder logic)
    # Note: Ensure these mappings match the encoding used in your training notebook
    input_data = pd.DataFrame({
        'Age': [age],
        'Salt_Intake': [salt_intake],
        'Stress_Score': [stress_score],
        'BP_History': [bp_history],
        'Smoking_Status': [smoking_status],
        'Exercise_Level': [exercise_level]
    })

    # Prediction Button
    if st.button("Predict Risk"):
        # Process inputs (e.g., scaling or encoding if necessary)
        prediction = model.predict(input_data)
        probability = model.predict_proba(input_data)[0][1]

        st.subheader("Results")
        if prediction[0] == 1:
            st.error(f"High Risk of Hypertension detected.")
            st.write(f"Confidence Level: {probability:.2%}")
        else:
            st.success(f"Low Risk of Hypertension detected.")
            st.write(f"Confidence Level: {(1 - probability):.2%}")

        st.info("**Research Findings Summary:** Factors like High BMI, sleep deprivation, and high salt intake were identified as strong predictors in this study. [cite: 7502, 7503]")

if __name__ == "__main__":
    main()
