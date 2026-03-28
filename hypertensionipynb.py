import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the trained model
model = joblib.load("hypertension_model.pkl")

# -------------------------------------------------------------------
# Label encoding mappings — must match the LabelEncoder order used
# during training (alphabetical, as fitted by sklearn.LabelEncoder)
# -------------------------------------------------------------------
BP_HISTORY_MAP = {"Hypertension": 0, "Normal": 1, "Prehypertension": 2}
SMOKING_MAP    = {"Non-Smoker": 0, "Smoker": 1}
EXERCISE_MAP   = {"High": 0, "Low": 1, "Moderate": 2}
MEDICATION_MAP = {"ACE Inhibitor": 0, "Beta Blocker": 1, "Diuretic": 2, "Other": 3}
FAMILY_MAP     = {"No": 0, "Yes": 1}


def main():
    st.set_page_config(page_title="Hypertension Risk Predictor", layout="centered")

    st.title("🩺 Hypertension Risk Assessment")
    st.markdown(
        """
        Evaluate hypertension likelihood based on your demographic,
        lifestyle, and clinical profile using a trained machine learning model.
        """
    )

    st.sidebar.header("Patient Input Features")

    # ── Numeric inputs ──────────────────────────────────────────────
    age            = st.sidebar.number_input("Age", min_value=1, max_value=120, value=30)
    salt_intake    = st.sidebar.slider("Daily Salt Intake (g)", 0.0, 20.0, 5.0, 0.5)
    stress_score   = st.sidebar.slider("Stress Score (1–10)", 1, 10, 5)
    sleep_duration = st.sidebar.slider("Sleep Duration (hours/night)", 3.0, 12.0, 7.0, 0.5)
    bmi            = st.sidebar.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0, step=0.1)

    # ── Categorical inputs ──────────────────────────────────────────
    bp_history     = st.sidebar.selectbox("Blood Pressure History",
                                          list(BP_HISTORY_MAP.keys()))
    smoking_status = st.sidebar.selectbox("Smoking Status",
                                          list(SMOKING_MAP.keys()))
    exercise_level = st.sidebar.selectbox("Exercise Level",
                                          list(EXERCISE_MAP.keys()))
    medication     = st.sidebar.selectbox("Current Medication",
                                          list(MEDICATION_MAP.keys()))
    family_history = st.sidebar.selectbox("Family History of Hypertension",
                                          list(FAMILY_MAP.keys()))

    # ── Build encoded feature DataFrame ────────────────────────────
    # Column order must match the training data (minus the target):
    # Age, Salt_Intake, Stress_Score, BP_History, Sleep_Duration,
    # BMI, Medication, Family_History, Exercise_Level, Smoking_Status
    input_data = pd.DataFrame({
        "Age":            [age],
        "Salt_Intake":    [salt_intake],
        "Stress_Score":   [stress_score],
        "BP_History":     [BP_HISTORY_MAP[bp_history]],
        "Sleep_Duration": [sleep_duration],
        "BMI":            [bmi],
        "Medication":     [MEDICATION_MAP[medication]],
        "Family_History": [FAMILY_MAP[family_history]],
        "Exercise_Level": [EXERCISE_MAP[exercise_level]],
        "Smoking_Status": [SMOKING_MAP[smoking_status]],
    })

    # ── Prediction ──────────────────────────────────────────────────
    if st.button("🔍 Predict Risk"):
        prediction  = model.predict(input_data)
        probability = model.predict_proba(input_data)[0][1]

        st.subheader("📊 Results")
        if prediction[0] == 1:
            st.error(f"⚠️ **High Risk of Hypertension detected.**")
            st.metric("Confidence", f"{probability:.1%}")
        else:
            st.success("✅ **Low Risk of Hypertension detected.**")
            st.metric("Confidence", f"{(1 - probability):.1%}")

        st.info(
            "**Research Findings Summary:** Factors like high BMI, sleep deprivation, "
            "and high salt intake were identified as strong predictors in this study."
        )

        with st.expander("View input summary"):
            st.dataframe(pd.DataFrame({
                "Feature": ["Age", "Salt Intake (g)", "Stress Score", "Sleep Duration (h)",
                             "BMI", "BP History", "Smoking Status", "Exercise Level",
                             "Medication", "Family History"],
                "Value":   [age, salt_intake, stress_score, sleep_duration, bmi,
                             bp_history, smoking_status, exercise_level,
                             medication, family_history]
            }))


if __name__ == "__main__":
    main()
