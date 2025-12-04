import streamlit as st
import pandas as pd
import joblib

# Set a wide page configuration for a better layout
st.set_page_config(layout="wide")

# Load saved model, scaler, and expected columns
# Assuming these files are correctly located relative to the script
try:
    model = joblib.load("knn_heart_model.pkl")
    scaler = joblib.load("heart_scaler.pkl")
    expected_columns = joblib.load("heart_columns.pkl")
except FileNotFoundError:
    st.error("Error: Model or column files not found. Please ensure 'knn_heart_model.pkl', 'heart_scaler.pkl', and 'heart_columns.pkl' are in the same directory.")
    st.stop() # Stop execution if files are missing

# --- Header Section ---
st.title("💖 Heart Health Risk Predictor")
st.markdown(
    """
    **Welcome!** Please provide your health details below to assess your potential heart disease risk.
    This prediction is based on a machine learning model and should **not** replace professional medical advice.
    """
)
st.markdown("---")

# --- Input Section ---
with st.container(border=True):
    st.subheader("👤 Personal and Vitals")

    # Use columns to group related inputs side-by-side
    col1, col2, col3 = st.columns(3)

    with col1:
        # Using st.slider for Age is good, adding a metric display for a modern touch
        age = st.slider("🗓️ Age (Years)", 18, 100, 40)
        # st.metric is great for showing selected values
        st.metric(label="Selected Age", value=f"{age} years")

        sex = st.selectbox("🚻 Sex", ["M", "F"])
        
    with col2:
        resting_bp = st.number_input("🩸 Resting Blood Pressure (mm Hg)", 80, 200, 120, help="Blood pressure at rest.")
        cholesterol = st.number_input("🧈 Cholesterol (mg/dL)", 100, 600, 200, help="Serum cholesterol level.")
        
    with col3:
        fasting_bs = st.selectbox("🍩 Fasting Blood Sugar > 120 mg/dL", [0, 1], format_func=lambda x: "Yes (1)" if x == 1 else "No (0)", help="1 if sugar is high, 0 otherwise.")
        resting_ecg = st.selectbox("📉 Resting ECG", ["Normal", "ST", "LVH"], help="Resting Electrocardiogram results.")


# Using an expander to optionally hide less critical or complex inputs
with st.expander("🩺 Advanced Clinical Details", expanded=True):
    col4, col5, col6 = st.columns(3)

    with col4:
        chest_pain = st.selectbox("💔 Chest Pain Type", ["ATA", "NAP", "TA", "ASY"], help="Atypical Angina (ATA), Non-Anginal Pain (NAP), Typical Angina (TA), Asymptomatic (ASY).")
        st_slope = st.selectbox("📈 ST Slope", ["Up", "Flat", "Down"], help="The slope of the peak exercise ST segment.")
        
    with col5:
        max_hr = st.slider("🏃‍♂️ Max Heart Rate (BPM)", 60, 220, 150, help="Maximum heart rate achieved during exercise.")
        exercise_angina = st.selectbox("🥵 Exercise-Induced Angina", ["Y", "N"], help="Y for Yes, N for No.")

    with col6:
        oldpeak = st.slider("⛰️ Oldpeak (ST Depression)", 0.0, 6.0, 1.0, step=0.1, help="ST depression induced by exercise relative to rest.")

st.markdown("---")

# --- Prediction Button and Logic ---
if st.button("🚀 Run Risk Prediction", type="primary"):
    
    # Show a spinner while processing
    with st.spinner('Analyzing health metrics...'):
        
        # Create a raw input dictionary
        raw_input = {
            'Age': age,
            'RestingBP': resting_bp,
            'Cholesterol': cholesterol,
            'FastingBS': fasting_bs,
            'MaxHR': max_hr,
            'Oldpeak': oldpeak,
            'Sex_' + sex: 1,
            'ChestPainType_' + chest_pain: 1,
            'RestingECG_' + resting_ecg: 1,
            'ExerciseAngina_' + exercise_angina: 1,
            'ST_Slope_' + st_slope: 1
        }

        # Create input dataframe
        input_df = pd.DataFrame([raw_input])

        # Fill in missing columns with 0s
        for col in expected_columns:
            if col not in input_df.columns:
                input_df[col] = 0

        # Reorder columns
        input_df = input_df[expected_columns]

        # Scale the input
        scaled_input = scaler.transform(input_df)

        # Make prediction
        prediction = model.predict(scaled_input)[0]

    # --- Show Result ---
    st.markdown("## 📊 Prediction Result")
    
    if prediction == 1:
        st.error("🚨 HIGH RISK ALERT! Immediate Consultation Recommended.", icon="❌")
        st.markdown(
            """
            Based on the provided data, the model indicates a **High Risk** of heart disease.
            Please consult a healthcare professional for a proper diagnosis and guidance.
            """
        )
    else:
        st.success("✅ Low Risk Indication", icon="👍")
        st.balloons()
        st.markdown(
            """
            Based on the provided data, the model suggests a **Low Risk** of heart disease.
            Continue to monitor your health and maintain a healthy lifestyle.
            """
        )

st.markdown("---")
st.caption("Disclaimer: This tool is for informational purposes only. Do not use it for self-diagnosis or self-treatment.")