import streamlit as st
import numpy as np
import pandas as pd
import joblib

# page config
st.set_page_config(
    page_title = "Heart Disease Predictor",
    page_icon = "❤️",
    layout = "centered"
)

# loading the trained model
model = joblib.load("heart_disease_model.pkl")

st.markdown("<h1 style = 'text-align : center; color : crimson;'>Heart Disease Prediction App</h1>", unsafe_allow_html = True)
st.markdown("<hr>", unsafe_allow_html = True)

st.write("Provide the patient's medical information below : ")
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value = 20, max_value = 100, step = 1)
    sex = st.selectbox("Sex", ["Female", "Male"])
    chest_pain_type = st.selectbox("Chest Pain Type (0-3)", ['Typical Angina', 'Atypical Angina', 'Non-Angina Pain', 'Asympotomatic'])
    resting_bp = st.number_input("Resting BP (systolic)", min_value = 80, max_value = 200)
    cholesterol = st.number_input("Cholesterol (mg/dl)", min_value = 100, max_value = 600)
with col2:
    fasting_blood_sugar = st.selectbox("Fasting Blood Suagr > 120 mg/dl", ['Yes', 'No'])
    resting_ecg = st.selectbox("Resting ECG (0-2)", ['Normal', 'st_t_abnormality', 'lvh_estes'])
    max_heart_rate = st.number_input("Max Heart Rate", min_value = 60, max_value = 220)
    exercise_angina = st.selectbox("Exercise Agina (Yes = 1, No = 0)", ['Yes', 'No'])
    oldpeak = st.number_input("Oldpeak (ST depression)", step = 0.1, format = "%.1f")
    st_slope = st.selectbox("ST Slope (0-2)", ['Upward', 'Flat', 'Downward'])

columns = ['age', 'sex', 'chest_pain_type', 'resting_bp_s', 'cholesterol',
       'fasting_blood_sugar', 'resting_ecg', 'max_heart_rate',
       'exercise_angina', 'oldpeak', 'ST_slope']
sex_d = {'Female' : 0, 'Male' : 1}
chest_d = {'Typical Angina' : 0, 
           'Atypical Angina' : 1,
           'Non-Angina Pain' : 2,
           'Asympotomatic' : 3}
fasting_bp_s_d = {'Yes' : 1,
                  'No' : 0}
ecg_d = {'Normal' : 0,
         'st_t_abnormality' : 1,
         'lvh_estes' : 2}
angina_d = {'Yes' : 1,
            'No' : 0}
slope_d = {'Upward' : 1,
           'Flat' : 2,
           'Downward' : 3}

if st.button("🔍 Predict Heart Disease"):
    sex = sex_d[sex]
    chest_pain_type = chest_d[chest_pain_type]
    fasting_blood_sugar = fasting_bp_s_d[fasting_blood_sugar]
    resting_ecg = ecg_d[resting_ecg]
    exercise_angina = angina_d[exercise_angina]
    st_slope = slope_d[st_slope]
    user_input = [age, sex, chest_pain_type, resting_bp, cholesterol, fasting_blood_sugar, resting_ecg, max_heart_rate, exercise_angina, oldpeak, st_slope]
    user_input_df = pd.DataFrame([user_input],  columns=columns)

    prediction = model.predict(user_input_df)[0]
    probability = model.predict_proba(user_input_df)[0][prediction]
    if prediction == 1:
        st.error(f"⚠️ High risk of heart disease! (Confidence : {probability:.2%})")
    else:
        st.success(f"✅ Low risk of heart disease. (Confidence : {probability:.2%})")

with st.expander("ℹ️ View Model Info"):
    st.markdown("""
    ### 🧠 Model Details
    - **Model Type:** Random Forest Classifier (with preprocessing pipeline)
    - **Features Used:** 
        - Age, Sex, Chest Pain Type, Resting BP, Cholesterol  
        - Fasting Blood Sugar, Resting ECG, Max Heart Rate, Exercise Angina  
        - Oldpeak (ST depression), ST Slope
    - **Target Variable:** Presence of Heart Disease (`target`)
    - **Preprocessing:** 
        - One-hot encoding for categorical variables
        - Standard scaling for numerical variables
        - Winsorizing outliers (IQR method)
    
    
    ---
    ### 📊 How to Interpret Output
    - **0 = Low Risk**, **1 = High Risk**  
    - Confidence score is the model's probability for the predicted class
    - This is a statistical estimate — not a medical diagnosis!

    🧬 *Built using scikit-learn & Streamlit*
    """)