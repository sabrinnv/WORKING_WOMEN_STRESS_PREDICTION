# ==========================================
# Working Women's Stress Risk Prediction
# ==========================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ----------------------------
# Load Saved Files
# ----------------------------

try:
    model = joblib.load("best_model.pkl")
    scaler = joblib.load("scaler.pkl")
    label_encoder = joblib.load("label_encoder.pkl")
except Exception as e:
    st.error(f"Error loading model files: {e}")
    st.stop()
# ----------------------------
# Page Configuration
# ----------------------------

st.set_page_config(
    page_title="Working Women's Stress Risk Prediction",
    page_icon="🧠",
    layout="wide"
)

# ----------------------------
# Sidebar
# ----------------------------

st.sidebar.title("📋 Project Information")

st.sidebar.info("""
**Working Women's Stress Risk Prediction**

This application predicts the stress level of a working woman using Machine Learning.

**Model Used:**
- Logistic Regression

**Target Classes**
- Low
- Moderate
- High
""")

# ----------------------------
# Main Title
# ----------------------------

st.title("🧠 Working Women's Stress Risk Prediction")

st.markdown("""
Enter the required details below and click **Predict Stress Level** to view the prediction.
""")

# ==========================================
# Personal Information
# ==========================================

st.header("👤 Personal Information")

col1, col2 = st.columns(2)

with col1:

    age = st.slider(
        "Age",
        min_value=20,
        max_value=60,
        value=30
    )

    children = st.slider(
        "Number of Children",
        min_value=0,
        max_value=6,
        value=1
    )

    marital_status = st.selectbox(
        "Marital Status",
        [
            "Divorced",
            "Married",
            "Single",
            "Widowed"
        ]
    )

with col2:

    education = st.selectbox(
        "Education",
        [
            "Graduate",
            "High School",
            "Postgraduate"
        ]
    )

    occupation = st.selectbox(
        "Occupation",
        [
            "Business",
            "Government Job",
            "Private Job",
            "Self-Employed"
        ]
    )

    industry = st.selectbox(
        "Industry",
        [
            "Education",
            "Finance",
            "Healthcare",
            "IT",
            "Manufacturing"
        ]
    )

# ==========================================
# Work Information
# ==========================================

st.header("💼 Work Information")

col3, col4 = st.columns(2)

with col3:

    working_hours = st.slider(
        "Working Hours",
        4,
        16,
        8
    )

    work_from_home = st.selectbox(
        "Work From Home",
        ["No", "Yes"]
    )

    job_satisfaction = st.slider(
        "Job Satisfaction",
        1,
        10,
        5
    )

with col4:

    team_support = st.slider(
        "Team Support",
        1,
        10,
        5
    )

    commute_time = st.slider(
        "Commute Time (minutes)",
        0,
        120,
        30
    )

# ==========================================
# Health Information
# ==========================================

st.header("❤️ Health Information")

col5, col6 = st.columns(2)

with col5:

    sleep_hours = st.slider(
        "Sleep Hours",
        3,
        10,
        7
    )

    exercise_frequency = st.slider(
        "Exercise Frequency (Days/Week)",
        0,
        7,
        3
    )

with col6:

    bmi = st.slider(
        "BMI",
        15.0,
        40.0,
        24.0
    )

    blood_pressure = st.slider(
        "Blood Pressure",
        80,
        180,
        120
    )

# ==========================================
# Family Information
# ==========================================

st.header("👨‍👩‍👧 Family Information")

col7, col8 = st.columns(2)

with col7:

    family_support = st.slider(
        "Family Support",
        1,
        10,
        5
    )

    elderly_care = st.selectbox(
        "Responsible for Elderly Care?",
        ["No", "Yes"]
    )

with col8:

    childcare = st.selectbox(
        "Responsible for Childcare?",
        ["No", "Yes"]
    )

# ==========================================
# Predict Button
# ==========================================

predict = st.button(
    "🔍 Predict Stress Level",
    use_container_width=True
)
# ==========================================
# Prediction
# ==========================================

if predict:

    input_data = {
        "Age": age,
        "Number_of_Children": children,
        "Working_Hours": working_hours,
        "Work_From_Home": 1 if work_from_home == "Yes" else 0,
        "Job_Satisfaction": job_satisfaction,
        "Team_Support": team_support,
        "Sleep_Hours": sleep_hours,
        "Exercise_Frequency": exercise_frequency,
        "Commute_Time": commute_time,
        "BMI": bmi,
        "Blood_Pressure": blood_pressure,
        "Family_Support": family_support,
        "Elderly_Care": 1 if elderly_care == "Yes" else 0,
        "Childcare": 1 if childcare == "Yes" else 0,

        "Marital_Status_Married": 0,
        "Marital_Status_Single": 0,
        "Marital_Status_Widowed": 0,

        "Occupation_Government Job": 0,
        "Occupation_Private Job": 0,
        "Occupation_Self-Employed": 0,

        "Education_High School": 0,
        "Education_Postgraduate": 0,

        "Industry_Finance": 0,
        "Industry_Healthcare": 0,
        "Industry_IT": 0,
        "Industry_Manufacturing": 0
    }

    # Marital Status

    if marital_status == "Married":
        input_data["Marital_Status_Married"] = 1

    elif marital_status == "Single":
        input_data["Marital_Status_Single"] = 1

    elif marital_status == "Widowed":
        input_data["Marital_Status_Widowed"] = 1


    # Occupation

    if occupation == "Government Job":
        input_data["Occupation_Government Job"] = 1

    elif occupation == "Private Job":
        input_data["Occupation_Private Job"] = 1

    elif occupation == "Self-Employed":
        input_data["Occupation_Self-Employed"] = 1


    # Education

    if education == "High School":
        input_data["Education_High School"] = 1

    elif education == "Postgraduate":
        input_data["Education_Postgraduate"] = 1


    # Industry

    if industry == "Finance":
        input_data["Industry_Finance"] = 1

    elif industry == "Healthcare":
        input_data["Industry_Healthcare"] = 1

    elif industry == "IT":
        input_data["Industry_IT"] = 1

    elif industry == "Manufacturing":
        input_data["Industry_Manufacturing"] = 1


    input_df = pd.DataFrame([input_data])

    feature_order = [
        'Age',
        'Number_of_Children',
        'Working_Hours',
        'Work_From_Home',
        'Job_Satisfaction',
        'Team_Support',
        'Sleep_Hours',
        'Exercise_Frequency',
        'Commute_Time',
        'BMI',
        'Blood_Pressure',
        'Family_Support',
        'Elderly_Care',
        'Childcare',
        'Marital_Status_Married',
        'Marital_Status_Single',
        'Marital_Status_Widowed',
        'Occupation_Government Job',
        'Occupation_Private Job',
        'Occupation_Self-Employed',
        'Education_High School',
        'Education_Postgraduate',
        'Industry_Finance',
        'Industry_Healthcare',
        'Industry_IT',
        'Industry_Manufacturing'
    ]

    input_df = input_df[feature_order]

        # -------------------------------------
    # Scale Numerical Features
    # -------------------------------------

    numerical_columns = [
        "Age",
        "Number_of_Children",
        "Working_Hours",
        "Job_Satisfaction",
        "Team_Support",
        "Sleep_Hours",
        "Exercise_Frequency",
        "Commute_Time",
        "BMI",
        "Blood_Pressure",
        "Family_Support"
    ]

    input_df[numerical_columns] = scaler.transform(
        input_df[numerical_columns]
    )

    # -------------------------------------
    # Prediction
    # -------------------------------------

    prediction = model.predict(input_df)[0]

    probabilities = model.predict_proba(input_df)[0]

    stress_level = label_encoder.inverse_transform([prediction])[0]

    confidence = np.max(probabilities) * 100

    st.divider()

    st.subheader("🎯 Prediction Result")

    if stress_level == "Low":

        st.success(f"Predicted Stress Level : **{stress_level}**")

    elif stress_level == "Moderate":

        st.warning(f"Predicted Stress Level : **{stress_level}**")

    else:

        st.error(f"Predicted Stress Level : **{stress_level}**")


    st.metric(
        label="Prediction Confidence",
        value=f"{confidence:.2f}%"
    )

    st.divider()

    st.subheader("💡 Recommendations")

    if stress_level == "Low":

        st.success("""
✅ Maintain your healthy lifestyle.

• Continue regular exercise.
• Maintain good sleep.
• Keep a healthy work-life balance.
• Continue social and family interaction.
""")

    elif stress_level == "Moderate":

        st.warning("""
⚠ Moderate Stress Detected

• Improve sleep schedule.
• Take regular breaks at work.
• Practice meditation or yoga.
• Increase physical activity.
• Spend quality time with family.
""")

    else:

        st.error("""
🚨 High Stress Detected

• Consult a mental health professional.
• Reduce excessive working hours.
• Improve work-life balance.
• Seek emotional support from family and friends.
• Practice mindfulness and relaxation techniques.
""")