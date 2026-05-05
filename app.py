import streamlit as st
import joblib
import numpy as np
import os
import pandas as pd

# Page config
st.set_page_config(page_title="AI Health Predictor", layout="centered")

# Load model
model_path = os.path.join(os.getcwd(), "model/model.pkl")
model = joblib.load(model_path)

# ---------- HEADER ----------
st.markdown("""
    <h1 style='text-align: center; color: #2E86C1;'>AI Health Score Predictor</h1>
    <p style='text-align: center; color: grey;'>Analyze your lifestyle and get instant health insights</p>
""", unsafe_allow_html=True)

st.divider()

st.markdown(
    "<style>body {background-color: #F4F6F7;}</style>",
    unsafe_allow_html=True
)

# ---------- INPUT SECTION ----------
st.subheader("Enter Your Details")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", 1, 100)
    bmi = st.number_input("BMI", 10.0, 40.0)

with col2:
    sleep = st.number_input("Sleep Hours", 1, 12)
    exercise = st.number_input("Exercise (days/week)", 0, 7)

diet = st.slider("Diet Quality (1 = Poor, 5 = Excellent)", 1, 5)

st.divider()

# ---------- PREDICT BUTTON ----------
if st.button("Predict Health Score"):

    input_data = pd.DataFrame([{
        "Age": age,
        "BMI": bmi,
        "Sleep_Hours": sleep,
        "Exercise_Frequency": exercise,
        "Diet_Quality": diet
    }])

    result = model.predict(input_data)[0]

    # ---------- RESULT CARD ----------
    st.markdown("## 📊 Result")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Health Score", f"{result:.2f}")

    # ---------- HEALTH STATUS ----------
    with col2:
        if result > 80:
            st.success("Excellent ✅")
        elif result > 60:
            st.info("Good 👍")
        else:
            st.warning("Needs Improvement ⚠️")

    st.divider()

    # ---------- INSIGHTS ----------
    st.subheader("💡 Personalized Insights")

    penalty = 0

    # BMI
    if bmi < 18.5 or bmi > 25:
        st.write("⚠️ BMI is not in healthy range")
        penalty += 10

    # Sleep
    if sleep < 7 or sleep > 8:
        st.write("😴 Sleep schedule is not optimal (7–8 hrs recommended)")
        penalty += 10

    # Exercise
    if exercise < 3:
        st.write("🏃 Increase exercise frequency")
        penalty += 10

    # Diet
    if diet <= 2:
        st.write("🥗 Poor diet quality - improve nutrition")
        penalty += 15

    # Adjust score
    adjusted_score = max(0, result - penalty)

    st.subheader("🎯 Adjusted Health Score")
    st.write(f"{adjusted_score:.2f}")

    # ---------- USER INPUT SUMMARY ----------
    st.subheader("📌 Your Inputs")

    st.write(f"**Age:** {age}")
    st.write(f"**BMI:** {bmi}")
    st.write(f"**Sleep Hours:** {sleep}")
    st.write(f"**Exercise Frequency:** {exercise}")
    st.write(f"**Diet Quality:** {diet}")
