# AI Health Score Predictor

A machine learning web application that predicts a user's health score based on lifestyle factors like BMI, sleep, exercise, and diet.

---

## Live Demo
👉 (https://health-score-prediction-system-qkqhj7mzugga2czfy4wk43.streamlit.app/)

---

## Overview
This project uses a hybrid approach:
- A **Machine Learning model (Linear Regression)** predicts a base health score
- A **rule-based system** adjusts the score based on real-world health conditions

This ensures predictions are both **data-driven and realistic**

---

##  Features
- Predicts health score based on user input  
- Personalized health insights  
- Adjusted score using rule-based penalties  
- Interactive web app using Streamlit  
- Clean and user-friendly UI  

---

##  Tech Stack
- Python  
- Pandas, NumPy  
- Scikit-learn  
- Streamlit  

---

##  How It Works

### Model Prediction
The ML model predicts a base health score using:
- Age  
- BMI  
- Sleep Hours  
- Exercise Frequency  
- Diet Quality  

---

###  Adjusted Health Score
A rule-based system applies penalties:

| Factor | Condition | Impact |
|------|--------|--------|
| BMI | Out of range | Score ↓ |
| Sleep | <7 or >8 | Score ↓ |
| Exercise | Low | Score ↓ |
| Diet | Poor | Score ↓ |

👉 Final Score = Model Score − Penalties

---

## Project Structure

health-score-ml/
│
├── app/ # Streamlit UI
│ └── app.py
│
├── src/ # Training script
│ └── train.py
│
├── model/ # Saved model
│ └── model.pkl
│
├── data/ # Dataset
│ └── health_data.csv
│
├── requirements.txt
└── README.md


## Sample Output
- Health Score: 78.5  
- Adjusted Score: 65.2  
- Insights: Improve sleep and diet  

---

## Future Improvements
- Add charts & visual analytics  
- Use advanced models (Random Forest, XGBoost)  
- Deploy mobile-friendly version  
- Add user authentication  

---

##  Author
Sneha Patil
