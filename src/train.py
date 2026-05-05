import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import numpy as np

# Load data
data = pd.read_csv("../data/health_data.csv")

# Features
# Create target column (Health Score)
def calculate_health_score(row):
    score = 100

    # BMI penalty
    if row["BMI"] > 25:
        score -= (row["BMI"] - 25) * 1.5
    elif row["BMI"] < 18.5:
        score -= (18.5 - row["BMI"]) * 1.5

    # Sleep penalty (ideal 7-8)
    if row["Sleep_Hours"] < 7:
        score -= (7 - row["Sleep_Hours"]) * 2
    elif row["Sleep_Hours"] > 8:
        score -= (row["Sleep_Hours"] - 8) * 2

    # Exercise penalty (ideal 3-5)
    if row["Exercise_Frequency"] < 3:
        score -= (3 - row["Exercise_Frequency"]) * 3
    elif row["Exercise_Frequency"] > 5:
        score -= (row["Exercise_Frequency"] - 5) * 1.5

    # Diet contribution
    score += row["Diet_Quality"] * 4

    return max(0, min(100, score))  # keep between 0-100

# Apply
data["Health_Score"] = data.apply(calculate_health_score, axis=1)
# Features
X = data[['Age', 'BMI', 'Sleep_Hours', 'Exercise_Frequency', 'Diet_Quality']]
y = data['Health_Score']

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Metrics
print("MSE:", mean_squared_error(y_test, y_pred))
print("R2:", r2_score(y_test, y_pred))

# Save model
joblib.dump(model, "../model/model.pkl")

print("Model saved successfully!")
