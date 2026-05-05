import pandas as pd
import numpy as np
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


# ==============================
# 1. Load Dataset
# ==============================
data_path = os.path.join(os.path.dirname(__file__), "../data/health_data.csv")
data = pd.read_csv(data_path)


# ==============================
# 2. Create Health Score (Balanced Logic)
# ==============================
def calculate_health_score(row):
    # ---- BMI (ideal: 18.5–24.9)
    if 18.5 <= row["BMI"] <= 24.9:
        bmi_score = 1.0
    else:
        if row["BMI"] < 18.5:
            dist = 18.5 - row["BMI"]
        else:
            dist = row["BMI"] - 24.9
        bmi_score = max(0, 1 - dist / 15)

    # ---- Sleep (ideal: 7–8 hours)
    if 7 <= row["Sleep_Hours"] <= 8:
        sleep_score = 1.0
    else:
        dist = abs(row["Sleep_Hours"] - 7.5)
        sleep_score = max(0, 1 - dist / 5)

    # ---- Exercise (ideal: 3–5 days/week)
    if 3 <= row["Exercise_Frequency"] <= 5:
        ex_score = 1.0
    else:
        if row["Exercise_Frequency"] < 3:
            dist = 3 - row["Exercise_Frequency"]
        else:
            dist = row["Exercise_Frequency"] - 5
        ex_score = max(0, 1 - dist / 5)

    # ---- Diet (1–5 → normalize)
    diet_score = (row["Diet_Quality"] - 1) / 4

    # ---- Weighted final score
    final_score = (
        0.3 * bmi_score +
        0.25 * sleep_score +
        0.25 * ex_score +
        0.2 * diet_score
    )

    return final_score * 100  # scale to 0–100


# Apply function
data["Health_Score"] = data.apply(calculate_health_score, axis=1)


# ==============================
# 3. Features & Target
# ==============================
X = data[['Age', 'BMI', 'Sleep_Hours', 'Exercise_Frequency', 'Diet_Quality']]
y = data['Health_Score']


# ==============================
# 4. Train-Test Split
# ==============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# ==============================
# 5. Train Model
# ==============================
model = LinearRegression()
model.fit(X_train, y_train)


# ==============================
# 6. Evaluate Model
# ==============================
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"MSE: {mse}")
print(f"R2 Score: {r2}")


# ==============================
# 7. Save Model
# ==============================
model_dir = os.path.join(os.path.dirname(__file__), "../model")
os.makedirs(model_dir, exist_ok=True)

model_path = os.path.join(model_dir, "model.pkl")
joblib.dump(model, model_path)

print("Model saved successfully!")
