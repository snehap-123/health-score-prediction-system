import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# Load data
data = pd.read_csv("../data/health_data.csv")

# Features
# Create target column (Health Score)
data['Health_Score'] = (
    (100 - data['BMI']) * 0.3 +
    data['Sleep_Hours'] * 5 +
    data['Exercise_Frequency'] * 10 +
    data['Diet_Quality'] * 2
)

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