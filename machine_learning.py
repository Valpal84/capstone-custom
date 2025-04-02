import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Load Dataset
file_path = 'sleep_health.csv'
data = pd.read_csv(file_path)

# Standardize Column Names (Lowercase, Replace Spaces)
data.columns = data.columns.str.lower().str.replace(' ', '_')

# Convert Categorical Variables to Numerical (Label Encoding)
data['gender'] = data['gender'].map({'Male': 0, 'Female': 1})
data['sleep_disorder'] = data['sleep_disorder'].map({'None': 0, 'Insomnia': 1, 'Sleep Apnea': 2})

# Handle Missing Values (Fill NaN in Sleep Disorder)
data['sleep_disorder'] = data['sleep_disorder'].fillna(0)  # Assuming "None" if missing

# Convert Blood Pressure from String ("120/80") to Numerical (Systolic Only)
data['blood_pressure'] = data['blood_pressure'].str.split('/').str[0].astype(float)

# Create a Numerical "Overall Health Risk" Score (Based on Sleep & Heart Metrics)
data['overall_health_risk'] = (
    (10 - data['sleep_quality']) +  # Lower quality → higher risk
    (7 - data['sleep_duration']) +  # Very low/high sleep → higher risk
    (data['heart_rate'] / 10) +     # High heart rate → higher risk
    (data['blood_pressure'] / 10)   # High systolic BP → higher risk
)

# Define Features (X) and Target (y)
X = data[['sleep_duration', 'sleep_quality', 'sleep_disorder', 'age', 'gender', 'heart_rate', 'blood_pressure']]
y = data['overall_health_risk']

# Split Data (80% Training, 20% Testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Linear Regression Model
model = LinearRegression()
model.fit(X_train, y_train)

# Make Predictions
y_pred = model.predict(X_test)

# Evaluate Model Performance
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'MSE: {mse:.2f}')
print(f'MAE: {mae:.2f}')
print(f'R²: {r2:.2f}')

# Display Feature Importance (Coefficient Values)
coefficients = pd.DataFrame(model.coef_, X.columns, columns=['Coefficient'])
print("\nFeature Importance:\n", coefficients)

import matplotlib.pyplot as plt

print("Shape of y_test:", y_test.shape)
print("Shape of y_pred:", y_pred.shape)

# Check if there are any NaN or infinite values in y_pred
print("Are there any NaN or infinite values in y_pred?", np.any(np.isnan(y_pred)), np.any(np.isinf(y_pred)))

