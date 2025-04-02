# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Load dataset
file_path = 'sleep_health.csv'  # Ensure the file is in the correct location
data = pd.read_csv(file_path)

# Convert column names to lowercase for consistency
data.columns = data.columns.str.lower()

# Encode categorical variables
data['gender'] = data['gender'].map({'Male': 0, 'Female': 1})
data['sleep_disorder'] = data['sleep_disorder'].map({'None': 0, 'Insomnia': 1, 'Sleep Apnea': 2})

# Handle missing values (optional)
data.dropna(inplace=True)

# Convert blood pressure from '120/80' format to separate systolic & diastolic values
data[['systolic_bp', 'diastolic_bp']] = data['blood_pressure'].str.split('/', expand=True).astype(float)

# Generate target variable (overall health risk)
data['overall_health_risk'] = (
    (data['sleep_quality'] < 5) |  # Poor sleep quality
    (data['sleep_duration'] < 6) |  # Less than 6 hours of sleep
    (data['heart_rate'] > 90) |  # High resting heart rate
    (data['systolic_bp'] > 130)  # High systolic blood pressure
).astype(int)  # Convert boolean to 0 (low risk) or 1 (high risk)

# Define independent (X) and dependent (y) variables
X = data[['sleep_duration', 'sleep_quality', 'sleep_disorder', 'age', 'gender', 'heart_rate', 'systolic_bp']]
y = data['overall_health_risk']

# Split data into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate model performance
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'MSE: {mse:.2f}')
print(f'MAE: {mae:.2f}')
print(f'R²: {r2:.2f}')

# Residual plot
residuals = y_test - y_pred
plt.figure(figsize=(8, 6))
sns.residplot(x=y_pred, y=residuals, lowess=True, line_kws={'color': 'red', 'linewidth': 2})
plt.axhline(y=0, color='black', linestyle='--')
plt.xlabel("Predicted Values")
plt.ylabel("Residuals")
plt.title("Residual Plot")
plt.show()
