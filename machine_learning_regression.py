import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Load Dataset
file_path = 'sleep_health.csv'  # Update with actual path
data = pd.read_csv(file_path)

# Convert column names to lowercase for consistency
data.columns = data.columns.str.lower()

# Splitting blood pressure into systolic and diastolic
data[['systolic_bp', 'diastolic_bp']] = data['blood_pressure'].str.split('/', expand=True).astype(float)

# Encode categorical variables
data['gender'] = data['gender'].map({'Male': 0, 'Female': 1})
data['sleep_disorder'] = data['sleep_disorder'].map({'None': 0, 'Insomnia': 1, 'Sleep Apnea': 2})

# Drop rows with missing values (optional)
data = data.dropna()

# Generate target variable (overall health risk)
data['overall_health_risk'] = (
    (data['sleep_quality'] < 6) |  # Poor sleep quality
    (data['age'] > 40) | #Adding age factor
    (data['sleep_disorder'] > 1) | #Sleep disorder
    (data['systolic_bp'] > 130)  # High systolic blood pressure
).astype(int)  # Convert boolean to 0 (low risk) or 1 (high risk)

# Select independent and dependent variables
X = data[['sleep_duration', 'sleep_quality', 'sleep_disorder', 'age', 'gender', 'heart_rate', 'systolic_bp', 'diastolic_bp']]
y = data['overall_health_risk']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and Train Model (Using Ridge Regression)
model = Ridge(alpha=1.0)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Model Evaluation
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'MSE: {mse:.2f}')
print(f'MAE: {mae:.2f}')
print(f'R²: {r2:.2f}')

# Residual Analysis
residuals = y_test - y_pred

# Plot Residuals in Separate Figures
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Residuals Histogram
sns.histplot(residuals, bins=20, kde=True, ax=axes[0])
axes[0].set_title('Residuals Distribution')
axes[0].set_xlabel('Residuals')
axes[0].set_ylabel('Frequency')

# Residuals Plot
sns.scatterplot(x=y_pred, y=residuals, ax=axes[1])
axes[1].axhline(0, color='red', linestyle='--', linewidth=2)
axes[1].set_title('Residual Plot')
axes[1].set_xlabel('Predicted Values')
axes[1].set_ylabel('Residuals')

plt.tight_layout()
plt.show()
