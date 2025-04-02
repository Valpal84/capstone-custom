import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc

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
    (data['age'] > 40) | # Adding age factor
    (data['sleep_disorder'] > 1) | # Sleep disorder
    (data['systolic_bp'] > 130)  # High systolic blood pressure
).astype(int)  # Convert boolean to 0 (low risk) or 1 (high risk)

# Select independent and dependent variables
X = data[['sleep_duration', 'sleep_quality', 'sleep_disorder', 'age', 'gender', 'heart_rate', 'systolic_bp', 'diastolic_bp']]
y = data['overall_health_risk']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and Train Random Forest Model
model = RandomForestClassifier(random_state=42, class_weight='balanced')  # Using class weight for balance
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

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Feature Importance
feature_importances = model.feature_importances_
features = X.columns

# Plot Feature Importance
plt.figure(figsize=(10, 6))
sns.barplot(x=features, y=feature_importances)
plt.title("Feature Importance")
plt.ylabel("Importance")
plt.xticks(rotation=45)
plt.show()

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

# Plot Confusion Matrix
plt.figure(figsize=(6, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, 
            xticklabels=['Low Risk', 'High Risk'], yticklabels=['Low Risk', 'High Risk'])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# ROC Curve and AUC Score
fpr, tpr, _ = roc_curve(y_test, model.predict_proba(X_test)[:, 1])
roc_auc = auc(fpr, tpr)

# Plot ROC Curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
plt.show()

# Class Distribution
plt.figure(figsize=(6, 4))
sns.countplot(x=y, palette='pastel')
plt.title("Class Distribution (Low Risk vs High Risk)")
plt.xlabel("Health Risk")
plt.ylabel("Count")
plt.xticks([0, 1], ['Low Risk', 'High Risk'])
plt.show()
