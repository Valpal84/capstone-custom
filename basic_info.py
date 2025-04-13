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

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Display the first few rows of the dataset
print(data.head())

# Show basic information about dataset
print(data.info())

# Summary statistics
print(data.describe())

# Check for missing values
print(data.isnull().sum())

# Check class distribution
print(data['overall_health_risk'].value_counts())

import matplotlib.pyplot as plt
import seaborn as sns

# Set plot style
sns.set_style("whitegrid")

# Histogram for numerical features
data.hist(figsize=(12, 8), bins=20, edgecolor='black')
plt.suptitle("Feature Distributions", fontsize=16)
plt.show()

# Boxplot to identify outliers
plt.figure(figsize=(12, 6))
sns.boxplot(data=data)
plt.xticks(rotation=45)
plt.title("Boxplot of Features to Identify Outliers")
plt.show()

# Count plot for gender
sns.countplot(x='gender', hue='overall_health_risk', data=data, palette='pastel')
plt.title("Gender Distribution Across Health Risk Levels")
plt.show()

# Count plot for sleep disorder
sns.countplot(x='sleep_disorder', hue='overall_health_risk', data=data, palette='pastel')
plt.title("Sleep Disorder vs. Health Risk")
plt.show()

# Boxplot for heart rate
plt.figure(figsize=(6, 4))
sns.boxplot(x=data["age"])
plt.title("Outliers in Age")
plt.show()


from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier()
model.fit(X_train, y_train)

feature_importances = pd.Series(model.feature_importances_, index=X_train.columns)
feature_importances.nlargest(10).plot(kind='barh', colormap="coolwarm")
plt.title("Top 10 Feature Importances")
plt.show()
