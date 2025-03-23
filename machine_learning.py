import pandas as pd
file_path = 'sleep_health.csv'
data = pd.read_csv(file_path)
print(data.head())


# Example DataFrame
data = pd.DataFrame({
    'gender': ['Male', 'Female', 'Female', 'Male'],
    'sleep_disorder': ['None', 'Insomnia', 'Sleep Apnea', 'None']
})

# Label encode gender
data['gender'] = data['gender'].map({'Male': 0, 'Female': 1})

# Label encode sleep disorder
data['sleep_disorder'] = data['sleep_disorder'].map({'None': 0, 'Insomnia': 1, 'Sleep Apnea': 2})

print(data)


from sklearn.model_selection import train_test_split

# Define independent (X) and dependent (y) variables
X = data[['sleep_duration', 'sleep_quality', 'sleep_disorder', 'Age', 'gender', 'heart_rate', 'blood_pressure']]
y = data['overall_health_risk']  

# Split into training and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.linear_model import LinearRegression

# Initialize model
model = LinearRegression()

# Train model
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'MSE: {mse:.2f}')
print(f'MAE: {mae:.2f}')
print(f'R²: {r2:.2f}')


