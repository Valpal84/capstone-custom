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

