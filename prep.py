import pandas as pd
from sklearn.preprocessing import StandardScaler
from imblearn.under_sampling import RandomUnderSampler

# Load the dataset
data = pd.read_csv("data/creditcard.csv")

# Normalize 'Time' and 'Amount'
scaler = StandardScaler()
data['norm_time'] = scaler.fit_transform(data[['Time']])
data['norm_amount'] = scaler.fit_transform(data[['Amount']])
data = data.drop(['Time', 'Amount'], axis=1)

# Handle class imbalance
X = data.drop('Class', axis=1)
y = data['Class']
undersample = RandomUnderSampler(sampling_strategy=0.5, random_state=42)
X_resampled, y_resampled = undersample.fit_resample(X, y)

# Save the preprocessed data
processed_data = pd.concat([pd.DataFrame(X_resampled, columns=X.columns), pd.Series(y_resampled, name="Class")], axis=1)
processed_data.to_csv("data/processed_data.csv", index=False)
print("Preprocessed data saved to 'processed_data.csv'")
