import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
import joblib

# Load processed data
data = pd.read_csv("processed_data.csv")

# Define features and labels
X = data.drop('Class', axis=1)
y = data['Class']

# Ensure feature order matches the expected input
EXPECTED_FEATURES = ['norm_time'] + [f'V{i}' for i in range(1, 29)] + ['norm_amount']
X = X[EXPECTED_FEATURES]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Isolation Forest model
model = IsolationForest(contamination=0.2, random_state=42)
model.fit(X_train)

# Save the trained model
joblib.dump(model, 'fraud_detection_model.pkl')
print("Model training complete and saved as 'fraud_detection_model.pkl'")
