import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib

# Load the credit scoring dataset
data = pd.read_csv("data/credit_scoring.csv")

# Define features and target
X = data.drop(["customer_id", "default"], axis=1)
y = data["default"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Save the model
joblib.dump(model, "models/credit_scoring_model.pkl")
print("Credit scoring model trained and saved.")

# Evaluate model
y_pred = model.predict(X_test)
print("Evaluation Report:")
print(classification_report(y_test, y_pred))
