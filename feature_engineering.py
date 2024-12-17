import pandas as pd
import numpy as np

# Load Kaggle dataset
data = pd.read_csv("data/creditcard.csv")

# Add synthetic customer IDs (assuming each row is a unique transaction for a unique customer)
data["customer_id"] = np.random.randint(1000, 1100, data.shape[0])  # Random customer IDs

# Derive features
customer_data = data.groupby("customer_id").agg(
    transaction_frequency=("Amount", "count"),
    avg_transaction_amount=("Amount", "mean"),
    fraud_history=("Class", "mean")  # Fraction of fraudulent transactions
).reset_index()

# Create a synthetic "default" column for credit scoring (based on fraud history)
customer_data["default"] = (customer_data["fraud_history"] > 0.2).astype(int)

# Save as credit_scoring.csv
customer_data.to_csv("data/credit_scoring.csv", index=False)
print("Credit scoring dataset created: data/credit_scoring.csv")
