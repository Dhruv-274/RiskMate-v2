from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd

app = Flask(__name__)

fraud_model = joblib.load('models/fraud_detection_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/credit', methods=['POST'])
def credit():
    try:
        data = request.get_json()
        if not data:
            raise ValueError("No JSON data provided")
        transaction_frequency = float(data.get('transaction_frequency', 0))
        avg_transaction_amount = float(data.get('avg_transaction_amount', 0))
        fraud_history = float(data.get('fraud_history', 0))
        if not (0 <= fraud_history <= 1):
            raise ValueError("Fraud history must be between 0 and 1.")
        base_score = 500
        credit_score = base_score + (transaction_frequency * 1.5) \
                                     - (avg_transaction_amount * 0.1) \
                                     - (fraud_history * 100)
        credit_score = max(300, min(850, round(credit_score)))
        print(f"Transaction Frequency: {transaction_frequency}, "
              f"Average Transaction Amount: {avg_transaction_amount}, "
              f"Fraud History: {fraud_history}, "
              f"Calculated Credit Score: {credit_score}")
        return jsonify({"status": "success", "credit_score": credit_score})

    except Exception as e:
        print(f"Error calculating credit score: {e}")
        return jsonify({"status": "error", "message": str(e)})

@app.route('/predict_fraud', methods=['POST'])
def predict_fraud():
    try:
        data = request.get_json()
        if not data:
            raise ValueError("No transaction data provided.")
        required_features = [
            'norm_time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 
            'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20', 'V21', 
            'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'norm_amount'
        ]
        missing_features = [feature for feature in required_features if feature not in data]
        if missing_features:
            return jsonify({"status": "error", "message": f"Missing required features: {', '.join(missing_features)}"})

        ordered_data = [data[feature] for feature in required_features]

        df = pd.DataFrame([ordered_data], columns=required_features)

        prediction = fraud_model.predict(df)
        is_fraud = prediction[0] == -1  

        is_fraud = bool(is_fraud)  

        print(f"Fraud Detection Result: {'Fraudulent' if is_fraud else 'Legitimate'}")

        return jsonify({"status": "success", "fraud": is_fraud})

    except Exception as e:
        print(f"Error detecting fraud: {e}")
        return jsonify({"status": "error", "message": str(e)})

if __name__ == '__main__':
    app.run(debug=True)
