from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load trained models
try:
    credit_score_model = joblib.load("credit_score_model.pkl")
    decision_rating_model = joblib.load("credit_decision_model.pkl")
except Exception as e:
    credit_score_model = None
    decision_rating_model = None
    print(f"Error loading models: {e}")

# Credit Score Prediction Function
def predict_credit_score(data):
    """Predicts credit score based on user financial data"""
    features = np.array(data).reshape(1, -1)
    
    if credit_score_model is None:
        raise ValueError("Credit score model not loaded.")
    
    return credit_score_model.predict(features)[0]

# Decision Rating Prediction Function
def rate_decision(data):
    """Rates loan decision based on credit score & financial factors"""
    features = np.array(data).reshape(1, -1)

    if decision_rating_model is None:
        raise ValueError("Decision rating model not loaded.")

    if features.shape[1] != 10:  # Ensure input size matches model training data
        raise ValueError(f"Feature shape mismatch, expected: 10, got {features.shape[1]}")

    return decision_rating_model.predict(features)[0]

@app.route("/", methods=["GET"])
def home():
    """Renders the home page"""
    return render_template("index.html")

@app.route("/predict_credit_score", methods=["POST"])
def credit_score_api():
    """API to predict credit score"""
    if credit_score_model is None:
        return jsonify({"error": "Credit score model not loaded"}), 500

    try:
        data = request.get_json()
        
        # Required features
        required_keys = ["income", "savings", "existing_loans", "employment_status", "expenses", "financial_goals"]
        
        if not all(key in data for key in required_keys):
            return jsonify({"error": f"Missing required input fields, expected: {required_keys}"}), 400
        
        # Prepare input features
        financial_data = [data[key] for key in required_keys]
        
        score = predict_credit_score(financial_data)
        return jsonify({"credit_score": float(score)})
    
    except ValueError as ve:
        return jsonify({"error": str(ve)}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/rate_decision", methods=["POST"])
def decision_rating_api():
    """API to rate loan decision"""
    if decision_rating_model is None:
        return jsonify({"error": "Decision rating model not loaded"}), 500

    try:
        data = request.get_json()
        
        # Required features for decision rating model
        required_keys = [
            "credit_score", "amount", "interest_rate", "tenure",
            "income", "savings", "existing_loans", "employment_status", "expenses", "financial_goals"
        ]

        if not all(key in data for key in required_keys):
            return jsonify({"error": f"Missing required input fields, expected: {required_keys}"}), 400

        # Prepare input features
        feature_vector = [data[key] for key in required_keys]
        
        rating = rate_decision(feature_vector)
        return jsonify({"decision_rating": float(rating)})
    
    except ValueError as ve:
        return jsonify({"error": str(ve)}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)