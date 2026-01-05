import os
import xgboost as xgb
import pandas as pd
from flask import Flask, request, jsonify

app = Flask(__name__)

# Correctly point to the JSON file you just exported
model_filename = 'glof_model.json'

# Load the model
model = xgb.XGBRegressor()

try:
    model.load_model(model_filename)
    print("✅ Model brain loaded into the backend!")
except Exception as e:
    print(f"❌ Still can't find the file. Error: {e}")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # 1. Get data from frontend
        data = request.get_json()
        
        # 2. Convert to DataFrame (ensure columns match your model training)
        input_df = pd.DataFrame([data])
        
        # 3. Get raw prediction from model
        raw_prediction = float(model.predict(input_df)[0])
        
        # 4. Apply your risk thresholds
        # 0-0.3: Low, 0.31-0.6: Moderate, 0.61-0.8: High, 0.81-1: Immediate
        if raw_prediction <= 0.30:
            category = "LOW"
            color = "#2ecc71" # Green
        elif raw_prediction <= 0.60:
            category = "MODERATE"
            color = "#f1c40f" # Yellow
        elif raw_prediction <= 0.80:
            category = "HIGH"
            color = "#e67e22" # Orange
        else:
            category = "IMMEDIATE ACTION NEEDED"
            color = "#e74c3c" # Red

        # 5. Return both the score and the category
        return jsonify({
            'risk_index': round(raw_prediction, 4),
            'risk_level': category,
            'color': color,
            'status': 'success'
        })

    except Exception as e:
        return jsonify({'error': str(e), 'status': 'failed'})

if __name__ == '__main__':
    app.run(debug=True)