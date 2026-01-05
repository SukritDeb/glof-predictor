from flask import Flask, request, jsonify
import xgboost as xgb
import pandas as pd

app = Flask(__name__)

# 1. Load the model globally so it stays in memory
model = xgb.XGBRegressor()
model.load_model('glof_model.json')

# Define the expected feature names (must match your training columns)
FEATURES = ['Lake_Area_km2', 'Dam_Slope_deg', 'Lake_Temp_C', 'Elevation_m']

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # 2. Receive JSON data from the user/frontend
        data = request.get_json()
        
        # 3. Convert input to DataFrame for the model
        # Input format: {"Lake_Area_km2": 0.8, "Dam_Slope_deg": 25, "Lake_Temp_C": 4.1, "Elevation_m": 5000}
        input_df = pd.DataFrame([data], columns=FEATURES)
        
        # 4. Perform Prediction
        prediction = model.predict(input_df)[0]
        
        # 5. Determine Risk Level Logic
        risk_level = ""
        if prediction <= 0.3: risk_level = "Low"
        elif prediction <= 0.6: risk_level = "Moderate"
        elif prediction <= 0.8: risk_level = "High"
        else: risk_level = "Immediate Action Needed"

        return jsonify({
            'risk_index': float(round(prediction, 4)),
            'risk_level': risk_level,
            'status': 'success'
        })

    except Exception as e:
        return jsonify({'error': str(e), 'status': 'failed'})

if __name__ == '__main__':
    # Run the server on localhost:5000
    app.run(debug=True, port=5000)