import os
import xgboost as xgb
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
    # ... (the rest of your prediction code from before)
    pass

if __name__ == '__main__':
    app.run(debug=True)