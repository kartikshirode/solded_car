from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
import os

app = Flask(__name__)

# Load the model from Hugging Face or local
def load_model():
    model_path = 'car_model.joblib'
    
    # If model doesn't exist locally, download from Hugging Face
    if not os.path.exists(model_path):
        try:
            from huggingface_hub import hf_hub_download
            print("Downloading model from Hugging Face...")
            model_path = hf_hub_download(
                repo_id="kartikshirode/car-price-model",
                filename="car_model.joblib",
                cache_dir=".cache"
            )
            print("✅ Model downloaded successfully!")
        except Exception as e:
            print(f"⚠️ Could not download from Hugging Face: {e}")
            raise
    
    return joblib.load(model_path)

# Load the model
try:
    model_data = load_model()
    model = model_data['model']
    mae = model_data['mae']
    model_name = model_data['name']
except Exception as e:
    print(f"Error loading model: {e}")
    raise

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from form
        data = request.json
        
        # Calculate months from purchase date
        purchase_date = pd.to_datetime(data['purchaseDate'])
        current_date = datetime.now()
        months_since = (current_date.year - purchase_date.year) * 12 + (current_date.month - purchase_date.month)
        
        # Create DataFrame with the input
        new_car = pd.DataFrame({
            'vehicleType': [data['vehicleType']],
            'gearbox': [data['gearbox']],
            'fuelType': [data['fuelType']],
            'brand': [data['brand']],
            'notRepairedDamage': [data['notRepairedDamage']],
            'powerPS': [int(data['powerPS'])],
            'kilometer': [int(data['kilometerUsed'])],
            'TimeforRegistration': [int(months_since)]
        })
        
        # Make prediction (model was trained on log-transformed data)
        y_pred_log = model.predict(new_car)
        y_pred_eur = np.expm1(y_pred_log)[0]  # Reverse log transformation
        
        # Convert EUR to INR (1 EUR ≈ 90 INR)
        eur_to_inr = 90
        y_pred_inr = y_pred_eur * eur_to_inr
        
        return jsonify({
            'success': True,
            'prediction_eur': round(y_pred_eur, 2),
            'prediction_inr': round(y_pred_inr, 2),
            'model': model_name,
            'mae_eur': round(mae, 2),
            'mae_inr': round(mae * eur_to_inr, 2)
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

if __name__ == '__main__':
    app.run(debug=True)
