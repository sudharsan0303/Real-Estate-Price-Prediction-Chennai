import pandas as pd
from flask import Flask, request, render_template
import pickle
import os

MODEL_PATH = 'model.pkl'
COLUMNS_PATH = 'model_columns.pkl'
SCALER_PATH = 'scaler.pkl'

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file '{MODEL_PATH}' not found.")
if not os.path.exists(COLUMNS_PATH):
    raise FileNotFoundError(f"Columns file '{COLUMNS_PATH}' not found.")
if not os.path.exists(SCALER_PATH):
    raise FileNotFoundError(f"Scaler file '{SCALER_PATH}' not found.")

with open(MODEL_PATH, 'rb') as model_file:
    model = pickle.load(model_file)
with open(COLUMNS_PATH, 'rb') as columns_file:
    required_columns = pickle.load(columns_file)
with open(SCALER_PATH, 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.form

        # Create a DataFrame from form data
        input_dict = {
            'area': float(data.get('area', 0)),
            'bedrooms': int(data.get('bedrooms', 0)),
            'bathrooms': int(data.get('bathrooms', 0)),
            'stories': int(data.get('stories', 0)),
            'parking': int(data.get('parking', 0)),
            'mainroad': data.get('mainroad', 'no'),
            'guestroom': data.get('guestroom', 'no'),
            'basement': data.get('basement', 'no'),
            'hotwaterheating': data.get('hotwaterheating', 'no'),
            'airconditioning': data.get('airconditioning', 'no'),
            'prefarea': data.get('prefarea', 'no'),
            'furnishingstatus': data.get('furnishingstatus', 'unfurnished')
        }
        df = pd.DataFrame([input_dict])

        # One-hot encode to match training
        df = pd.get_dummies(df)

        # Add any missing columns
        for col in required_columns:
            if col not in df.columns:
                df[col] = 0

        # Ensure column order matches training
        df = df[required_columns]

        # Scale features
        df_scaled = scaler.transform(df)

        prediction = model.predict(df_scaled)

        return render_template('index.html', predicted_price=round(prediction[0], 2))

    except Exception as e:
        return render_template('index.html', predicted_price=f"Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)
