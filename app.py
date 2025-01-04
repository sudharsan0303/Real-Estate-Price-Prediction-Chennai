import pandas as pd
import numpy as np
from flask import Flask, request, render_template
import pickle

# Load the model
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

app = Flask(__name__)

# Columns used during training
required_columns = [
    'area', 'bedrooms', 'bathrooms', 'stories', 'mainroad_yes', 'mainroad_no', 
    'guestroom_yes', 'guestroom_no', 'basement_yes', 'basement_no',
    'hotwaterheating_yes', 'hotwaterheating_no', 'airconditioning_yes', 
    'airconditioning_no', 'parking', 'prefarea_yes', 'prefarea_no', 
    'furnishingstatus_furnished', 'furnishingstatus_semi-furnished', 'furnishingstatus_unfurnished'
]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.form
    
    # Create DataFrame with input values
    features = pd.DataFrame({
        'area': [float(data['area'])],
        'bedrooms': [int(data['bedrooms'])],
        'bathrooms': [int(data['bathrooms'])],
        'stories': [int(data['stories'])],
        'mainroad_yes': [1 if data['mainroad'] == 'yes' else 0],
        'mainroad_no': [1 if data['mainroad'] == 'no' else 0],
        'guestroom_yes': [1 if data['guestroom'] == 'yes' else 0],
        'guestroom_no': [1 if data['guestroom'] == 'no' else 0],
        'basement_yes': [1 if data['basement'] == 'yes' else 0],
        'basement_no': [1 if data['basement'] == 'no' else 0],
        'hotwaterheating_yes': [1 if data['hotwaterheating'] == 'yes' else 0],
        'hotwaterheating_no': [1 if data['hotwaterheating'] == 'no' else 0],
        'airconditioning_yes': [1 if data['airconditioning'] == 'yes' else 0],
        'airconditioning_no': [1 if data['airconditioning'] == 'no' else 0],
        'parking': [int(data['parking'])],
        'prefarea_yes': [1 if data['prefarea'] == 'yes' else 0],
        'prefarea_no': [1 if data['prefarea'] == 'no' else 0],
        'furnishingstatus_furnished': [1 if data['furnishingstatus'] == 'furnished' else 0],
        'furnishingstatus_semi-furnished': [1 if data['furnishingstatus'] == 'semi-furnished' else 0],
        'furnishingstatus_unfurnished': [1 if data['furnishingstatus'] == 'unfurnished' else 0]
    })

    # Ensure the input has all the columns required (even if they are 0)
    for col in required_columns:
        if col not in features.columns:
            features[col] = 0

    # Reorder the columns to match the training set
    features = features[required_columns]

    # Make prediction
    prediction = model.predict(features)

    # Return the prediction result
    return render_template('index.html', predicted_price=prediction[0])

if __name__ == '__main__':
    app.run(debug=True)
