# app.py

from flask import Flask, request, render_template
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import pickle
from pathlib import Path

# Initialize the Flask application
app = Flask(__name__)

# --- Model and Preprocessor Loading ---
base_path = Path(__file__).parent
artifacts_path = base_path / "saved_artifacts"

model = load_model(artifacts_path / 'model.h5')
label_encoder_gender = pickle.load(open(artifacts_path / 'label_encoder_gender.pkl', 'rb'))
onehot_encoder_geo = pickle.load(open(artifacts_path / 'onehot_encoder_geo.pkl', 'rb'))
scaler = pickle.load(open(artifacts_path / 'scaler.pkl', 'rb'))

# --- Helper Function for Preprocessing ---
def preprocess_data(df):
    """Preprocesses the input DataFrame for prediction."""
    # Handle Gender
    df['Gender'] = label_encoder_gender.transform(df['Gender'])
    
    # Handle Geography
    geo_encoded = onehot_encoder_geo.transform(df[['Geography']]).toarray()
    geo_df_cols = onehot_encoder_geo.get_feature_names_out(['Geography'])
    geo_encoded_df = pd.DataFrame(geo_encoded, columns=geo_df_cols, index=df.index)
    
    # Combine dataframes
    df = pd.concat([df.drop('Geography', axis=1), geo_encoded_df], axis=1)
    
    # Reorder columns to match scaler's expectation
    feature_names = scaler.feature_names_in_
    df = df[feature_names]
    
    # Scale the data
    scaled_df = scaler.transform(df)
    return scaled_df

# --- Flask Routes ---
@app.route('/')
def home():
    """Renders the main page with the input form."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Receives form data, preprocesses it, makes a prediction, and renders the result."""
    # Collect data from form
    form_data = {
        'CreditScore': [int(request.form['CreditScore'])],
        'Geography': [request.form['Geography']],
        'Gender': [request.form['Gender']],
        'Age': [int(request.form['Age'])],
        'Tenure': [int(request.form['Tenure'])],
        'Balance': [float(request.form['Balance'])],
        'NumOfProducts': [int(request.form['NumOfProducts'])],
        'HasCrCard': [int(request.form['HasCrCard'])],
        'IsActiveMember': [int(request.form['IsActiveMember'])],
        'EstimatedSalary': [float(request.form['EstimatedSalary'])]
    }
    input_df = pd.DataFrame.from_dict(form_data)

    # Preprocess the data and predict
    scaled_input = preprocess_data(input_df.copy())
    prediction_proba = model.predict(scaled_input)[0][0]
    
    # Create prediction text
    if prediction_proba > 0.5:
        result_text = f"Result: Likely to Churn (Probability: {prediction_proba:.2%})"
    else:
        result_text = f"Result: Likely to Stay (Probability: {1 - prediction_proba:.2%})"

    # Render the page again with the prediction result
    return render_template('index.html', prediction_text=result_text)

if __name__ == "__main__":
    app.run(debug=True)