from flask import Flask, request, jsonify,render_template
from flask_cors import CORS
import pandas as pd
import pickle
import datetime
import numpy as np
import sys
import os
from sklearn.preprocessing import OneHotEncoder
import joblib

# Add the project directory to sys.path
current_directory = os.path.dirname(os.path.abspath(__file__))
project_directory = os.path.abspath(os.path.join(current_directory, '..'))  
sys.path.append(project_directory)

sys.path.append(os.path.abspath('../models'))

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Load the pre-trained randomforrest model using pickle
with open('../models/fraud_data/RandomForestClassifier-28-10-2024-09-00-19-00.pkl', 'rb') as f:
    rf_model = pickle.load(f)
    

from script.Fraud_data_preprocessing import FraudDataPreprocessing
preprocessecor=FraudDataPreprocessing()

def preprocess_input(data):
    # Convert JSON data to DataFrame
    df = pd.DataFrame([data])
    #process the data
    feat_engineered=preprocessecor.feature_engineering(df)
    scaler = joblib.load('../models/scaler.joblib') # use the scaler saved for training
    feat_engineered[['purchase_value', 'transaction_count']] = scaler.transform(feat_engineered[['purchase_value', 'transaction_count']])
    # Select columns of interest
    selected_features = feat_engineered[['purchase_value', 'age', 'transaction_count', 'hour_of_day', 'day_of_week',
                                    'source', 'browser', 'sex', 'country']]    # Manually specify all possible categories for one-hot encoding
    all_possible_values = {
        'source': ['Ads', 'Direct', 'SEO'],
        'browser': ['Chrome', 'FireFox', 'IE', 'Opera', 'Safari'],
        'sex': ['F', 'M']
    }
    
    
    # One-hot encode `source`, `browser`, and `sex` using predefined categories
    one_hot_encoder = OneHotEncoder(categories=[all_possible_values['source'],
                                                all_possible_values['browser'],
                                                all_possible_values['sex']],
                                    drop=None, sparse_output=False, handle_unknown='ignore')
    one_hot_encoded = one_hot_encoder.fit_transform(selected_features[['source', 'browser', 'sex']])

    # Get feature names from the encoder and create a DataFrame
    one_hot_columns = one_hot_encoder.get_feature_names_out(['source', 'browser', 'sex'])
    one_hot_df = pd.DataFrame(one_hot_encoded, columns=one_hot_columns, index=feat_engineered.index)
    
    # Concatenate one-hot encoded columns with the selected features
    encoded = pd.concat([selected_features.drop(columns=['source', 'browser', 'sex']), one_hot_df], axis=1)
    
    # Load the saved frequency encoding mapping
    freq_encoding = joblib.load('../models/country_freq_encoding.joblib')
    # Map the frequency encoding to the 'country' column in the new data
    encoded['country_encoded'] = encoded['country'].map(freq_encoding).fillna(0)
    # Drop the original 'country' column if needed
    processed = encoded.drop(columns=['country'])
    
    return processed

@app.route('/')
def form():
    return render_template('form.html')

@app.route('/submit', methods=['POST'])
def submit():
    form_data = {
        "age": int(request.form['age']),
        "browser": request.form['browser'],
        "country": request.form['country'],
        "device_id": request.form['device_id'],
        "purchase_time": request.form['purchase_time'],
        "purchase_value": float(request.form['purchase_value']),
        "sex": request.form['sex'],
        "signup_time": request.form['signup_time'],
        "source": request.form['source'],
        "user_id": request.form['user_id']
    }

    processed_data = preprocess_input(form_data)

    # Make prediction with the loaded model
    prediction = rf_model.predict(processed_data)
    result_data={'prediction':int(prediction[0]),
                 'meaning': 'a Fraud' if int(prediction[0])==1 else 'Not a Fraud' }

    return render_template('result.html', result=result_data)

 
# # Define logistic regression prediction endpoint for CSV file input
# @app.route('/detect_fraud', methods=['POST'])
# def predict():
#     try:
#         data = request.get_json()
#         processed_data = preprocess_input_rf(data)
        
#         # Make prediction with the loaded model
#         prediction = rf_model.predict(processed_data)

#         return jsonify({'prediction': int(prediction[0])})

#     except Exception as e:
#         return jsonify({'error': str(e)}), 500


# Define a health check route
@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'API is running', 'time': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")})

# Run Flask app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
