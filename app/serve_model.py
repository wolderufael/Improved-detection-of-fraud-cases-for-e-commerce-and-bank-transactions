from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import pickle
import datetime
import numpy as np
import sys
import os
from sklearn.model_selection import train_test_split

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

def preprocess_input_rf(data):
    # Convert JSON data to DataFrame
    df = pd.DataFrame([data])
    #process the data
    feat_engineered=preprocessecor.feature_engineering(df)
    normalized=preprocessecor.normalize_scale_features(feat_engineered)
    encoded=preprocessecor.one_hot_encode(normalized)
    processed=preprocessecor.frequency_encode_country(encoded)    
    X_data = processed.drop(['user_id', 'signup_time', 'purchase_time','device_id'],axis = 1)
     
    return X_data

    
# Define logistic regression prediction endpoint for CSV file input
@app.route('/detect_fraud', methods=['POST'])
def predict():
    try:
        # data = request.get_json()
        # processed_data = preprocess_input_rf(data)
        
        # # Make prediction with the loaded model
        # prediction = rf_model.predict(processed_data)

        # return jsonify({'prediction': int(prediction[0])})
        # return jsonify(response)
        # Get JSON data from the request
        input_data = request.get_json()
        
        # Return the received JSON data as the response
        return jsonify(input_data)

    except Exception as e:
        return jsonify({'error': str(e)}), 500


# Define a health check route
@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'API is running', 'time': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")})

# Run Flask app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
