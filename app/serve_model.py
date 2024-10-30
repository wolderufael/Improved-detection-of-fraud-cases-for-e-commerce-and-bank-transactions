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
import gdown

# Add the project directory to sys.path
current_directory = os.path.dirname(os.path.abspath(__file__))
project_directory = os.path.abspath(os.path.join(current_directory, '..'))  
sys.path.append(project_directory)

sys.path.append(os.path.abspath('../models'))

# Initialize Flask app
app = Flask(__name__)
CORS(app)

df = pd.read_csv('data/merged_data.csv')

# # Load the pre-trained randomforrest model using pickle
# with open('../models/fraud_data/RandomForestClassifier-30-10-2024-10-10-39-00.pkl', 'rb') as f:
#     rf_model = pickle.load(f)
    
# Google Drive file ID
file_id = '1tTDVR3k11cbygLeZGNf_fBMy6dCWu_zu'
# Output file path
destination = '../model.pkl'

# Download the file using gdown
gdown.download(f'https://drive.google.com/uc?id={file_id}', destination, quiet=False)

# Load the model
with open(destination, 'rb') as f:
    rf_model = pickle.load(f)

from script.Fraud_data_preprocessing import FraudDataPreprocessing
preprocessecor=FraudDataPreprocessing()

def preprocess_input(data):
    # Convert JSON data to DataFrame
    df = pd.DataFrame([data])
    #process the data
    feat_engineered=preprocessecor.feature_engineering(df)
    scaler = joblib.load('params/scaler.joblib') # use the scaler saved for training
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
    freq_encoding = joblib.load('params/country_freq_encoding.joblib')
    # Map the frequency encoding to the 'country' column in the new data
    encoded['country_encoded'] = encoded['country'].map(freq_encoding).fillna(0)
    # Drop the original 'country' column if needed
    processed = encoded.drop(columns=['country'])
    
    return processed

@app.route('/')
def form():
    return render_template('form.html')

# Endpoint for summary statistics
@app.route('/api/summary', methods=['GET'])
def get_summary():
    total_transactions = df.shape[0]
    total_fraud_cases = df[df['class'] == 1].shape[0]
    fraud_percentage = (total_fraud_cases / total_transactions) * 100

    summary = {
        "total_transactions": total_transactions,
        "total_fraud_cases": total_fraud_cases,
        "fraud_percentage": fraud_percentage
    }
    return jsonify(summary)

# Endpoint for fraud cases over time
@app.route('/api/fraud_trend', methods=['GET'])
def get_fraud_trend():
    # Assuming `signup_time` is in datetime format
    df['signup_time'] = pd.to_datetime(df['signup_time'])
    fraud_trend = df[df['class'] == 1].resample('M', on='signup_time').size().reset_index(name='fraud_count')
    
    trend_data = fraud_trend.to_dict(orient='records')
    return jsonify(trend_data)

# Endpoint for frequency distribution of selected feature
@app.route('/api/frequency_distribution/<string:feature>', methods=['GET'])
def get_frequency_distribution(feature):
    value_counts = df[feature].value_counts().reset_index()
    value_counts.columns = [feature, 'count']
    
    freq_data = value_counts.to_dict(orient='records')
    return jsonify(freq_data)

# Endpoint for fraud distribution by country
@app.route('/api/fraud_distribution', methods=['GET'])
def get_fraud_distribution():
    fraud_counts = df[df['class'] == 1]['country'].value_counts().reset_index()
    fraud_counts.columns = ['country', 'fraud_count']

    # Return as JSON
    fraud_data = fraud_counts.to_dict(orient='records')
    return jsonify(fraud_data)

@app.route('/submit', methods=['POST'])
def submit():
    data = request.form
    processed_data = preprocess_input(data)

    # Make prediction with the loaded model
    prediction = rf_model.predict(processed_data)
    result_data={'prediction':int(prediction[0]),
                 'meaning': 'a Fraud' if int(prediction[0])==1 else 'Not a Fraud' }

    # return render_template('result.html', result=result_data)
    return jsonify(result_data) 

# Define a health check route
@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'API is running', 'time': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")})

# Run Flask app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
