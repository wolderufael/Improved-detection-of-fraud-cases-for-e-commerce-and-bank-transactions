# Improved Detection of Fraud Cases for E-commerce and Bank Transactions

## Overview

This project aims to enhance the detection of fraud cases in e-commerce and bank transactions using advanced machine learning techniques. The goal is to provide a more accurate and efficient approach to identifying fraudulent activities, thereby reducing financial losses and improving security for users.

## Table of Contents

- [Project Structure](#project-structure)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Data](#data)
- [Models Used](#models-used)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Project Structure

```plaintext
Improved Detection of Fraud Cases for E-commerce and Bank Transactions/
│
├── api/ 
│   ├── serve_model.py
│ 
├── Dash/                 
│   ├── pages/
│            ├── dashboard.py
│            ├── form.py
│            ├── frequency.py
│            ├── geo.py
│            ├── home.py
│            ├── summary.py
|   ├── app_instance.py
|   ├── app.py
|   ├── callbacks.py
|   ├── layout.py
│
├── models/                  # Contains the trained models
│   ├── /fraud_data/RandomForestClassifier-22-10-2024-22-28-18-00.pkl
│   
|   
├── notebooks/             # Jupyter notebooks for data exploration and model development
│   ├── Fraud_data_preprocessing.ipynb
│   ├── fraud_prediction_for_creditcard.ipynb
│   ├── fraud_prediction.ipynb
│   ├── SHAP_LIME_explainaibility.ipynb
│
├── script/  
│   ├── Fraud_data_preprocessing.py # Python scripts for EDA and feature engineering 
│   ├── modeling.py        # Model building and evaluation script
│   ├── SHAP_LIME.py       # Model explainaibility
│
├── requirements.txt       # List of required Python packages
├── README.md              # Project's README file
└── LICENSE                # License for the project
```

## Features

- **Data Preprocessing**: Efficient handling of missing values, categorical variables, and data normalization.
- **Machine Learning Models**: Implementation of various algorithms including Logistic Regression, Decision Trees, Random Forests, and Gradient Boosting.
- **Performance Evaluation**: Detailed metrics for model performance, including accuracy, precision, recall, and F1-score.
- **API Development**: develop api using Flask.
- **Visualization**: Graphical representation of data distributions and model performance.
  
## Installation

To get started, clone the repository and install the required packages:

    ```bash
    git clone https://github.com/wolderufael/Improved-detection-of-fraud-cases-for-e-commerce-and-bank-transactions.git
    cd Improved-detection-of-fraud-cases-for-e-commerce-and-bank-transactions
    pip install -r requirements.txt
    ```
## Models Used
in this project the following models are used:

- **Logistic Regression**
- **Decision Tree**
- **Random Forest**
- **Gradient Boosting**
- **Multi-Layer Perceptron (MLP)**
- **Convolutional Neural Network (CNN)**
- **Recurrent Neural Network (RNN)**
- **Long Short-Term Memory (LSTM)**

## Results
### Dash board
![Dashboard](Data/dashboard/dashboard_1.png)
![Faud Distribution](Data/dashboard/dashboard_2.png)



## Contributing
Contributions are welcome! Please feel free to submit a pull request or open an issue if you find any bugs or have suggestions for improvements.

## License
This project is licensed under the MIT License - see the LICENSE file for details.