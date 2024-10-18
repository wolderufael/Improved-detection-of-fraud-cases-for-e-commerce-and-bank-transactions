import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

class FraudDataPreprocessing:
        def __init__(self):
            """
            Constructor to initialize file paths for data loading
            """
            self.fraud_data_path = 'Data/Fraud_Data.csv'
            self.ip_country_path = 'Data/IpAddress_to_Country.csv'
            self.creditcard_data_path='Data/creditcard.csv/creditcard.csv'

        def load_data(self):
            
            fraud_data = pd.read_csv(self.fraud_data_path)
            ip_country_data = pd.read_csv(self.ip_country_path)
            creditcard_data= pd.read_csv(self.creditcard_data_path)
            
            return fraud_data,ip_country_data,creditcard_data
        
        def data_overview(self,df):
            num_rows = df.shape[0]
            num_columns = df.shape[1]
            data_types = df.dtypes

            print(f"Number of rows:{num_rows}")
            print(f"Number of columns:{num_columns}")
            print(f"Data types of each column:\n{data_types}")
        def check_missing(self,df):
            missing=df.isnull().sum()
            
            return missing 
        
        def data_cleaning(self,df):
            # Remove duplicates
            df.drop_duplicates(inplace=True)
            
            print("Duplicates are removed from fraud data!")
            
            # Correct data  (convert timestamps)
            df['signup_time'] = pd.to_datetime(df['signup_time'])
            df['purchase_time'] = pd.to_datetime(df['purchase_time'])
            
            print("Timestamps are converted!")
        
        def summarize_dataset(self,df,numerical_columns):
            # Select numerical columns only
            # numerical_columns=['purchase_value']
            
            # Initialize a list to hold summary statistics for each column
            summary_list = []
        
            for col in numerical_columns:
                summary_stats = {
                    'Mean': df[col].mean(),
                    'Median': df[col].median(),
                    'Mode': df[col].mode().iloc[0],  # Taking the first mode in case of multiple modes
                    'Standard Deviation': df[col].std(),
                    'Variance': df[col].var(),
                    'Range': df[col].max() - df[col].min(),
                    'IQR': df[col].quantile(0.75) - df[col].quantile(0.25),
                    'Skewness': df[col].skew(),
                    'Kurtosis': df[col].kurtosis()
                }
                
                # Append the summary statistics for the current column to the list
                summary_list.append(summary_stats)
            
            # Convert summary stats list to DataFrame with appropriate index
            summary_df = pd.DataFrame(summary_list, index=numerical_columns)
            
            return summary_df
        
        def plot_univariate(self,df):
            # Select categorical columns only
            categorical_columns=['source','browser','sex','class']
            
            # Create bar plots for each categorical feature
            for col in categorical_columns:
                plt.figure(figsize=(10, 5))
                
                # Plot a bar chart for the frequency of each category
                sns.countplot(x=df[col], order=df[col].value_counts().index, hue=df[col], palette="Set2", legend=False)

                # sns.countplot(x=df[col], order=df[col].value_counts().index, palette="Set2")
                
                plt.title(f'Distribution of {col}')
                plt.xticks(rotation=45, ha='right')
                plt.ylabel('Count')
                plt.tight_layout()
                plt.show()
