import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import ipaddress
from scipy.stats import chi2_contingency
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
                
        def cramers_v(self,df,var1, var2):
            # Create a contingency table
            contingency_table = pd.crosstab(df[var1], df[var2])
            
            # Perform Chi-Square test
            chi2, p, dof, expected = chi2_contingency(contingency_table)
            
            # Calculate Cramér's V
            n = contingency_table.sum().sum()
            cramers_v = np.sqrt(chi2 / (n * (min(contingency_table.shape) - 1)))
            
            # Visualize the heatmap of the contingency table
            plt.figure(figsize=(8, 6))
            sns.heatmap(contingency_table, annot=True, fmt="d", cmap='Blues')
            plt.title(f'Contingency Table for {var1} and {var2}')
            plt.xlabel(var2)
            plt.ylabel(var1)
            plt.show()
            
            print(f"Cramér's V correlation between {var1} and {var2}: {cramers_v}")
            
            
        def merge_fraud_with_ip(self,fraud_df, ip_df):
            # Function to convert IP address to integer
            def ip_to_int(ip):
                ip=int(ip)
                try:
                    return int(ipaddress.ip_address(ip))
                except ValueError:
                    return None  # Handle invalid IPs by returning None

            # Convert 'ip_address' in fraud_df to integer format
            fraud_df['ip_address'] = fraud_df['ip_address'].apply(ip_to_int)
            
            # Convert 'lower_bound_ip_address' and 'upper_bound_ip_address' in ip_df to integer format
            ip_df['lower_bound_ip_address'] = ip_df['lower_bound_ip_address'].apply(ip_to_int)
            ip_df['upper_bound_ip_address'] = ip_df['upper_bound_ip_address'].apply(ip_to_int)
            
            # Function to find the country for a given IP address
            def find_country(ip_address):
                # Filter the rows in ip_df where the IP falls between lower and upper bounds
                matching_row = ip_df[(ip_df['lower_bound_ip_address'] <= ip_address) & (ip_df['upper_bound_ip_address'] >= ip_address)]
                if not matching_row.empty:
                    return matching_row.iloc[0]['country']  # Return the country of the first match
                return None  # Return None if no match is found

            # Create a new column 'country' by applying the find_country function to each row in fraud_df
            fraud_df['country'] = fraud_df['ip_address'].apply(find_country)
            
            return fraud_df

        def drop_null_country(self,df):
            # Drop rows where 'country' column has null values
            df_cleaned = df.dropna(subset=['country'])
            
            return df_cleaned
        
        def feature_engineering(self,df):
            # Transaction frequency (number of transactions by user_id)
            df['transaction_count'] = df.groupby('user_id')['user_id'].transform('count')
            
            # Time-based features
            df['purchase_time'] = pd.to_datetime(df['purchase_time'])
            df['hour_of_day'] = df['purchase_time'].dt.hour
            df['day_of_week'] = df['purchase_time'].dt.weekday
            
            return df
        
        def normalize_scale_features(self,df):
            scaler = StandardScaler()
            
            df[['purchase_value', 'transaction_count']] = scaler.fit_transform(df[['purchase_value', 'transaction_count']])
            
            return df
        
        def one_hot_encode(self,df):
            columns_to_encode = ['source', 'browser','sex']
            # Perform one-hot encoding on the specified columns
            df_encoded = pd.get_dummies(df, columns=columns_to_encode, drop_first=False)
            
            # Get the newly created one-hot encoded columns (those that contain the original column names)
            encoded_columns = [col for col in df_encoded.columns if any(c in col for c in columns_to_encode)]
    
            # Convert the one-hot encoded columns to integer (0 and 1)
            df_encoded[encoded_columns] = df_encoded[encoded_columns].astype(int)
            
            return df_encoded
        
            
        def frequency_encode_country(self,df):
            # Calculate the frequency of each country
            freq_encoding = df['country'].value_counts() / len(df)
            
            # Map the country names to their frequency
            df['country' + '_encoded'] = df['country'].map(freq_encoding)
            
            # Optionally, drop the original 'country' column
            df = df.drop(columns=['country'])
            
            return df
        
