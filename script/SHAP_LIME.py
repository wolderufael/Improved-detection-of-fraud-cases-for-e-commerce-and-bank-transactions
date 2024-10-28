import pickle
from sklearn.model_selection import train_test_split
import shap
import lime
from lime.lime_tabular import LimeTabularExplainer
import numpy as np

class Explainaibility:
    def load_model(self,model_path):
        # Load the pre-trained logistic re model model using pickle
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
            
        return model
    
    def split_data(self,df):
        X = df.drop(['Unnamed: 0', 'user_id', 'signup_time', 'purchase_time','device_id','ip_address','class'],axis = 1)
        y = df['class']

        # Split the dataset into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        return X_train, X_test, y_train, y_test
    
    def shap_params(self,model,X_test):
        explainer = shap.Explainer(model, X_test)
        shap_values = explainer(X_test)
        
        return explainer,shap_values
    
    def summary_plot(self,shap_values,X_test):
        shap.summary_plot(shap_values, X_test)
        
    def force_plot(self,shap_values,explainer,X_test):
        # Make sure to reshape shap_values for the single instance if needed
        shap_values_single = shap_values[0]  # extract shap values for the first observation

        # Generate the force plot for the first prediction
        shap.force_plot(
            explainer.expected_value, 
            shap_values_single.values, 
            X_test.iloc[0, :],  # Use the row as a Series or array
            matplotlib=True
        )
        
    def dependece_plot(self,shap_values,feature,X_test):
        shap.dependence_plot(feature, shap_values.values, X_test)
        
    def lime_feature_importance_plot(self,model,X_train,X_test):
        explainer = LimeTabularExplainer(
            training_data=np.array(X_train),
            feature_names=X_train.columns,
            class_names=[0, 1],  
            mode='classification'
        )

        # Select an instance for explanation
        instance_to_explain = X_test.iloc[0]  

        # Explain the prediction for this instance
        explanation = explainer.explain_instance(
            data_row=instance_to_explain,
            predict_fn=model.predict_proba  
        )

        # Plot the explanation (Feature Importance Plot)
        explanation.show_in_notebook(show_table=True)