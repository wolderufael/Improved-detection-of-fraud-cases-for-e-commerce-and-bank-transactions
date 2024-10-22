# Import necessary libraries
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score,confusion_matrix, roc_curve, auc
import numpy as np
import seaborn as sns
import os
import pickle
from datetime import datetime



class Modelling:
        def split_data(self,df):
            X = df.drop(['Unnamed: 0', 'user_id', 'signup_time', 'purchase_time','device_id','ip_address','class'],axis = 1)
            y = df['class']

            # Split the dataset into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
            
            return X_train, X_test, y_train, y_test
        def split_creditcard_data(self,df):
            X = df.drop(['Class'],axis = 1)
            y = df['Class']

            # Split the dataset into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
            
            return X_train, X_test, y_train, y_test
        
        def train_with_ml_model(self,X_train, X_test, y_train, y_test,model_name):
            # Enable autologging
            mlflow.sklearn.autolog()
            
            with mlflow.start_run(run_name=f"fraud_data_{model_name}_model"):
                # Model training
                if model_name=="LogisticRegression":
                    model = LogisticRegression(max_iter=100)
                elif model_name=="RandomForestClassifier":
                    model = RandomForestClassifier(n_estimators=100, random_state=42)
                elif model_name=="DecisionTreeClassifier":
                    model = DecisionTreeClassifier(max_depth=3, random_state=42)
                elif model_name=="GradientBoostingClassifier":
                    model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
                elif model_name=="MLPClassifier":
                    model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, alpha=0.001, solver='adam', random_state=42)
                    
                model.fit(X_train, y_train)

                # Predict on the test set
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test)
                
                # Evaluate the model
                accuracy = accuracy_score(y_test, y_pred)
                print(f"Accuracy: {accuracy}")

                # Log parameters and metrics manually 
                mlflow.log_metric("accuracy", accuracy)

                # Log the model to MLflow
                mlflow.sklearn.log_model(model, f"{model_name}_model")
                
                # Log parameters
                mlflow.log_param("max_iter", 100)
                mlflow.log_param("solver", "lbfgs")
                
                # Confusion Matrix
                cm = confusion_matrix(y_test, y_pred)

                # Plot confusion matrix
                plt.figure(figsize=(6,6))
                class_names=['1','0']
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                            xticklabels=class_names, yticklabels=class_names)
                plt.xlabel('Predicted Label')
                plt.ylabel('True Label')
                plt.title('Confusion Matrix')
                plt.show()
                
                # Save the confusion matrix plot
                cm_plot_path = f"Data/{model_name}_confusion_matrix.png"
                plt.savefig(cm_plot_path)
                mlflow.log_artifact(cm_plot_path)

                # ROC Curve
                fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba[:, 0], pos_label=0)
                roc_auc = auc(fpr, tpr)
                
                # Plot ROC curve
                plt.figure()
                plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
                plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title('Receiver Operating Characteristic')
                plt.legend(loc="lower right")
                plt.show()
                
                # Save the ROC curve plot
                roc_plot_path = f"Data/{model_name}_roc_curve.png"
                plt.savefig(roc_plot_path)
                mlflow.log_artifact(roc_plot_path)
      
                # Print the run ID for reference
                run_id = mlflow.active_run().info.run_id
                print(f"Run ID: {run_id}")
                
                #Save the model in .pkl format
                # Create the folder if it doesn't exist
                folder_path='models/fraud_data/'
                if not os.path.exists(folder_path):
                    os.makedirs(folder_path)
                
                # Generate timestamp in format dd-mm-yyyy-HH-MM-SS-00
                timestamp = datetime.now().strftime("%d-%m-%Y-%H-%M-%S-00")
                
                # Create a filename with the timestamp
                filename = f'{folder_path}{model_name}-{timestamp}.pkl'
                
                # Save the model using pickle
                with open(filename, 'wb') as file:
                    pickle.dump(model, file)
                
                print(f"Model saved as {filename}")
                
        def predict_fraud_for_creditcard(self,X_train, X_test, y_train, y_test,model_name):
            # Enable autologging
            mlflow.sklearn.autolog()
            
            with mlflow.start_run(run_name=f"creditcard_{model_name}_model"):
                # Model training
                if model_name=="LogisticRegression":
                    model = LogisticRegression(max_iter=100)
                elif model_name=="RandomForestClassifier":
                    model = RandomForestClassifier(n_estimators=100, random_state=42)
                elif model_name=="DecisionTreeClassifier":
                    model = DecisionTreeClassifier(max_depth=3, random_state=42)
                elif model_name=="GradientBoostingClassifier":
                    model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
                elif model_name=="MLPClassifier":
                    model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, alpha=0.001, solver='adam', random_state=42)
                    
                model.fit(X_train, y_train)

                # Predict on the test set
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test)
                
                # Evaluate the model
                accuracy = accuracy_score(y_test, y_pred)
                print(f"Accuracy: {accuracy}")

                # Log parameters and metrics manually 
                mlflow.log_metric("accuracy", accuracy)

                # Log the model to MLflow
                mlflow.sklearn.log_model(model, f"{model_name}_model")
                
                # Log parameters
                mlflow.log_param("max_iter", 100)
                mlflow.log_param("solver", "lbfgs")
                
                # Confusion Matrix
                cm = confusion_matrix(y_test, y_pred)

                # Plot confusion matrix
                plt.figure(figsize=(6,6))
                class_names=['1','0']
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                            xticklabels=class_names, yticklabels=class_names)
                plt.xlabel('Predicted Label')
                plt.ylabel('True Label')
                plt.title('Confusion Matrix')
                plt.show()
                
                # Save the confusion matrix plot
                cm_plot_path = f"Data/creditcard/{model_name}_confusion_matrix.png"
                plt.savefig(cm_plot_path)
                mlflow.log_artifact(cm_plot_path)

                # ROC Curve
                fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba[:, 0], pos_label=0)
                roc_auc = auc(fpr, tpr)
                
                # Plot ROC curve
                plt.figure()
                plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
                plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title('Receiver Operating Characteristic')
                plt.legend(loc="lower right")
                plt.show()
                
                # Save the ROC curve plot
                roc_plot_path = f"Data/creditcard/{model_name}_roc_curve.png"
                plt.savefig(roc_plot_path)
                mlflow.log_artifact(roc_plot_path)
      
                # Print the run ID for reference
                run_id = mlflow.active_run().info.run_id
                print(f"Run ID: {run_id}")
                #Save the model in .pkl format
                # Create the folder if it doesn't exist
                folder_path='models/creditcard/'
                if not os.path.exists(folder_path):
                    os.makedirs(folder_path)
                
                # Generate timestamp in format dd-mm-yyyy-HH-MM-SS-00
                timestamp = datetime.now().strftime("%d-%m-%Y-%H-%M-%S-00")
                
                # Create a filename with the timestamp
                filename = f'{folder_path}{model_name}-{timestamp}.pkl'
                
                # Save the model using pickle
                with open(filename, 'wb') as file:
                    pickle.dump(model, file)
                
                print(f"Model saved as {filename}")