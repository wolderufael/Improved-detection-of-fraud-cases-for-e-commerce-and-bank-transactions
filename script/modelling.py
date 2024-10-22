# Import necessary libraries
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix, roc_curve, auc
import numpy as np
import seaborn as sns
import os



class Modelling:
        def split_data(self,df):
            X = df.drop(['Unnamed: 0', 'user_id', 'signup_time', 'purchase_time','device_id','ip_address','class'],axis = 1)
            y = df['class']

            # Split the dataset into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
            
            return X_train, X_test, y_train, y_test
        
        def train_with_ml_model(self,X_train, X_test, y_train, y_test,model_name):
            # Enable autologging
            mlflow.sklearn.autolog()
            
            with mlflow.start_run(run_name=f"{model_name}_model"):
                # Model training
                if model_name=="LogisticRegression":
                    model = LogisticRegression(max_iter=100)
                elif model_name=="RandomForestClassifier":
                    model = RandomForestClassifier(n_estimators=100, random_state=42)
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
                mlflow.sklearn.log_model(model, "logistic_regression_model")
                
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
                
                # Save the ROC curve plot
                roc_plot_path = f"{model_name}_roc_curve.png"
                plt.savefig(roc_plot_path)
                mlflow.log_artifact(roc_plot_path)
      
                # Print the run ID for reference
                run_id = mlflow.active_run().info.run_id
                print(f"Run ID: {run_id}")