import pandas as pd
import numpy as np
from dash.dependencies import Input, Output
import dash
from dash import html, dcc, Input, Output
import dash_bootstrap_components as dbc
import requests
import plotly.express as px
from scipy.stats import chi2_contingency

# from app_instance import app  # Import the app instance
from app import app

# Import layouts
from form import form_layout
# from another_page import another_page_layout

# Load and preprocess data
df = pd.read_csv('../Data/merged_data.csv')

from pages.home import layout_home
from pages.form import layout_form
from pages.another import layout_another

@app.callback(Output("tab-content", "children"), Input("tabs", "value"))
def render_content(tab):
    if tab == "home":
        return layout_home()
    elif tab == "form":
        return layout_form()
    elif tab == "another":
        return layout_another()

# dash_app.py continued

@app.callback(
    Output("result", "children"),
    Input("submit", "n_clicks"),
    [Input("user_id", "value"),
     Input("signup_time", "value"),
     Input("source", "value"),
     Input("purchase_value", "value"),
     Input("age", "value"),
     Input("device_id", "value"),
     Input("purchase_time", "value"),
     Input("browser", "value"),
     Input("sex", "value"),
     Input("country", "value")]
)
def submit_form(n_clicks, user_id, signup_time, source, purchase_value, age, device_id, purchase_time, browser, sex, country):
    if n_clicks is None:
        return ""
    
    # Prepare the data to send
    form_data = {
        "user_id": user_id,
        "signup_time": signup_time,
        "source": source,
        "purchase_value": purchase_value,
        "age": age,
        "device_id": device_id,
        "purchase_time": purchase_time,
        "browser": browser,
        "sex": sex,
        "country": country
    }

    try:
        # Send a POST request to Flask
        response = requests.post("http://127.0.0.1:5000/submit", data=form_data)
        if response.status_code == 200:
            result = response.json()
            return f"Prediction: {result['meaning']} (Code: {result['prediction']})"
        else:
            return "Error: Could not get a prediction"
    except requests.exceptions.RequestException as e:
        return f"Error: {str(e)}"

# Callback to fetch and display summary statistics
# @app.callback(
#     [Output("total-transactions", "children"),
#      Output("total-fraud-cases", "children"),
#      Output("fraud-percentage", "children")],
#     [Input("total-transactions", "id")]  # dummy input for initial load
# )
# def update_summary(_):
#     # Request summary data from Flask API
#     response = requests.get("http://127.0.0.1:5000/api/summary")
#     summary = response.json()

#     return (
#         f"{summary['total_transactions']:,}",
#         f"{summary['total_fraud_cases']:,}",
#         f"{summary['fraud_percentage']:.2f}%"
#     )

# # Populate dropdown options on initial load
# @app.callback(
#     [
#         Output('feature-dropdown', 'options'),
#         Output('feature-dropdown', 'value'),
#         # Output('month-dropdown', 'options'),
#         # Output('month-dropdown', 'value'),
#     ],
#     [Input('feature-dropdown', 'options')]  # Trigger once on load
# )
# def set_dropdown_options(_):
#     # years = sorted(df['Year'].unique())
#     features=['source','browser','sex']
#     # months = sorted(df['Month'].unique(), key=lambda x: pd.to_datetime(x, format='%B').month)
#     feature_options = [{'label': feature, 'value': feature} for feature in features]
#     # month_options = [{'label': month, 'value': month} for month in months]
#     return feature_options, features[0]

# #Update frequency distribution
# @app.callback(
#     [
#         Output("frequency-distribution",'figure')
#     ],[
#         Input("feature-dropdown","value")
#     ]
# )
# def plot_freq_dist(selected_feature):
#     # Calculate frequency counts for the selected feature
#     value_counts = df[selected_feature].value_counts().reset_index()
#     value_counts.columns = [selected_feature, 'count']
    
#     fig = px.bar(
#         value_counts,
#         x=selected_feature,
#         y='count',
#         labels={selected_feature: selected_feature.capitalize(), 'count': 'Count'},
#         title=f'Frequency Distribution of {selected_feature.capitalize()}',
#         color=selected_feature,  
#         color_discrete_sequence=px.colors.qualitative.Set2
#     )
#     fig.update_layout(xaxis_title=selected_feature.capitalize(), yaxis_title='Count')
#     fig.update_xaxes(tickangle=45) 
    
#     return [fig]

# # Update correlation
# @app.callback(
#     [
#         Output("contingency-table",'figure'),
#         Output('cramers-v-output', 'children'),
#     ],[
#         Input("feature-dropdown","value")
#     ]
# )
# def cramers_v(selected_feature):
#     # Create a contingency table
#     # contingency_table = pd.crosstab(df[selected_feature], df['class'])
#     # Create a contingency table
#     df['class'] = df['class'].astype('category')
#     contingency_table = pd.crosstab(df[selected_feature].astype('category'), df['class'])

#     # Perform Chi-Square test
#     chi2, p, dof, expected = chi2_contingency(contingency_table)

#     # Calculate Cramér's V
#     n = contingency_table.sum().sum()
#     cramers_v = np.sqrt(chi2 / (n * (min(contingency_table.shape) - 1)))

#     # Create a heatmap of the contingency table using Plotly
#     fig = px.imshow(contingency_table,
#                      text_auto=True,
#                      color_continuous_scale='Blues',
#                      labels={'x': 'class', 'y': selected_feature, 'color': 'Count'},
#                      title=f'Contingency Table for {selected_feature} and class')
    
#     # Update output text
#     output_text = f"Cramér's V correlation between {selected_feature} and class: {cramers_v:.4f}"

#     return fig, output_text

# @app.callback(
#     Output('fraud-country-map', 'figure'),
#     Input('fraud-country-map', 'id')  # Trigger on component load
# )
# def plot_fraud_distribution(_):
#     # Count the number of fraud cases (class=1) by country
#     fraud_counts = df[df['class'] == 1]['country'].value_counts().reset_index()
#     fraud_counts.columns = ['country', 'fraud_count']

#     # Check the fraud_counts DataFrame
#     # print(fraud_counts)

#     # Define color scale ranges and labels
#     bins = [0, 1, 2, 3, 4, 5]  # 6 edges for 5 ranges
#     labels = ['0', '1-2', '3-4', '5-6', '7+']  # 5 labels

#     # Create a categorical column for fraud categories
#     fraud_counts['fraud_category'] = pd.cut(fraud_counts['fraud_count'], bins=bins, labels=labels, right=False)

#     # Create a choropleth map
#     fig = px.choropleth(
#         fraud_counts,
#         locations='country',
#         locationmode='country names',
#         color='fraud_count',
#         hover_name='country',
#         color_continuous_scale=px.colors.sequential.Reds,
#         title='Fraud Distribution by Country',
#         labels={'fraud_count': 'Number of Frauds'},
#         range_color=[0, fraud_counts['fraud_count'].max()] if not fraud_counts['fraud_count'].empty else [0, 1]
#     )


    # # Update geos with additional customizations
    # fig.update_geos(
    #     showcoastlines=True,
    #     coastlinecolor='Black',
    #     showcountries=True,
    #     countrycolor='gray',
    #     landcolor='lightgray',
    #     showlakes=True,
    #     lakecolor='blue',
    #     projection_type='natural earth'  # Set projection type
    # )

    # # Update layout for better visualization
    # fig.update_layout(margin={"r": 0, "t": 40, "l": 0, "b": 0})


    # return fig