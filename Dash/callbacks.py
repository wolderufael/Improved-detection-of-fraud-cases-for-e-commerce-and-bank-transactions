import pandas as pd
import numpy as np
from dash.dependencies import Input, Output
import dash
from dash import html, dcc, Input, Output
import dash_bootstrap_components as dbc
import requests
import plotly.express as px


# from app_instance import app  # Import the app instance
from app import app


from pages.home import layout_home
from pages.form import layout_form
from pages.dashboard import layout_dashboard
from pages.summary import layout_summary
from pages.frequency import layout_frequency
from pages.geo import layout_geo

@app.callback(Output("tab-content", "children"), Input("tabs", "value"))
def render_content(tab):
    if tab == "home":
        return layout_home()
    elif tab == "form":
        return layout_form()
    elif tab == "dashboard":
        return layout_dashboard()
    
@app.callback(Output("dashboard-content", "children"), Input("dashboard-tabs", "value"))
def render_dashboard(tab):
    if tab == "summary":
        return layout_summary()
    elif tab == "geo":
        return layout_geo()
    elif tab == "frequency":
        return layout_frequency()

@app.callback(
    Output("result", "children"),
    Input("submit-button", "n_clicks"),
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
        response = requests.post("https://improved-detection-of-fraud-cases-for-e.onrender.com/submit", data=form_data)
        if response.status_code == 200:
            result = response.json()
            return f"Prediction: {result['meaning']} (Code: {result['prediction']})"
        else:
            return "Error: Could not get a prediction"
    except requests.exceptions.RequestException as e:
        return f"Error: {str(e)}"

# Callback to fetch and display summary statistics
@app.callback(
    [Output("total-transactions", "children"),
     Output("total-fraud-cases", "children"),
     Output("fraud-percentage", "children")],
    [Input("total-transactions", "id")]  # dummy input for initial load
)
def update_summary(_):
    # Request summary data from Flask API
    response = requests.get("https://improved-detection-of-fraud-cases-for-e.onrender.com/api/summary")
    summary = response.json()

    return (
        f"{summary['total_transactions']:,}",
        f"{summary['total_fraud_cases']:,}",
        f"{summary['fraud_percentage']:.2f}%"
    )
    
@app.callback(
    Output("fraud-trend-chart", "figure"),
    [Input("fraud-trend-chart", "id")]  # dummy input for initial load
)
def update_fraud_trend(_):
    # Fetch fraud trend data from Flask API
    response = requests.get("https://improved-detection-of-fraud-cases-for-e.onrender.com/api/fraud_trend")
    fraud_trend = pd.DataFrame(response.json())
    
    # Line chart
    fig = px.line(fraud_trend, x='signup_time', y='fraud_count', title="Fraud Cases Over Time")
    fig.update_layout(xaxis_title="Time", yaxis_title="Number of Fraud Cases")
    
    return fig

# Populate dropdown options on initial load
@app.callback(
    Output('feature-dropdown', 'options'),
    Input('feature-dropdown', 'id')  # Trigger once on load
)
def set_dropdown_options(_):
    features = ['source', 'browser', 'sex']
    feature_options = [{'label': feature, 'value': feature} for feature in features]
    return feature_options

# Update frequency distribution based on selected feature
@app.callback(
    Output("frequency-distribution", 'figure'),
    Input("feature-dropdown", "value")
)
def plot_freq_dist(selected_feature):
    if selected_feature is None:
        return {}

    # Request frequency distribution data from Flask API
    response = requests.get(f"https://improved-detection-of-fraud-cases-for-e.onrender.com/api/frequency_distribution/{selected_feature}")
    value_counts = response.json()

    # Convert to DataFrame for plotting
    value_counts_df = pd.DataFrame(value_counts)

    fig = px.bar(
        value_counts_df,
        x=selected_feature,
        y='count',
        labels={selected_feature: selected_feature.capitalize(), 'count': 'Count'},
        title=f'Frequency Distribution of {selected_feature.capitalize()}',
        color=selected_feature,
        color_discrete_sequence=px.colors.qualitative.Set2
    )
    fig.update_layout(xaxis_title=selected_feature.capitalize(), yaxis_title='Count')
    fig.update_xaxes(tickangle=45) 
    
    return fig

# Update fraud distribution map by country
@app.callback(
    Output('fraud-country-map', 'figure'),
    Input('fraud-country-map', 'id')  # Trigger on component load
)
def plot_fraud_distribution(_):
    # Request fraud distribution data from Flask API
    response = requests.get("https://improved-detection-of-fraud-cases-for-e.onrender.com/api/fraud_distribution")
    fraud_counts = pd.DataFrame(response.json())

    # Create a choropleth map
    fig = px.choropleth(
        fraud_counts,
        locations='country',
        locationmode='country names',
        color='fraud_count',
        hover_name='country',
        color_continuous_scale=px.colors.sequential.Reds,
        title='Fraud Distribution by Country',
        labels={'fraud_count': 'Number of Frauds'},
        range_color=[0, fraud_counts['fraud_count'].max()] if not fraud_counts['fraud_count'].empty else [0, 1]
    )

    # Update geos with additional customizations
    fig.update_geos(
        showcoastlines=True,
        coastlinecolor='Black',
        showcountries=True,
        countrycolor='gray',
        landcolor='lightgray',
        showlakes=True,
        lakecolor='blue',
        projection_type='natural earth'  # Set projection type
    )

    # Update layout for better visualization
    fig.update_layout(margin={"r": 0, "t": 40, "l": 0, "b": 0})

    return fig 

