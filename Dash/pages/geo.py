import dash
from dash import html, dcc, Input, Output
import dash_bootstrap_components as dbc

def layout_geo():
    return dbc.Container([
        dbc.Row([
            dbc.Col(dcc.Graph(id='fraud-country-map'))
        ])
    ],fluid=True, className='container')
    
    # return html.Div([
    #     html.H2("Geographical"),
    #     html.P("Addey Innovations aims to create accurate and adaptable models that can identify fraudulent activities within both e-commerce transactions and bank credit transactions. By leveraging advanced machine learning techniques and in-depth data analysis, we aim to significantly enhance fraud detection capabilities")
    # ])