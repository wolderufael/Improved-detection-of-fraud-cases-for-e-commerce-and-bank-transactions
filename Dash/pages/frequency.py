import dash
from dash import html, dcc, Input, Output
import dash_bootstrap_components as dbc

def layout_frequency():
    return dbc.Container([
        dbc.Row([
            dbc.Col([
                dbc.Label("Select Feature"),
                dcc.Dropdown(
                id='feature-dropdown',
                placeholder='Select a feature'
            )], width=4)
    ], className='mb-4'),
        dbc.Row([
            dbc.Col(dcc.Graph(id='frequency-distribution'), width=12)
        ], className="mb-5")
    ],fluid=True, className='container')
    