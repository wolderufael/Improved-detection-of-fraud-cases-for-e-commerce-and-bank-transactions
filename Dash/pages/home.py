import dash
from dash import html, dcc, Input, Output
import dash_bootstrap_components as dbc

def layout_home():
    return html.Div([
        html.H2("Home Page"),
        html.P("Welcome to the home page!")
    ])