import dash
from dash import html, dcc, Input, Output
import dash_bootstrap_components as dbc

def layout_another():
    return html.Div([
        html.H2("Another Page"),
        html.P("This is another page.")
    ])