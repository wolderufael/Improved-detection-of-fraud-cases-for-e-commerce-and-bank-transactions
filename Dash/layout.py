from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc
from dash import Dash

def create_layout(app):
    return dbc.Container([
    dbc.Row([
        dcc.Tabs(id="tabs", value="home",className="tab", children=[
            dcc.Tab(label="Home", value="home" ,className="tab"),
            dcc.Tab(label="Predict", value="form",className="tab"),
            dcc.Tab(label="Dash Board", value="dashboard",className="tab"),
        ]),
        html.Div(id="tab-content")  
    ])
], fluid=True,className='container')




