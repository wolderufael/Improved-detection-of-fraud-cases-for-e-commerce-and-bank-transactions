from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc
from dash import Dash



def layout_dashboard():
    return dbc.Container([
    dbc.Row([
            dbc.Col(html.H2("Dashboard"), className="mb-2")
        ]),
    dbc.Row([
        dcc.Tabs(id="dashboard-tabs", value="home",className="tab", children=[
            dcc.Tab(label="Summary Statistics", value="summary" ,className="tab"),
            dcc.Tab(label="Geographical Distibutiion", value="geo",className="tab"),
            dcc.Tab(label="Frequency Distribution", value="frequency",className="tab"),
        ]),
        html.Div(id="dashboard-content")  
    ])
], fluid=True,className='container')