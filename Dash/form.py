from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc
from dash import Dash

# app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

def form_layout(app):
  return dbc.Container([
    dbc.Row([
        dbc.Col(html.H2("Fraud Detection"), className="text-center mb-4")
    ]),

    dbc.Row([
        # Left column
        dbc.Col([
            dbc.Label("User ID:"),
            dbc.Input(type="text", id="user_id", required=True, placeholder="Enter User ID"),
            html.Br(),

            dbc.Label("Signup Time:"),
            dbc.Input(type="datetime-local", id="signup_time", required=True),
            html.Br(),

            dbc.Label("Source:"),
            dcc.Dropdown(
                id="source",
                options=[
                    {"label": "Ads", "value": "Ads"},
                    {"label": "Direct", "value": "Direct"},
                    {"label": "SEO", "value": "SEO"}
                ],
                value="Ads",
                clearable=False,
                className="dropdown-custom"
            ),
            html.Br(),

            dbc.Label("Purchase Value:"),
            dbc.Input(type="number", id="purchase_value", required=True, placeholder="Enter Purchase Value"),
            html.Br(),

            dbc.Label("Age:"),
            dbc.Input(type="number", id="age", required=True, placeholder="Enter Age")
        ], width=6, className="left"),

        # Right column
        dbc.Col([
            dbc.Label("Device ID:"),
            dbc.Input(type="text", id="device_id", required=True, placeholder="Enter Device ID"),
            html.Br(),

            dbc.Label("Purchase Time:"),
            dbc.Input(type="datetime-local", id="purchase_time", required=True),
            html.Br(),

            dbc.Label("Browser:"),
            dcc.Dropdown(
                id="browser",
                options=[
                    {"label": "Chrome", "value": "Chrome"},
                    {"label": "Firefox", "value": "Firefox"},
                    {"label": "IE", "value": "IE"},
                    {"label": "Opera", "value": "Opera"},
                    {"label": "Safari", "value": "Safari"}
                ],
                value="Chrome",
                clearable=False
            ),
            html.Br(),

            dbc.Label("Sex:"),
            dcc.Dropdown(
                id="sex",
                options=[
                    {"label": "Male", "value": "M"},
                    {"label": "Female", "value": "F"}
                ],
                value="M",
                clearable=False
            ),
            html.Br(),

            dbc.Label("Country:"),
            dbc.Input(type="text", id="country", required=True, placeholder="Enter Country")
        ], width=6, className="right")
    ]),

    dbc.Row([
        dbc.Col(dbc.Button("Submit", id="submit", color="primary", className="mt-6"), width={"size": 6, "offset": 3})
    ]),
    html.Div(id="result")
], fluid=True, className="form-container")



