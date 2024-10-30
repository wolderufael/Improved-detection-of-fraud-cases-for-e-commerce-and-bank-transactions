import dash
from dash import html, dcc, Input, Output
import dash_bootstrap_components as dbc

def layout_summary():
    return dbc.Container([
        dbc.Row([
            dbc.Col(html.H2("Summary Statistics"), className="mb-2")
            ]),
        dbc.Row([
            # dbc.Col(html.H4("Summary Statistics", className="mb-4")),
            dbc.Col(dbc.CardGroup([
                dbc.Card([
                    dbc.CardBody([
                        html.H5(id="total-transactions", className="card-title"),
                        html.P("Total Transactions")
                    ])
                ], color="primary", inverse=True),
                dbc.Card([
                    dbc.CardBody([
                        html.H5(id="total-fraud-cases", className="card-title"),
                        html.P("Total Fraud Cases")
                    ])
                ], color="danger", inverse=True),
                dbc.Card([
                    dbc.CardBody([
                        html.H5(id="fraud-percentage", className="card-title"),
                        html.P("Fraud Percentage (%)")
                    ])
                ], color="warning", inverse=True)
            ]))
        ], className="mb-4"),
        dbc.Row([
            dbc.Col(dcc.Graph(id="fraud-trend-chart"))
    ])
    ],fluid=True, className='container')
    
