import dash
from dash import html, dcc, Input, Output
import dash_bootstrap_components as dbc

def layout_home():
    return dbc.Container([
                dbc.Row([
                    dbc.Col(html.H2("About This Site!"), className="mb-2")
                    ]),
                dbc.Row([
                    dbc.Col(html.P(
                        "This dashboard provides insights into transaction data for fraud detection analysis. "
                        "Navigate to the sections below to explore the summary statistics, frequency distribution, "
                        "geographic distribution of fraud, and feature correlation. You can also access a form for data entry "
                        "and a prediction tool to evaluate new transaction data."
                    ), width=8, className="mb-5 text-center mx-auto")
        ])
    ],fluid=True, className='container')
