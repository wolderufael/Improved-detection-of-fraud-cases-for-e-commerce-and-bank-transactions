# from dash import html, dcc
# import dash_bootstrap_components as dbc

# def create_layout(app):
#     return dbc.Container([
#         dbc.Row([
#             dbc.Col(html.H1("Fraud Data Dashboard"), className="mb-2")
#         ]),
#         dbc.Row([
#             dbc.Col(html.H2(children='Summary Statistics'), className="mb-4")
#         ]),
#         dbc.Row([
#         dbc.Col(dbc.Card([
#             dbc.CardBody([
#                 html.H5(id="total-transactions", className="card-title"),
#                 html.P("Total Transactions")
#             ])
#         ], color="primary", inverse=True)),
        
#         dbc.Col(dbc.Card([
#             dbc.CardBody([
#                 html.H5(id="total-fraud-cases", className="card-title"),
#                 html.P("Total Fraud Cases")
#             ])
#         ], color="danger", inverse=True)),
        
#         dbc.Col(dbc.Card([
#             dbc.CardBody([
#                 html.H5(id="fraud-percentage", className="card-title"),
#                 html.P("Fraud Percentage (%)")
#             ])
#         ], color="warning", inverse=True))
#     ], className="mb-4"),
#         dbc.Row([
#             dbc.Col(html.H2(children='Freqquency Distribution of Features'), className="mb-4")
#         ]),
#         dbc.Row([
#             dbc.Col([
#                 dbc.CardGroup([
#                     dbc.Label("Select feature"),
#                     dcc.Dropdown(
#                         id='feature-dropdown',
#                         options=[],
#                         value=None,
#                         clearable=False
#                     )
#                 ])
#             ], md=4),
#         ], className="mb-3"),
#         dbc.Row([
#             dbc.Col([
#                 dcc.Graph(id='frequency-distribution')
#             ])
#         ]),
#         dbc.Row([
#             dbc.Col(html.H2(children='Correlation of features with Fraud'), className="mb-4")
#         ]),
#         dbc.Row([
#             dbc.Col([
#                 dcc.Graph(id='contingency-table')
#             ])
#         ]),
#         dbc.Row([
#             dbc.Col(html.H5(id='cramers-v-output')),
#         ]),
#         dbc.Row([
#             dbc.Col(html.H2(children='Fraud Ditribution over the world'), className="mb-4")
#         ]),
#         dbc.Row([
#             dbc.Col([
#                 dcc.Graph(id='fraud-country-map')
#             ])
#         ])
#     ], fluid=True,className='container')






#########################################################################################################
# from dash import html, dcc
# import dash_bootstrap_components as dbc

# def create_layout(app):
#     return dbc.Container([
#         # Header section
#         dbc.Row([
#             dbc.Col(html.H1("Fraud Detection Dashboard"), className="mb-4 text-center")
#         ]),
#         # # Navigation Links
#         # dbc.Row([
#         #     dbc.Col(dbc.Button("Form", href="/form", color="primary", className="mx-1"), width="auto"),
#         #     dbc.Col(dbc.Button("Prediction", href="/prediction", color="success", className="mx-1"), width="auto"),
#         #     dbc.Col(dbc.Button("Dashboard", href="/dashboard", color="info", className="mx-1"), width="auto")
#         # ], className="mb-5 justify-content-center"),
#         dbc.Row([
#             html.Div([
#             dcc.Location(id='url', refresh=False),  # Location component to manage URL
#             dbc.NavbarSimple(
#                 children=[
#                     dbc.NavItem(dbc.NavLink("Home", href="/form")),
#                     dbc.NavItem(dbc.NavLink("Another Page", href="/another")),
#                 ],
#                 brand="My Dashboard",
#                 brand_href="/",
#                 color="primary",
#                 dark=True,
#             ),
#             html.Div(id='page-content')  # Content will be rendered here
#         ])
#         ]),
#         # Description Section
#         dbc.Row([
#             dbc.Col(html.P(
#                 "This dashboard provides insights into transaction data for fraud detection analysis. "
#                 "Navigate to the sections below to explore the summary statistics, frequency distribution, "
#                 "geographic distribution of fraud, and feature correlation. You can also access a form for data entry "
#                 "and a prediction tool to evaluate new transaction data."
#             ), width=8, className="mb-5 text-center mx-auto")
#         ]),

#         # Dashboard Content
#         dbc.Row([
#             dbc.Col(html.H2("Dashboard Overview", className="mb-4 text-center")),
#         ]),

#         # Summary Statistics Section
#         dbc.Row([
#             dbc.Col(html.H4("Summary Statistics", className="mb-4")),
#             dbc.Col(dbc.CardGroup([
#                 dbc.Card([
#                     dbc.CardBody([
#                         html.H5(id="total-transactions", className="card-title"),
#                         html.P("Total Transactions")
#                     ])
#                 ], color="primary", inverse=True),
#                 dbc.Card([
#                     dbc.CardBody([
#                         html.H5(id="total-fraud-cases", className="card-title"),
#                         html.P("Total Fraud Cases")
#                     ])
#                 ], color="danger", inverse=True),
#                 dbc.Card([
#                     dbc.CardBody([
#                         html.H5(id="fraud-percentage", className="card-title"),
#                         html.P("Fraud Percentage (%)")
#                     ])
#                 ], color="warning", inverse=True)
#             ]))
#         ], className="mb-4"),

#         # Frequency Distribution Section
#         dbc.Row([
#             dbc.Col(html.H4("Frequency Distribution of Features", className="mb-4")),
#             dbc.Col([
#                 dbc.Label("Select Feature"),
#                 dcc.Dropdown(
#                     id='feature-dropdown',
#                     options=[],
#                     value=None,
#                     clearable=False
#                 )
#             ], width=4)
#         ]),
#         dbc.Row([
#             dbc.Col(dcc.Graph(id='frequency-distribution'), width=12)
#         ], className="mb-5"),

#         # Correlation Section
#         dbc.Row([
#             dbc.Col(html.H4("Correlation of Features with Fraud", className="mb-4")),
#             dbc.Col(dcc.Graph(id='contingency-table'), width=12),
#             dbc.Col(html.H5(id='cramers-v-output'), className="text-center")
#         ], className="mb-5"),

#         # Geographic Distribution Section
#         dbc.Row([
#             dbc.Col(html.H4("Fraud Distribution Over the World", className="mb-4")),
#             dbc.Col(dcc.Graph(id='fraud-country-map'), width=12)
#         ], className="mb-5")
        
#     ], fluid=True, className='container')

################################################################################################

from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc
from dash import Dash

# app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

def create_layout(app):
    return dbc.Container([
    dbc.Row([
            dbc.Col(html.H1("Fraud Data Dashboard"), className="mb-2")
        ]),
    dbc.Row([
        dcc.Tabs(id="tabs", value="home",className="tab", children=[
            dcc.Tab(label="Home", value="home" ,className="tab"),
            dcc.Tab(label="Predict", value="form",className="tab"),
            dcc.Tab(label="Another Page", value="another",className="tab"),
        ]),
        html.Div(id="tab-content")  # Dynamic content will render here based on selected tab
    ])
], fluid=True,className='container')




