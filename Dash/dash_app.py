from dash import Dash, dcc, html, Input, Output, callback_context
import dash_bootstrap_components as dbc
import requests

# Initialize Dash app
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.H2("Fraud Detection"), className="text-center mb-4")
    ]),

    dbc.Row([
        dbc.Col([
            dbc.Label("User ID:"),
            dcc.Input(type="text", id="user_id", placeholder="Enter User ID", required=True),
            html.Br(),

            dbc.Label("Signup Time:"),
            dbc.Input(type="datetime-local", id="signup_time", required=True),  # Changed to dbc.Input
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
                clearable=False
            ),
            html.Br(),

            dbc.Label("Purchase Value:"),
            dcc.Input(type="number", id="purchase_value", placeholder="Enter Purchase Value", required=True),
            html.Br(),

            dbc.Label("Age:"),
            dcc.Input(type="number", id="age", placeholder="Enter Age", required=True)
        ], width=6),

        dbc.Col([
            dbc.Label("Device ID:"),
            dcc.Input(type="text", id="device_id", placeholder="Enter Device ID", required=True),
            html.Br(),

            dbc.Label("Purchase Time:"),
            dbc.Input(type="datetime-local", id="purchase_time", required=True),  # Changed to dbc.Input
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
            dcc.Input(type="text", id="country", placeholder="Enter Country", required=True)
        ], width=6)
    ]),

    dbc.Row([
        dbc.Col(dbc.Button("Submit", id="submit-button", color="primary", className="mt-6"), width={"size": 6, "offset": 3})
    ]),

    html.Div(id="result")  # Placeholder for the prediction result
], fluid=True)

@app.callback(
    Output("result", "children"),
    Input("submit-button", "n_clicks"),
    [Input("user_id", "value"),
     Input("signup_time", "value"),
     Input("source", "value"),
     Input("purchase_value", "value"),
     Input("age", "value"),
     Input("device_id", "value"),
     Input("purchase_time", "value"),
     Input("browser", "value"),
     Input("sex", "value"),
     Input("country", "value")]
)
def submit_form(n_clicks, user_id, signup_time, source, purchase_value, age, device_id, purchase_time, browser, sex, country):
    if n_clicks is None:
        return ""
    
    # Prepare the data to send
    form_data = {
        "user_id": user_id,
        "signup_time": signup_time,
        "source": source,
        "purchase_value": purchase_value,
        "age": age,
        "device_id": device_id,
        "purchase_time": purchase_time,
        "browser": browser,
        "sex": sex,
        "country": country
    }

    try:
        # Send a POST request to Flask
        response = requests.post("http://127.0.0.1:5000/submit", data=form_data)
        if response.status_code == 200:
            result = response.json()
            return f"Prediction: {result['meaning']} (Code: {result['prediction']})"
        else:
            return "Error: Could not get a prediction"
    except requests.exceptions.RequestException as e:
        return f"Error: {str(e)}"

if __name__ == '__main__':
    app.run_server(debug=True, port=4000)
