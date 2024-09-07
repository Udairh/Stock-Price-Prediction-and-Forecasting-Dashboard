import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import yfinance as yf
from model import train_model, prepare_data, fetch_stock_data
from helpers import generate_forecast_plot, get_indicator_fig, get_stock_price_fig

# Initialize the Dash app
app = dash.Dash(__name__, suppress_callback_exceptions=True)
server = app.server

# Layout of the app
app.layout = html.Div([
    html.Div([
        html.P("Welcome to the Stock Dash App!", className="start"),
        html.Div([
            dcc.Input(id='stock-input', type='text', placeholder="Enter Stock Code"),
            html.Button('Submit', id='submit-button', n_clicks=0)  # Added Submit Button
        ], className="stock-code"),
        html.Div([
            dcc.DatePickerRange(id='date-picker', start_date='2023-01-01', end_date='2024-01-01'),
            html.Button('Stock Price', id='stock-price-button', n_clicks=0),
            html.Button('Indicators', id='indicators-button', n_clicks=0),
            dcc.Input(id='forecast-days', type='number', placeholder="Number of days for forecast"),
            html.Button('Forecast', id='forecast-button', n_clicks=0),
        ], className="inputs"),
    ], className="nav"),

    html.Div([
        html.Div([
            html.Img(id="logo", src="assets/stock.jpg", style={"width": "250px"}),  # Static image URL
            html.H1(id="company-name", children="Company Name"),
        ], className="header"),

        html.Div(id="description", className="description_ticker"),
        html.Div(id="graphs-content"),
        html.Div(id="main-content"),
        html.Div(id="forecast-content"),
    ], className="content"),
])

# Callback to update company information
@app.callback(
    [Output("description", "children"),
     Output("company-name", "children")],
    [Input("submit-button", "n_clicks")],
    [State("stock-input", "value")]
)
def update_company_info(n_clicks, stock_code):
    if n_clicks == 0 or not stock_code:
        raise dash.exceptions.PreventUpdate
    ticker = yf.Ticker(stock_code)
    inf = ticker.info
    
    # Get company description
    description = inf.get('longBusinessSummary', 'No description available.')
    
    # Get company name
    company_name = inf.get('shortName', 'Unknown')
    
    return description, company_name


# Callback to update stock price plot
@app.callback(
    Output("graphs-content", "children"),
    [Input("stock-price-button", "n_clicks")],
    [State("stock-input", "value"),
     State("date-picker", "start_date"),
     State("date-picker", "end_date")]
)
def update_stock_data(n_clicks, stock_code, start_date, end_date):
    if n_clicks == 0 or not stock_code:
        raise dash.exceptions.PreventUpdate
    df = yf.download(stock_code, start=start_date, end=end_date)
    df.reset_index(inplace=True)
    fig = get_stock_price_fig(df)
    return [dcc.Graph(figure=fig)]

# Callback to update indicator plot
@app.callback(
    Output("main-content", "children"),
    [Input("indicators-button", "n_clicks")],
    [State("stock-input", "value"),
     State("date-picker", "start_date"),
     State("date-picker", "end_date")]
)
def update_indicator_plot(n_clicks, stock_code, start_date, end_date):
    if n_clicks == 0 or not stock_code:
        raise dash.exceptions.PreventUpdate
    df = yf.download(stock_code, start=start_date, end=end_date)
    df.reset_index(inplace=True)
    fig = get_indicator_fig(df)
    return [dcc.Graph(figure=fig)]

# Callback to predict stock prices
@app.callback(
    Output("forecast-content", "children"),
    [Input("forecast-button", "n_clicks")],
    [State("stock-input", "value"),
     State("forecast-days", "value")]
)
def predict_stock_price(n_clicks, stock_code, forecast_days):
    if n_clicks == 0 or not stock_code or not forecast_days:
        raise dash.exceptions.PreventUpdate
    df = fetch_stock_data(stock_code)
    X_train, X_test, y_train, y_test = prepare_data(df)
    model = train_model(X_train, y_train)
    # Use the model to predict here and generate forecast plot
    forecast_fig = generate_forecast_plot(model, df, forecast_days)
    return [dcc.Graph(figure=forecast_fig)]

if __name__ == '__main__':
    app.run_server(debug=True)
