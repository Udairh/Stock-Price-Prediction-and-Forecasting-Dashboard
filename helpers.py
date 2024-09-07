import plotly.express as px
import pandas as pd

def get_stock_price_fig(df):
    """Generate a plotly figure for stock prices."""
    fig = px.line(df,
                  x='Date',
                  y=['Open', 'Close'],
                  title="Closing and Opening Price vs Date")
    return fig

def get_indicator_fig(df):
    """Generate a plotly figure for the Exponential Moving Average (EMA)."""
    df['EWA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
    fig = px.scatter(df,
                     x='Date',
                     y='EWA_20',
                     title="Exponential Moving Average vs Date")
    fig.update_traces(mode='lines+markers')
    return fig

def generate_forecast_plot(model, df, forecast_days):
    """Generate a forecast plot using the trained model."""
    # Generate future dates
    last_date = df.index[-1]
    future_dates = [last_date + pd.DateOffset(days=i) for i in range(1, forecast_days + 1)]
    future_df = pd.DataFrame({'Date': future_dates})
    future_df['Date'] = future_df['Date'].map(pd.Timestamp.toordinal)
    
    # Assuming volume is constant for forecast; adjust as needed
    future_df['Volume'] = df['Volume'].iloc[-1]
    
    # Make predictions
    X_future = future_df[['Date', 'Volume']]
    predictions = model.predict(X_future)
    
    # Prepare the forecast plot
    forecast_fig = px.line(x=future_dates, y=predictions, title="Forecasted Prices")
    return forecast_fig
