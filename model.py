import yfinance as yf
import pandas as pd
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split, GridSearchCV
import numpy as np

def fetch_stock_data(stock_code):
    """Fetch historical stock data using yfinance."""
    ticker = yf.Ticker(stock_code)
    df = ticker.history(period="1y")  # Fetch data for the last year
    return df

def prepare_data(df):
    """Prepare the stock data for training the machine learning model."""
    # Convert date index to numerical format
    df['Date'] = pd.to_datetime(df.index)
    df['Date'] = df['Date'].map(pd.Timestamp.toordinal)

    # Use date and volume as features for training
    X = df[['Date', 'Volume']].values  
    y = df['Close'].values  # Target is the 'Close' price

    # Split data into training and testing sets (90% train, 10% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train):
    """Train the Support Vector Regression model."""
    svr = SVR(kernel='rbf')
    # Set up GridSearchCV to find the best parameters
    parameters = {'C': [1e0, 1e1, 1e2], 'gamma': np.logspace(-2, 2, 5)}
    grid_search = GridSearchCV(svr, parameters, cv=5)
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_  # Get the model with the best parameters
    return best_model
