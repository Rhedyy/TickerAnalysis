import requests
import pandas as pd
import json
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt

#Pre-requisite Libraries
#pip install requests
#pip install pandas
#pip install statsmodels
#pip install matplotlib

# Tickers List
tickers = ['XRPUSDT', 'ETHUSDT', 'RNDRUSDT', 'USDCUSDT']  # Assuming the tickers are in the format 'SYMBOLUSDT'

# Function to fetch and process data for a given ticker. Using BINANCE API.
def fetch_ticker_data(symbol):
    url = 'https://api.binance.com/api/v3/klines'
    interval = '1d'  # Daily data
    limit = '60'  # Last 60 days
    par = {'symbol': symbol, 'interval': interval, 'limit': limit}
    response = requests.get(url, params=par)
    response_data = json.loads(response.text)

    # Extract the relevant data and create a DataFrame with a timestamp index
    binance_data = pd.DataFrame(response_data, columns=['Open time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close time',
                                                       'Quote asset volume', 'Number of trades', 'Taker buy base asset volume',
                                                       'Taker buy quote asset volume', 'Ignore'])

    # Convert the timestamp to a datetime object and set it as the index
    binance_data['Open time'] = pd.to_datetime(binance_data['Open time'], unit='ms')
    binance_data.set_index('Open time', inplace=True)
    
    # Technical Analysis
    
    #1. Simple Moving Averages
    # Calculate the 5-day and 20-day SMAs
    binance_data['Close'] = binance_data['Close'].astype(float)
    binance_data['SMA5'] = calculate_sma(binance_data['Close'], window=5)
    binance_data['SMA20'] = calculate_sma(binance_data['Close'], window=20)
    
    #2. Relative Strength Index
    binance_data['RSI14'] = calculate_rsi(binance_data['Close'], window=14)
    
    #3. Calculate Fibonacci retracement levels
    fibonacci_levels = calculate_fibonacci_levels(binance_data)
    
    # 4. Time-Series Forecasting of Next Week Prices
    arima_forecast = forecast_arima(binance_data['Close'])
    
    # Identify bullish tickers
    closest_fibonacci = min(fibonacci_levels.values(), key=lambda x: abs(x - binance_data['Close'].iloc[-1]))
    bullish_ticker = (binance_data['SMA5'] > binance_data['SMA20']) & (binance_data['RSI14'] > 60) & (binance_data['RSI14'] > binance_data['RSI14'].shift(1)) & (binance_data['Close'] > 1.02 * closest_fibonacci)

    # Possible additional criteria here (Sentiment Analysis, Fundamental Analysis, More TA ETC)
    
    # Save data to CSV file
    binance_data.to_csv(f'{symbol}_data.csv')

    return binance_data, bullish_ticker, arima_forecast

# Function to calculate the Simple Moving Average (SMA)
def calculate_sma(data, window):
    return data.rolling(window).mean()

# Function to calculate the Relative Strength Index 
def calculate_rsi(data, window):
    delta = data.diff(1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Function to perform ARIMA forecasting
def forecast_arima(data):
    model = ARIMA(data, order=(7, 1, 0), freq='D')
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=7)  # Forecast the next 7 values
    return forecast

# Function to calculate Fibonacci retracement levels
def calculate_fibonacci_levels(data, window = 30):
    last_14_data = data.iloc[-window:]
    high = last_14_data['High'].astype(float)
    low = last_14_data['Low'].astype(float)
    
    # Calculate the highest and lowest prices for the given data
    highest_price = max(high)
    lowest_price = min(low)
    
    # Calculate Fibonacci levels
    fibonacci_levels = {
        0.236: highest_price - (0.236 * (highest_price - lowest_price)),
        0.382: highest_price - (0.382 * (highest_price - lowest_price)),
        0.618: highest_price - (0.618 * (highest_price - lowest_price)),
    }
    
    return fibonacci_levels

# Fetch data for each ticker and store in the dictionary
ticker_data = {}
for ticker in tickers:
    data, bullish, arima_forecast = fetch_ticker_data(ticker)
    ticker_data[ticker] = {'data': data, 'bullish': bullish, 'arima_forecast': arima_forecast}

# Return the top 2 tickers that are highly likely to be bullish and their ARIMA forecasts
top_bullish_tickers = []
for ticker, data in ticker_data.items():
    if data['bullish'].iloc[-1] and data['bullish'].iloc[-2]:
        last_week_close = data['data']['Close'].iloc[-7]
        current_close = data['data']['Close'].iloc[-1]
        price_change = ((current_close - last_week_close) / last_week_close) * 100

        if price_change > 2:
            top_bullish_tickers.append((ticker, price_change, data['arima_forecast']))

# Sort the tickers by price change and select the top 2
top_bullish_tickers.sort(key=lambda x: x[1], reverse=True)

# Calculate the percentage difference between maximum ARIMA forecast and current price
for ticker, data in ticker_data.items():
    current_close = data['data']['Close'].iloc[-1]
    max_arima_forecast = max(data['arima_forecast'])
    
    percentage_difference = ((max_arima_forecast - current_close) / current_close) * 100
    
    if percentage_difference > 2:
        print(f'{ticker}: Current Price: {current_close:.4f}, Max ARIMA Forecast: {max_arima_forecast:.2f}, Percentage Difference: {percentage_difference:.4f}%')

# Print ARIMA forecasts for top two bullish tickers
for i, (ticker, price_change, arima_forecast) in enumerate(top_bullish_tickers[:2]):
    current_close = ticker_data[ticker]['data']['Close'].iloc[-1]
    max_arima_forecast = max(arima_forecast)
    percentage_difference = ((max_arima_forecast - current_close) / current_close) * 100
    
    print(f'{ticker} - Weekly Price Change: {price_change:.2f}%')
    print(f'ARIMA Forecast for the next 7 candles: {arima_forecast}')
    print(f'Percentage Difference (Next Week Forecast): {percentage_difference:.4f}%')
    
# Create a single figure with two subplots
fig, axs = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

for i, (ticker, price_change, arima_forecast) in enumerate(top_bullish_tickers[:2]):
    # Plot historical data for each top ticker
    axs[i].plot(ticker_data[ticker]['data'].index, ticker_data[ticker]['data']['Close'], label=f'{ticker} Close Prices')
    current_close = ticker_data[ticker]['data']['Close'].iloc[-1]
    max_arima_forecast = max(arima_forecast)
    percentage_difference = ((max_arima_forecast - current_close) / current_close) * 100
    
    # Plot ARIMA forecast for each top ticker
    forecast_dates = pd.date_range(start=ticker_data[ticker]['data'].index[-1], periods=len(arima_forecast))
    axs[i].plot(forecast_dates, arima_forecast, linestyle='dashed', label=f'{ticker} ARIMA Forecast')
    axs[i].set_title(f'{ticker} - Price Forecast: {percentage_difference:.2f}%')

axs[0].legend()
axs[1].legend()
axs[1].set_xlabel('Date')
plt.show()