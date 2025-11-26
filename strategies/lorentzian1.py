import pandas as pd
import numpy as np
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
import yfinance as yf
from datetime import datetime, timedelta
import time
import pytz
import requests
from scipy.fft import fft, ifft, fftfreq
from scipy.signal import butter, filtfilt
import pandas_ta as ta
from dataclasses import dataclass
from collections import deque
import warnings
# Method 3: Suppress all FutureWarnings (most broad - use cautiously)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="backtesting.backtesting")

class LorentzianStrategy(Strategy):
    pass

def fetch_data(symbol, start_date, end_date, interval='15m', max_retries=3, delay=2):
    """
    Fetch data from Yahoo Finance with retry mechanism
    :param symbol: Stock symbol (e.g., 'AAPL')
    :param start_date: Start date (str or datetime)
    :param end_date: End date (str or datetime)
    :param interval: Data interval ('1d', '1h', '15m', etc.)
    :param max_retries: Maximum number of retry attempts
    :param delay: Delay between retries in seconds
    :return: DataFrame with OHLC data
    """
    # Convert dates to UTC if they're not already
    if isinstance(start_date, datetime):
        start_date = start_date.astimezone(pytz.UTC)
    if isinstance(end_date, datetime):
        end_date = end_date.astimezone(pytz.UTC)
    
    for attempt in range(max_retries):
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(start=start_date, end=end_date, interval=interval)
            if not data.empty:
                # Convert index from UTC to PST
                data.index = data.index.tz_convert('America/Los_Angeles')
                return data
            print(f"Attempt {attempt + 1}: No data received, retrying...")
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {str(e)}")
            if attempt < max_retries - 1:
                print(f"Waiting {delay} seconds before retrying...")
                time.sleep(delay)
            else:
                raise Exception(f"Failed to fetch data after {max_retries} attempts")
    return None

def fetch_data_from_pionex(symbol="XRP_USDT", interval="15M", limit=500):
    url = "https://api.pionex.com/api/v1/market/klines"
    params = {
        "symbol": symbol,
        "interval": interval,
        "limit": limit
    }

    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()

        if not data.get("result"):
            print("API returned an error:", data)
            return None

        klines = data["data"]["klines"]
        df = pd.DataFrame(klines)
        df["timestamp"] = pd.to_datetime(df["time"], unit='ms', utc=True)
        df.set_index("timestamp", inplace=True)

        df = df.rename(columns={
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "volume": "Volume"
        })

        # Convert all columns to numeric types
        df = df[["Open", "High", "Low", "Close", "Volume"]].astype(float)

        return df

    except Exception as e:
        print("Error fetching Pionex data:", e)
        return None

def fetch_from_file():
    df = pd.read_csv('data.csv', parse_dates=['timestamp'])
    df.set_index('timestamp', inplace=True)
    return df

if __name__ == '__main__':
    # Example usage with yfinance
    symbol = 'RAY-USD'
    # symbol = 'XRP-USD'
    # symbol = 'SOXL'
    end_date = datetime.now()
    start_date = end_date - timedelta(days=59)  # 59 days of data
    
    print(f"Fetching data for {symbol}...")
    # Fetch data with retry mechanism
    data = fetch_data(symbol, start_date, end_date, "15m")
    # data = fetch_data_from_pionex(symbol, "15M")
    # data = fetch_from_file()
    
    if data is not None and not data.empty:
        print(f"Successfully fetched {len(data)} data points")
        # Run backtest
        bt = Backtest(data, LorentzianStrategyBT, cash=100000, commission=.001)
        stats = bt.run()
        print(stats)
        
        
        # data.to_csv('data.csv')

        # Export trades to CSV
        trades = stats['_trades']
        csv_filename = f'trades1.csv'
        trades.to_csv(csv_filename)
        # print(f"\nExported trades to {csv_filename}")
        
        # bt.plot()
        bt.plot(filename='backtest_report.html', open_browser=True)

    else:
        print("Failed to fetch data. Please try again later.")
