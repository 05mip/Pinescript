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


@dataclass
class FeatureConfig:
    name: str
    param_a: int
    param_b: int = 1


@dataclass
class Settings:
    neighbors_count: int = 8
    max_bars_back: int = 500
    feature_count: int = 3


class LorentzianClassificationStrategy:
    def __init__(self, settings: Settings = None, features: list = None):
        self.settings = settings or Settings()
        self.features = features or [
            FeatureConfig('RSI', 14),
            FeatureConfig('WT', 10, 11),
            FeatureConfig('CCI', 20)
        ]
        self.feature_arrays = [deque(maxlen=self.settings.max_bars_back) for _ in range(self.settings.feature_count)]
        self.y_train_array = deque(maxlen=self.settings.max_bars_back)
        self.signal = 0
        self.predictions = deque(maxlen=self.settings.neighbors_count)
        self.distances = deque(maxlen=self.settings.neighbors_count)
        self.bars_held = 0

    def calculate_feature(self, feature: FeatureConfig, df: pd.DataFrame) -> pd.Series:
        if feature.name == 'RSI':
            rsi = ta.rsi(df['Close'], length=feature.param_a)
            if feature.param_b > 1:
                rsi = ta.sma(rsi, length=feature.param_b)
            return (rsi - 50) / 50

        elif feature.name == 'WT':
            hlc3 = (df['High'] + df['Low'] + df['Close']) / 3
            ema1 = ta.ema(hlc3, length=feature.param_a)
            ema2 = ta.ema((hlc3 - ema1).abs(), length=feature.param_a)
            ci = (hlc3 - ema1) / (0.015 * ema2)
            wt1 = ta.ema(ci, length=feature.param_b)
            wt2 = ta.sma(wt1, length=4)
            return (wt1 - wt2) / 100

        elif feature.name == 'CCI':
            cci = ta.cci(df['High'], df['Low'], df['Close'], length=feature.param_a)
            return np.tanh(cci / 500)

        else:
            raise ValueError(f"Unknown feature: {feature.name}")

    def get_lorentzian_distance(self, current_features, i):
        dist = 0
        for j in range(self.settings.feature_count):
            if len(self.feature_arrays[j]) > i:
                past_val = list(self.feature_arrays[j])[-1 - i]
                dist += np.log(1 + abs(current_features[j] - past_val))
        return dist

    def update_ml_model(self, current_features, label):
        self.y_train_array.append(label)
        for i, f in enumerate(current_features):
            self.feature_arrays[i].append(f)

        self.predictions.clear()
        self.distances.clear()

        loop_range = min(self.settings.max_bars_back - 1, len(self.y_train_array) - 1)
        for i in range(0, loop_range, 4):
            if i >= len(self.y_train_array):
                break
            d = self.get_lorentzian_distance(current_features, i)
            self.distances.append(d)
            self.predictions.append(list(self.y_train_array)[-1 - i])
            if len(self.predictions) > self.settings.neighbors_count:
                self.predictions.popleft()
                self.distances.popleft()
        return sum(self.predictions) if self.predictions else 0

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        features_data = []

        for f in self.features[:self.settings.feature_count]:
            feature_series = self.calculate_feature(f, df)
            features_data.append(feature_series)

        df['Signal'] = 0
        df['Long_Entry'] = False
        df['Short_Entry'] = False
        df['Long_Exit'] = False
        df['Short_Exit'] = False

        for i in range(50, len(df) - 4):  # make sure we have forward price for labeling
            current_features = [fd.iloc[i] for fd in features_data]
            if any(pd.isna(f) for f in current_features):
                continue

            # Label based on future price
            future_price = df['Close'].iloc[i + 4]
            current_price = df['Close'].iloc[i]
            label = np.sign(future_price - current_price)

            prediction = self.update_ml_model(current_features, label)
            new_signal = int(np.sign(prediction))
            changed = new_signal != self.signal
            self.signal = new_signal

            if changed:
                self.bars_held = 0
            else:
                self.bars_held += 1

            df.at[df.index[i], 'Signal'] = self.signal

            if changed and self.signal == 1:
                df.at[df.index[i], 'Long_Entry'] = True
            elif changed and self.signal == -1:
                df.at[df.index[i], 'Long_Exit'] = True

            # if self.bars_held >= 4:
            #     if self.signal == 1:
            #         df.at[df.index[i], 'Long_Exit'] = True
            #     elif self.signal == -1:
            #         df.at[df.index[i], 'Short_Exit'] = True

        return df


# Strategy class for backtesting.py compatibility
class LorentzianStrategyBT(Strategy):
    def init(self):
        # Precompute signals once at the beginning using full dataset
        df = self.data.df.copy()
        self.lorentz = LorentzianClassificationStrategy()  # You can pass settings here
        signals_df = self.lorentz.generate_signals(df)
        self.signals = signals_df
        self.signal_series = signals_df['Signal'].values
        self.long_entry = signals_df['Long_Entry'].values
        self.long_exit = signals_df['Long_Exit'].values
        self.short_entry = signals_df['Short_Entry'].values
        self.short_exit = signals_df['Short_Exit'].values

    def next(self):
        i = len(self.data) - 1
        if i >= len(self.signal_series):
            return
        
        if self.position:
            # Exit logic
            if self.position.is_long and self.long_exit[i]:
                self.position.close()
            elif self.position.is_short and self.short_exit[i]:
                self.position.close()
        else:
            # Entry logic
            if self.long_entry[i]:
                self.buy()

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
