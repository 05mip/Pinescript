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

class HARSIStrategy(Strategy):
    # Strategy parameters
    length_rsi = 31
    length_stoch = 21
    smooth_k = 9
    smooth_d = 10
    window = 100
    ha_smooth_period = 25  # New parameter for Heikin Ashi smoothing
    rsi_oversold = 0.37
    rsi_overbought = 0.72

    use_butterworth_filter = True
    filter_order = 3
    filter_cutoff = 0.176

    sl = 0.01

    def init(self):
        # Calculate Heikin Ashi values
        self.entry_price = 0
        # Step 1: Base Heikin Ashi Close
        self.ha_close = (self.data.Open + self.data.High + self.data.Low + self.data.Close) / 4

        # Step 2: Initialize arrays for HA components
        ha_open = np.zeros(len(self.data))
        ha_high = np.zeros(len(self.data))
        ha_low = np.zeros(len(self.data))

        # Step 3: Calculate Heikin Ashi Open
        ha_open[0] = self.data.Open[0]
        for i in range(1, len(self.data)):
            ha_open[i] = (ha_open[i-1] + self.ha_close[i-1]) / 2

        # Step 4: Calculate High and Low
        for i in range(len(self.data)):
            ha_high[i] = max(self.data.High[i], ha_open[i], self.ha_close[i])
            ha_low[i] = min(self.data.Low[i], ha_open[i], self.ha_close[i])

        self.ha_open = ha_open
        self.ha_high = ha_high
        self.ha_low = ha_low

        # Step 5: Apply smoothing
        def smooth(x, period):
            """Causal moving average using pandas"""
            return pd.Series(x).rolling(window=period, min_periods=1).mean().values

        self.ha_open = smooth(self.ha_open, self.ha_smooth_period)
        self.ha_close = smooth(self.ha_close, self.ha_smooth_period)
        self.ha_high = smooth(self.ha_high, self.ha_smooth_period)
        self.ha_low = smooth(self.ha_low, self.ha_smooth_period)

        # Scaled version as indicators
        def scale_ha(values, ha_low, ha_high):
            scaled = np.zeros_like(values)
            for i in range(len(values)):
                start = max(0, i - self.window + 1)
                local_min = np.min(ha_low[start:i+1])
                local_max = np.max(ha_high[start:i+1])
                scale = 1.0 / (local_max - local_min) if local_max != local_min else 1.0
                scaled[i] = (values[i] - local_min) * scale
            return scaled

        self.ha_open_scaled = self.I(scale_ha, self.ha_open, self.ha_low, self.ha_high)
        self.ha_close_scaled = self.I(scale_ha, self.ha_close, self.ha_low, self.ha_high)
        self.ha_high_scaled = self.I(scale_ha, self.ha_high, self.ha_low, self.ha_high)
        self.ha_low_scaled = self.I(scale_ha, self.ha_low, self.ha_low, self.ha_high)
        # Calculate RSI and Stochastic RSI
        # Alternative vectorized version (more efficient)
        def calc_rsi(close, period=self.length_rsi):
            """
            Vectorized RSI calculation - more efficient for large datasets
            """
            close = np.array(close, dtype=float)
            delta = np.diff(close)

            gain = np.where(delta > 0, delta, 0)
            loss = np.where(delta < 0, -delta, 0)

            # Use pandas-style exponential weighted mean if available
            gain_ema = pd.Series(gain).ewm(span=period, adjust=False).mean()
            loss_ema = pd.Series(loss).ewm(span=period, adjust=False).mean()

            rs = gain_ema / loss_ema
            rsi_values = 100 - (100 / (1 + rs))

            # Pad with NaN for first value and return
            rsi = np.full(len(close), np.nan)
            rsi[1:] = rsi_values
            rsi[:period] = np.nan  # First 'period' values should be NaN

            return rsi

        rsi = self.I(calc_rsi, self.data.Close.copy())

        def calc_stoch_rsi(rsi):
            stoch_rsi = np.zeros_like(rsi)
            for i in range(len(rsi)):
                if i < self.length_stoch:
                    stoch_rsi[i] = 0
                else:
                    rsi_min = np.min(rsi[i-self.length_stoch+1:i+1])
                    rsi_max = np.max(rsi[i-self.length_stoch+1:i+1])
                    stoch_rsi[i] = (rsi[i] - rsi_min) / (rsi_max - rsi_min) if rsi_max != rsi_min else 0
            return stoch_rsi

        self.stoch_rsi = self.I(calc_stoch_rsi, rsi)
        if self.use_butterworth_filter:
            self.stoch_rsi = self.I(self.butterworth_filter, self.stoch_rsi)

        # Calculate K and D lines
        def calc_k(stoch_rsi):
            k = np.zeros_like(stoch_rsi)
            for i in range(len(stoch_rsi)):
                if i < self.smooth_k:
                    k[i] = np.mean(stoch_rsi[:i+1])
                else:
                    k[i] = np.mean(stoch_rsi[i-self.smooth_k+1:i+1])
            return k

        def calc_d(k):
            d = np.zeros_like(k)
            for i in range(len(k)):
                if i < self.smooth_d:
                    d[i] = np.mean(k[:i+1])
                else:
                    d[i] = np.mean(k[i-self.smooth_d+1:i+1])
            return d

        self.k = self.I(calc_k, self.stoch_rsi)
        self.d = self.I(calc_d, self.k)

        # Rescale K and D from [0, 1] to [-40, 40]
        self.k_scaled = self.I(lambda k: k * 80 - 40, self.k)
        self.d_scaled = self.I(lambda d: d * 80 - 40, self.d)

    def butterworth_filter(self, signal):
        """
        Apply Butterworth low-pass filter - simpler than FFT
        """
        # Need at least 3x the filter order points
        min_length = 3 * self.filter_order

        if len(signal) < min_length:
            return signal

        # Remove NaN values for filtering
        valid_mask = ~np.isnan(signal)
        if not np.any(valid_mask):
            return signal

        valid_signal = signal[valid_mask]
        if len(valid_signal) < min_length:
            return signal

        # Design filter
        b, a = butter(self.filter_order, self.filter_cutoff, btype='low')

        # Apply filter
        try:
            filtered_valid = filtfilt(b, a, valid_signal)

            # Reconstruct full array
            filtered_signal = signal.copy()
            filtered_signal[valid_mask] = filtered_valid

            return filtered_signal
        except:
            # If filtering fails, return original signal
            return signal

    def next(self):
        ha_green = self.ha_close_scaled[-1] > self.ha_open_scaled[-1]
        ha_red = self.ha_close_scaled[-1] < self.ha_open_scaled[-1]

        rsi_rising = self.stoch_rsi[-1] > self.stoch_rsi[-2]
        rsi_falling = self.stoch_rsi[-1] < self.stoch_rsi[-2]

        if len(self.stoch_rsi) >= 3:
            rsi_low = min([self.stoch_rsi[-3], self.stoch_rsi[-2], self.stoch_rsi[-1]]) < self.rsi_oversold
        else:
            return  # Not enough data yet to proceed
        rsi_high = self.stoch_rsi[-1] > self.rsi_overbought

        long_condition = ha_green and rsi_rising
        exit_condition =  ha_red and self.ha_close_scaled[-1] > rsi_high

        if self.data.Close[-1] < self.entry_price * (1-self.sl):
            exit_condition = True

        if long_condition and not self.position:
            self.entry_price = self.data.Close[-1]
            self.buy(size=0.5)
        elif exit_condition and self.position:
            self.position.close()

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
    symbol = 'RAY_USDT'
    symbol = 'RAY-USD'
    # symbol = 'XRP-USD'
    # symbol = 'SOXL'
    end_date = datetime.now()
    start_date = end_date - timedelta(days=59)  # 59 days of data
    
    print(f"Fetching data for {symbol}...")
    # Fetch data with retry mechanism
    # data = fetch_data(symbol, start_date, end_date, "30m").iloc[:-60]
    data = fetch_data(symbol, start_date, end_date, "30m")
    # data = fetch_data_from_pionex(symbol, "30M")
    # data = fetch_from_file()
    
    if data is not None and not data.empty:
        print(f"Successfully fetched {len(data)} data points")
        # Run backtest
        bt = Backtest(data, HARSIStrategy, cash=100000, commission=.001, finalize_trades=True)
        stats = bt.run()
        print(stats)
        
        
        # data.to_csv('data.csv')

        # Export trades to CSV
        trades = stats['_trades']
        print(trades.iloc[:, [9, 10]])
        # csv_filename = f'trades1.csv'
        # trades.to_csv(csv_filename)
        # print(f"\nExported trades to {csv_filename}")
        
        # bt.plot()
        bt.plot(filename='backtest_report.html', open_browser=True)

    else:
        print("Failed to fetch data. Please try again later.")
