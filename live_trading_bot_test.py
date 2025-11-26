import pandas as pd
import numpy as np
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
import yfinance as yf
from datetime import datetime, timedelta
import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.firefox.service import Service
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.common.action_chains import ActionChains
import logging
import pytz
import os
import pickle
import requests
import base64
from scipy.signal import butter, filtfilt

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading_bot.log'),
        logging.StreamHandler()
    ]
)

class HARSIStrategy(Strategy):
    # Strategy parameters
    length_rsi = 31
    length_stoch = 21
    smooth_k = 9
    smooth_d = 10
    window = 250
    ha_smooth_period = 50  # New parameter for Heikin Ashi smoothing
    rsi_oversold = 0.37
    rsi_overbought = 0.72

    use_butterworth_filter = True
    filter_order = 3
    filter_cutoff = 0.176

    sl = 0.01  # Stop loss percentage

    def init(self):
        # Calculate Heikin Ashi values
        self.entry_price = 0
        # Step 1: Base Heikin Ashi Close
        self.ha_close = (self.data.Open + self.data.High + self.data.Low + self.data.Close) / 4
        self.last_action = None

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
            """Apply smoothing while maintaining original array length"""
            x = np.array(x)
            smoothed = np.zeros_like(x)
            
            # For the first 'period' values, use expanding window
            for i in range(len(x)):
                if i < period:
                    # Use expanding window for early values
                    smoothed[i] = np.mean(x[:i+1])
                else:
                    # Use rolling window for later values
                    smoothed[i] = np.mean(x[i-period+1:i+1])
            
            return smoothed

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
        # long_condition d ha_green 
        exit_condition =  ha_red and self.ha_close_scaled[-1] > rsi_high
        
        if self.data.Close[-1] < self.entry_price * (1-self.sl):
            exit_condition = True
        
        self.action_candle_open = self.data.index[-1]
        if long_condition and not self.position:
            self.entry_price = self.data.Close[-1]
            self.buy()
            self.last_action = "BUY"
            return "BUY"
        elif exit_condition and self.position:
            self.position.close()
            self.last_action = "SELL"
            return "SELL"
        return None

class LiveTrader:
    def __init__(self, interval='15M', symbol='XRP_USDT'):
        self.driver = None
        self.symbol = symbol
        self.interval = interval
        self.last_action = None
        self.last_check_time = None
        self.BUY_PERCENTAGE = 0.5
        
        # NEW: Add position tracking to match backtesting behavior
        self.in_position = False  # Track whether we're currently in a position

        self.allcandles = None

        # Map intervals to minutes for timing calculations
        self.interval_minutes = {
            '1M': 1,
            '5M': 5,
            '15M': 15,
            '30M': 30,
            '1H': 60,
            '4H': 240,
            '1D': 1440
        }

    def setup_driver(self):
        """Initialize the Firefox driver with saved cookies if available"""
        # Create firefox profile directory in the user's home directory

        # Add Firefox options for stability
        options = Options()
        options.set_preference("browser.privatebrowsing.autostart", False)
        options.set_preference("dom.webdriver.enabled", False)
        options.set_preference("useAutomationExtension", False)
        # options.add_argument("--headless")
        # options.add_argument("--disable-gpu")
        # options.add_argument("--window-size=1920,1080")

        # gecko_path = "geckodriver.exe"
        gecko_path = "/usr/bin/geckodriver"

        try:
            self.driver = webdriver.Firefox(service=Service(gecko_path), options=options)
            self.driver.get("https://www.pionex.us/")
            self.driver.maximize_window()

            self.driver.get("https://accounts.pionex.us/en/sign")
            try:
                qr_canvas = WebDriverWait(self.driver, 15).until(
                    EC.presence_of_element_located((By.XPATH, "//div[contains(@class, 'qrCodeBox')]/div/div/canvas"))
                )
                qr_data_url = self.driver.execute_script(
                    "return arguments[0].toDataURL('image/png');", qr_canvas
                )
                qr_base64 = qr_data_url.split(',')[1]
                qr_bytes = base64.b64decode(qr_base64)
                with open("pionex_login_qr.png", "wb") as f:
                    f.write(qr_bytes)
                print("Saved QR code as pionex_login_qr.png")
            except Exception as e:
                print(f"Could not save QR code: {e}")
            # Wait for the URL to change after QR code scan, then navigate to trading panel
            try:
                WebDriverWait(self.driver, 120).until(
                    lambda d: d.current_url != "https://accounts.pionex.us/en/sign"
                )
                logging.info(f"Login detected. Waiting 30s for page load")
                time.sleep(30)
                # Convert symbol format from API format (XRP_USDT) to URL format (XRP_USD)
                url_symbol = self.symbol.replace('_USDT', '_USD')
                self.driver.get(f"https://www.pionex.us/en-US/trade/{url_symbol}/Manual")
                logging.info("Navigated to trading panel.")

                window_size = self.driver.get_window_size()
                mid_x = window_size['width'] // 2
                mid_y = window_size['height'] // 2

                actions = ActionChains(self.driver)
                time.sleep(3)
                for _ in range(4):
                    actions.move_by_offset(mid_x, mid_y).click().perform()
                    actions.move_by_offset(-mid_x, -mid_y)  # Reset mouse position to avoid offset stacking
                time.sleep(3)
            except Exception as e:
                logging.error(f"Timeout or error waiting for login: {e}")
                raise Exception("Login was not detected in time. Please try again.")

        except Exception as e:
            logging.error(f"Failed to initialize Firefox driver: {str(e)}")
            if self.driver:
                try:
                    self.driver.quit()
                except:
                    pass
            raise Exception("Failed to initialize Firefox. Please make sure Firefox and geckodriver are installed.")

    def buy(self):
        logging.info("Placing Buy Order...")
        
        # NEW: Update position tracking
        self.in_position = True

        # Get the latest price from the most recent data
        latest_data = self.fetch_live_data()
        if latest_data is not None and not latest_data.empty:
            current_price = latest_data['Close'].iloc[-1]
            logging.info(f"Current {self.symbol} price (from fetched): ${current_price:.4f}")
        else:
            logging.warning("Could not fetch current price")

        # ... rest of buy logic remains the same ...

    def sell(self):
        logging.info("Placing Sell Order...")
        
        # NEW: Update position tracking
        self.in_position = False

        # Get the latest price from the most recent data
        latest_data = self.fetch_live_data()
        if latest_data is not None and not latest_data.empty:
            current_price = latest_data['Close'].iloc[-1]
            logging.info(f"Current {self.symbol} price (from fetched): ${current_price:.4f}")
        else:
            logging.warning("Could not fetch current price")

        # ... rest of sell logic remains the same ...

    def fetch_live_data(self):
        url = "https://api.pionex.com/api/v1/market/klines"
        params = {
            "symbol": self.symbol,
            "interval": self.interval,
            "limit": 500
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
            df = df.iloc[::-1].reset_index(drop=True)
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

    def fetch_data(self, symbol, start_date, end_date):
        max_retries = 5
        delay = 5  # seconds
        interval = '30m'  # Default interval, can be changed as needed
        if isinstance(start_date, datetime):
            start_date = start_date.astimezone(pytz.UTC)
        if isinstance(end_date, datetime):
            end_date = end_date.astimezone(pytz.UTC)
        
        for attempt in range(int(max_retries)):
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

    def fetch_test_data(self, num_call):
        return self.allcandles.iloc[:num_call+500] if num_call+500 < len(self.allcandles) else -1

    def run(self):
        self.allcandles = self.fetch_live_data()
        symbol = 'RAY-USD'
        # symbol = 'XRP-USD'
        # symbol = 'SOXL'
        end_date = datetime.now()
        start_date = end_date - timedelta(days=59)  # 59 days of data
        
        print(f"Fetching data for {symbol}...")
        # Fetch data with retry mechanism
        self.allcandles = self.fetch_data(symbol, start_date, end_date)
        
        logging.info("Starting live trading bot...")
        count = 0
        # self.setup_driver()  # Initialize driver with cookies
        first_sell_received = False
        
        while True:
            try:
                # Get current time
                current_time = datetime.now()

                # Fetch latest data
                data = self.fetch_test_data(count)
                if isinstance(data, int): 
                    return
                count += 1
                data = data.iloc[:-1]
                if data is None or data.empty:
                    logging.warning("Failed to fetch data, retrying in 5 seconds...")
                    time.sleep(5)
                    continue

                # Run strategy
                bt = Backtest(data, HARSIStrategy, cash=1000, commission=.001, trade_on_close=True)
                stats = bt.run()

                # Get the last action from the strategy
                strat_action = stats['_strategy'].last_action
                candle_open_time = stats['_strategy'].action_candle_open
                is_last_candle = stats['_strategy'].action_candle_open == data.index[-1]
                
                # NEW: Get the position state from the backtest
                strategy_in_position = bool(stats['_strategy'].position)
                
                # Execute trades if needed - NOW WITH PROPER POSITION TRACKING
                if is_last_candle and strat_action == "SELL" and strategy_in_position and self.in_position:
                    logging.info(f"Sell signal detected - candle opened at {candle_open_time}")
                    # self.sell()  # Uncomment for actual trading
                    self.in_position = False
                    self.last_action = "SELL"
                        
                elif is_last_candle and strat_action == "BUY" and not strategy_in_position and not self.in_position:
                    logging.info(f"Buy signal detected - candle opened at {candle_open_time}")
                    # self.buy()  # Uncomment for actual trading
                    self.in_position = True
                    self.last_action = "BUY"


            except Exception as e:
                logging.error(f"Error in main loop: {str(e)}")
                time.sleep(5)  # Wait 5 seconds before retrying

    def cleanup(self):
        logging.info("Cleaning up and closing browser...")
        if self.driver:
            self.driver.quit()

if __name__ == "__main__":
    # Change these parameters to switch between different time intervals and trading pairs
    # Available intervals: '1M', '5M', '15M', '30M', '1H', '4H', '1D'
    # Available symbols: 'XRP_USDT', 'BTC_USDT', 'ETH_USDT', 'ADA_USDT', 'DOT_USDT', etc.
    interval = '30M'  # Change this to your desired interval
    symbol = 'RAY_USDT'  # Change this to your desired trading pair

    trader = LiveTrader(interval=interval, symbol=symbol)
    try:
        trader.run()
        # data = trader.fetch_live_data()
    except KeyboardInterrupt:
        logging.info("\nShutting down trading bot...")
    finally:
        trader.cleanup()