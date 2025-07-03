import time
import requests
import urllib.parse
import hashlib
import hmac
import base64
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
from scipy.signal import butter, filtfilt
import logging
import pytz

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('kraken_trading.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Global trading configuration
TRADING_PAIR = "TQQQ/USD"  # Trading pair on Kraken
BUY_PERCENTAGE = 0.3      # Percentage of available USD to use for buying

with open("kraken/keys.txt", "r") as file:
    lines = file.read().splitlines()
    API_KEY = lines[0]
    PRIVATE_KEY = lines[1]


def get_kraken_signature(urlpath, data, secret):
    """Generate Kraken API signature"""
    postdata = urllib.parse.urlencode(data)
    encoded = (str(data['nonce']) + postdata).encode()
    message = urlpath.encode() + hashlib.sha256(encoded).digest()
    mac = hmac.new(base64.b64decode(secret), message, hashlib.sha512)
    sigdigest = base64.b64encode(mac.digest())
    return sigdigest.decode()


def kraken_request(uri_path, data, api_key, api_sec):
    """Make authenticated request to Kraken API"""
    url = f"https://api.kraken.com{uri_path}"
    data['nonce'] = str(int(time.time() * 1000))
    headers = {
        'API-Key': api_key,
        'API-Sign': get_kraken_signature(uri_path, data, api_sec)
    }
    response = requests.post(url, headers=headers, data=data)
    return response.json()


def get_balance():
    """Get account balance"""
    data = {
        'type': 'all'
    }
    response = kraken_request('/0/private/Balance', data, API_KEY, PRIVATE_KEY)
    if response.get('error'):
        logger.error(f"Error getting balance: {response['error']}")
        return None
    return response['result']

def get_current_price(pair):
    """Get current price for a trading pair"""
    url = "https://api.kraken.com/0/public/Ticker"
    params = {'pair': pair}
    
    try:
        response = requests.get(url, params=params)
        data = response.json()
        
        if data.get('error'):
            logger.error(f"Error getting price: {data['error']}")
            return None
            
        ticker_data = data['result'][pair]
        current_price = float(ticker_data['c'][0])  # Last trade price
        return current_price
        
    except Exception as e:
        logger.error(f"Error fetching current price: {e}")
        return None

def place_order(pair, order_type, amount):
    """
    Place a market order on Kraken
    
    Args:
        pair (str): Trading pair (e.g., 'XXRPZUSD')
        order_type (str): 'buy' or 'sell'
        amount (float): Amount to trade (in USD for buy, in crypto for sell)
    
    Returns:
        dict: Order response from Kraken
    """
    if order_type == 'buy':
        current_price = get_current_price(pair)
        if not current_price:
            logger.error("Could not get current price for volume calculation")
            return None

        volume = amount / current_price
        logger.info(f"Calculated volume: {volume:.6f} {TRADING_PAIR.split('Z')[0]} for ${amount:.2f} USD at price ${current_price:.2f}")
    elif order_type == 'sell':
        volume = amount
    else:
        logger.error(f"Invalid order type: {order_type}. Must be 'buy' or 'sell'")
        return None
        
    data = {
        'pair': pair,
        'type': order_type,
        'ordertype': 'market',
        'volume': str(volume)
    }
    
    response = kraken_request('/0/private/AddOrder', data, API_KEY, PRIVATE_KEY)
    if response.get('error'):
        logger.error(f"Error placing {order_type} order: {response['error']}")
        return None
    return response['result']


def buy():
    balance = get_balance()
    if not balance:
        return None
    
    usd_balance = float(balance.get('ZUSD', 0))
    if usd_balance <= 0:
        logger.warning("No USD balance available")
        return None
    
    # amount_to_buy = usd_balance * BUY_PERCENTAGE
    amount_to_buy = 20
    logger.info(f"Buying {TRADING_PAIR.split('Z')[0]} with ${amount_to_buy:.2f} USD")
    return place_order(TRADING_PAIR, 'buy', amount_to_buy)


def sell():
    balance = get_balance()
    if not balance:
        return None
    
    balance = float(balance.get(TRADING_PAIR.split('Z')[0], 0))
    if balance <= 0:
        logger.warning("No balance available")
        return None
    
    logger.info(f"Selling {balance:.2f} {TRADING_PAIR.split('Z')[0]}")
    return place_order(TRADING_PAIR, 'sell', balance)

class HARSIStrategy(Strategy):
    # Strategy parameters
    bottom = -30
    middle_low = -20
    middle_high = 20
    top = 30
    length_rsi = 14
    length_stoch = 14
    smooth_k = 3
    smooth_d = 3
    max_ha_cross = 10
    window = 100
    ha_smooth_period = 1  # New parameter for Heikin Ashi smoothing

    use_butterworth_filter = True
    filter_order = 2
    filter_cutoff = 0.1  # Normalized frequency (0.1 = remove fastest 90% of frequencies)

    def init(self):
        self.last_action = None

        # Calculate Heikin Ashi values
        self.ha_close = (self.data.Open + self.data.High + self.data.Low + self.data.Close) / 4
        self.ha_open = self.I(lambda x: x, self.data.Open)  # Initialize with regular open
        self.ha_high = self.I(lambda x: x, self.data.High)  # Initialize with regular high
        self.ha_low = self.I(lambda x: x, self.data.Low)    # Initialize with regular low
        
        # Calculate HA values using indicators
        def calc_ha_open(ha_open, ha_close):
            ha_open[0] = self.data.Open[0]
            for i in range(1, len(ha_open)):
                ha_open[i] = (ha_open[i-1] + ha_close[i-1]) / 2
            return ha_open
        
        def calc_ha_high(ha_high, ha_open, ha_close):
            for i in range(len(ha_high)):
                ha_high[i] = max(self.data.High[i], max(ha_open[i], ha_close[i]))
            return ha_high
        
        def calc_ha_low(ha_low, ha_open, ha_close):
            for i in range(len(ha_low)):
                ha_low[i] = min(self.data.Low[i], min(ha_open[i], ha_close[i]))
            return ha_low
        
        self.ha_open = self.I(calc_ha_open, self.ha_open, self.ha_close)
        self.ha_high = self.I(calc_ha_high, self.ha_high, self.ha_open, self.ha_close)
        self.ha_low = self.I(calc_ha_low, self.ha_low, self.ha_open, self.ha_close)
        
        # Apply smoothing to Heikin Ashi values
        def smooth_ha_values(values):
            smoothed = np.zeros_like(values)
            for i in range(len(values)):
                start_idx = max(0, i - self.ha_smooth_period + 1)
                smoothed[i] = np.mean(values[start_idx:i+1])
            return smoothed
        
        self.ha_open = self.I(smooth_ha_values, self.ha_open)
        self.ha_close = self.I(smooth_ha_values, self.ha_close)
        self.ha_high = self.I(smooth_ha_values, self.ha_high)
        self.ha_low = self.I(smooth_ha_values, self.ha_low)
        
        # Calculate scaling factors
        def calc_scaled_values(ha_low, ha_high, ha_open, ha_close):
            min_val = min(ha_low[-self.window:])
            max_val = max(ha_high[-self.window:])
            scale = 1.0 / (max_val - min_val) if max_val != min_val else 1
            
            ha_open_scaled = (ha_open - min_val) * scale
            ha_close_scaled = (ha_close - min_val) * scale
            ha_high_scaled = (ha_high - min_val) * scale
            ha_low_scaled = (ha_low - min_val) * scale
            
            return ha_open_scaled, ha_close_scaled, ha_high_scaled, ha_low_scaled
        
        self.ha_open_scaled, self.ha_close_scaled, self.ha_high_scaled, self.ha_low_scaled = self.I(
            calc_scaled_values, self.ha_low, self.ha_high, self.ha_open, self.ha_close
        )
        
        # Calculate RSI and Stochastic RSI
        # Alternative vectorized version (more efficient)
        def calc_rsi(close, period=14):
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
        # Check if Heikin Ashi candle is green (uptrend) or red (downtrend)
        ha_green = self.ha_close_scaled[-1] > self.ha_open_scaled[-1]
        ha_red = self.ha_close_scaled[-1] < self.ha_open_scaled[-1]
        
        # Check RSI direction
        rsi_rising = self.stoch_rsi[-1] > self.stoch_rsi[-2]
        rsi_falling = self.stoch_rsi[-1] < self.stoch_rsi[-2]
        
        # Entry condition: RSI rising AND Heikin Ashi candle is green
        long_condition = rsi_rising and ha_green
        
        # Exit condition: RSI falling AND Heikin Ashi candle is red
        exit_condition = rsi_falling and ha_red
        
        # Execute trades
        if long_condition and not self.position:
            self.buy()
            self.last_action = "BUY"
            return "BUY"
        elif exit_condition and self.position:
            self.position.close()
            self.last_action = "SELL"
            return "SELL"
        return None

def get_data():
    # Kraken API endpoint for getting OHLC data
    url = "https://api.kraken.com/0/public/OHLC"
    
    params = {
        "pair": TRADING_PAIR,
        "interval": 15  # 15-minute intervals
    }
    
    try:
        # Make the API request
        response = requests.get(url, params=params)
        response.raise_for_status()
        
        # Parse the response
        data = response.json()
        
        if data["error"]:
            logger.error(f"Error: {data['error']}")
            return None
            
        # Extract OHLC data
        ohlc_data = data["result"][TRADING_PAIR]
        
        # Convert to DataFrame
        df = pd.DataFrame(ohlc_data, columns=[
            'timestamp', 'Open', 'High', 'Low', 'Close',
            'vwap', 'Volume', 'count'
        ])
        
        df.drop(columns=['vwap', 'count'], inplace=True)

        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s', utc=True)
        df['timestamp'] = df['timestamp'].dt.tz_convert(None)

        # Convert string values to float
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            df[col] = df[col].astype(float)
            
        # Set timestamp as index
        df.set_index('timestamp', inplace=True)
        
        # Get only the last 24 hours of data
        one_day_ago = datetime.now() - timedelta(days=1)
        df = df[df.index >= one_day_ago]
        
        logger.info(f"{TRADING_PAIR} OHLC Data (15m intervals):")
        logger.info(f"Data points: {len(df)}")
        logger.info(f"Date range: {df.index.min()} to {df.index.max()}")
        logger.info(f"Sample of data:\n{df.head()}")
        
        return df
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Error making request: {e}")
        return None
    except (KeyError, ValueError) as e:
        logger.error(f"Error parsing response: {e}")
        return None


def run_strategy():
    last_action = None
    first_sell_received = False  # Wait for the first sell signal
    while True:
        try:
            # Get current time
            current_time = datetime.now()
            
            # Calculate time until next 15-minute candle open
            minutes_to_next = 15 - (current_time.minute % 15)
            seconds_to_next = minutes_to_next * 60 - current_time.second
            
            if seconds_to_next > 0:
                logger.info(f"Waiting {seconds_to_next} seconds for next candle open...")
                time.sleep(seconds_to_next + 2)
            
            # Fetch latest data
            data = get_data()
            if data is None or data.empty:
                logger.warning("Failed to fetch data, retrying in 5 seconds...")
                time.sleep(5)
                continue
            
            # Run strategy
            bt = Backtest(data, HARSIStrategy, cash=1000, commission=.001, trade_on_close=False)
            stats = bt.run()
            bt.plot(filename='backtest_report.html', open_browser=False)
            
            # Get the last action from the strategy
            strat_action = stats['_strategy'].last_action
            
            # Execute trades if needed
            if strat_action == "SELL" and last_action != "SELL":
                if first_sell_received:
                    logger.info("Sell signal detected")
                    sell()
                    last_action = "SELL"
                else:
                    first_sell_received = True
                    last_action = "SELL"  # Mark sell as the last action to avoid re-triggering
                    logger.info("First sell signal received. The bot is now active and can place buy orders on the next signal.")
            elif strat_action == "BUY" and last_action != "BUY":
                if first_sell_received:
                    logger.info("Buy signal detected")
                    buy()
                    last_action = "BUY"
                else:
                    logger.info("Buy signal detected, but waiting for the first sell signal before buying.")
            
            if not first_sell_received:
                logger.info("Waiting for the first sell signal to activate trading...")

            logger.info(f"Strategy check completed at {current_time}")
            time.sleep(5)
        except Exception as e:
            logger.error(f"Error in main loop: {str(e)}")
            time.sleep(5)  # Wait 5 seconds before retrying


if __name__ == "__main__":
    # run_strategy()
    data = get_data()
    print(data)