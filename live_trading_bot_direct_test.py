import pandas as pd
import numpy as np
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

class HARSIStrategy:
    def __init__(self):
        # Strategy parameters
        self.length_rsi = 31
        self.length_stoch = 21
        self.smooth_k = 9
        self.smooth_d = 10
        self.window = 250
        self.ha_smooth_period = 50
        self.rsi_oversold = 0.37
        self.rsi_overbought = 0.72
        self.use_butterworth_filter = True
        self.filter_order = 3
        self.filter_cutoff = 0.176
        
        # Position and trade tracking
        self.in_position = False
        self.entry_price = None
        self.entry_time = None
        self.trades = []
        
    def calculate_indicators(self, data):
        """Calculate all indicators for the strategy"""
        df = data.copy()
        
        # Calculate Heikin Ashi values
        ha_close = (df['Open'] + df['High'] + df['Low'] + df['Close']) / 4
        
        # Initialize arrays for HA components
        ha_open = np.zeros(len(df))
        ha_high = np.zeros(len(df))
        ha_low = np.zeros(len(df))
        
        # Calculate Heikin Ashi Open
        ha_open[0] = df['Open'].iloc[0]
        for i in range(1, len(df)):
            ha_open[i] = (ha_open[i-1] + ha_close.iloc[i-1]) / 2
        
        # Calculate High and Low
        for i in range(len(df)):
            ha_high[i] = max(df['High'].iloc[i], ha_open[i], ha_close.iloc[i])
            ha_low[i] = min(df['Low'].iloc[i], ha_open[i], ha_close.iloc[i])
        
        # Apply smoothing
        def smooth(x, period):
            return pd.Series(x).rolling(window=period, center=True).mean().bfill().ffill()
        
        ha_open_smooth = smooth(ha_open, self.ha_smooth_period)
        ha_close_smooth = smooth(ha_close, self.ha_smooth_period)
        ha_high_smooth = smooth(ha_high, self.ha_smooth_period)
        ha_low_smooth = smooth(ha_low, self.ha_smooth_period)
        
        # Scale HA values
        def scale_ha(values, ha_low, ha_high):
            scaled = np.zeros_like(values)
            for i in range(len(values)):
                start = max(0, i - self.window + 1)
                if isinstance(ha_low, pd.Series):
                    local_min = np.min(ha_low.iloc[start:i+1])
                    local_max = np.max(ha_high.iloc[start:i+1])
                    current_val = values.iloc[i] if isinstance(values, pd.Series) else values[i]
                else:
                    local_min = np.min(ha_low[start:i+1])
                    local_max = np.max(ha_high[start:i+1])
                    current_val = values[i]
                scale = 1.0 / (local_max - local_min) if local_max != local_min else 1.0
                scaled[i] = (current_val - local_min) * scale
            return scaled
        
        ha_open_scaled = scale_ha(ha_open_smooth, ha_low_smooth, ha_high_smooth)
        ha_close_scaled = scale_ha(ha_close_smooth, ha_low_smooth, ha_high_smooth)
        
        # Calculate RSI
        def calc_rsi(close, period=31):
            close = np.array(close, dtype=float)
            delta = np.diff(close)
            
            gain = np.where(delta > 0, delta, 0)
            loss = np.where(delta < 0, -delta, 0)
            
            # Use pandas-style exponential weighted mean
            gain_ema = pd.Series(gain).ewm(span=period, adjust=False).mean()
            loss_ema = pd.Series(loss).ewm(span=period, adjust=False).mean()
            
            rs = gain_ema / loss_ema
            rsi_values = 100 - (100 / (1 + rs))
            
            # Pad with NaN for first value and return
            rsi = np.full(len(close), np.nan)
            rsi[1:] = rsi_values
            rsi[:period] = np.nan
            
            return rsi
        
        rsi = calc_rsi(df['Close'].values)
        
        # Calculate Stochastic RSI
        def calc_stoch_rsi(rsi):
            stoch_rsi = np.zeros_like(rsi)
            for i in range(len(rsi)):
                if i < self.length_stoch or np.isnan(rsi[i]):
                    stoch_rsi[i] = np.nan
                else:
                    rsi_window = rsi[max(0, i-self.length_stoch+1):i+1]
                    rsi_window = rsi_window[~np.isnan(rsi_window)]
                    if len(rsi_window) > 0:
                        rsi_min = np.min(rsi_window)
                        rsi_max = np.max(rsi_window)
                        stoch_rsi[i] = (rsi[i] - rsi_min) / (rsi_max - rsi_min) if rsi_max != rsi_min else 0
                    else:
                        stoch_rsi[i] = 0
            return stoch_rsi
        
        stoch_rsi = calc_stoch_rsi(rsi)
        
        # Apply Butterworth filter if enabled
        if self.use_butterworth_filter:
            stoch_rsi = self.butterworth_filter(stoch_rsi)
        
        # Calculate K and D lines
        def calc_k(stoch_rsi):
            k = np.zeros_like(stoch_rsi)
            for i in range(len(stoch_rsi)):
                if i < self.smooth_k:
                    valid_values = stoch_rsi[:i+1]
                    valid_values = valid_values[~np.isnan(valid_values)]
                    k[i] = np.mean(valid_values) if len(valid_values) > 0 else np.nan
                else:
                    valid_values = stoch_rsi[i-self.smooth_k+1:i+1]
                    valid_values = valid_values[~np.isnan(valid_values)]
                    k[i] = np.mean(valid_values) if len(valid_values) > 0 else np.nan
            return k
        
        def calc_d(k):
            d = np.zeros_like(k)
            for i in range(len(k)):
                if i < self.smooth_d:
                    valid_values = k[:i+1]
                    valid_values = valid_values[~np.isnan(valid_values)]
                    d[i] = np.mean(valid_values) if len(valid_values) > 0 else np.nan
                else:
                    valid_values = k[i-self.smooth_d+1:i+1]
                    valid_values = valid_values[~np.isnan(valid_values)]
                    d[i] = np.mean(valid_values) if len(valid_values) > 0 else np.nan
            return d
        
        k = calc_k(stoch_rsi)
        d = calc_d(k)
        
        # Store calculated indicators in dataframe
        df['ha_open_scaled'] = ha_open_scaled
        df['ha_close_scaled'] = ha_close_scaled
        df['rsi'] = rsi
        df['stoch_rsi'] = stoch_rsi
        df['k'] = k
        df['d'] = d
        
        return df
    
    def butterworth_filter(self, signal):
        """Apply Butterworth low-pass filter"""
        min_length = 3 * self.filter_order
        
        if len(signal) < min_length:
            return signal
        
        valid_mask = ~np.isnan(signal)
        if not np.any(valid_mask):
            return signal
        
        valid_signal = signal[valid_mask]
        if len(valid_signal) < min_length:
            return signal
        
        try:
            b, a = butter(self.filter_order, self.filter_cutoff, btype='low')
            filtered_valid = filtfilt(b, a, valid_signal)
            
            filtered_signal = signal.copy()
            filtered_signal[valid_mask] = filtered_valid
            
            return filtered_signal
        except:
            return signal
    
    def check_signals(self, data, current_idx):
        """Check for buy/sell signals at the current index"""
        if current_idx < max(self.length_rsi, self.length_stoch, self.smooth_k, self.smooth_d):
            return None
        
        # Get current values
        ha_close_scaled = data['ha_close_scaled'].iloc[current_idx]
        ha_open_scaled = data['ha_open_scaled'].iloc[current_idx]
        stoch_rsi_current = data['stoch_rsi'].iloc[current_idx]
        
        # Check if we have valid data
        if np.isnan(ha_close_scaled) or np.isnan(ha_open_scaled) or np.isnan(stoch_rsi_current):
            return None
        
        # Get previous values for trend detection
        if current_idx > 0:
            stoch_rsi_prev = data['stoch_rsi'].iloc[current_idx - 1]
            if np.isnan(stoch_rsi_prev):
                return None
        else:
            return None
        
        # Calculate conditions
        ha_green = ha_close_scaled > ha_open_scaled
        ha_red = ha_close_scaled < ha_open_scaled
        rsi_rising = stoch_rsi_current > stoch_rsi_prev
        rsi_falling = stoch_rsi_current < stoch_rsi_prev
        
        # Check for RSI low condition (looking at last 3 values)
        if current_idx >= 2:
            rsi_values = [
                data['stoch_rsi'].iloc[current_idx - 2],
                data['stoch_rsi'].iloc[current_idx - 1],
                data['stoch_rsi'].iloc[current_idx]
            ]
            rsi_values = [x for x in rsi_values if not np.isnan(x)]
            rsi_low = len(rsi_values) > 0 and min(rsi_values) < self.rsi_oversold
        else:
            rsi_low = stoch_rsi_current < self.rsi_oversold
        
        rsi_high = stoch_rsi_current > self.rsi_overbought
        
        # Trading logic
        long_condition = ha_green and rsi_rising and rsi_low
        exit_condition = ha_red and rsi_high
        
        current_time = data.index[current_idx]
        current_price = data['Close'].iloc[current_idx]
        
        # Execute trades
        if long_condition and not self.in_position:
            self.buy(current_time, current_price)
            return "BUY"
        elif exit_condition and self.in_position:
            self.sell(current_time, current_price)
            return "SELL"
        
        return None
    
    def buy(self, timestamp, price):
        """Execute buy order"""
        self.in_position = True
        self.entry_price = price
        self.entry_time = timestamp
        logging.info(f"BUY signal at {timestamp}: Price ${price:.4f}")
    
    def sell(self, timestamp, price):
        """Execute sell order"""
        if self.in_position and self.entry_price is not None:
            pnl = price - self.entry_price
            pnl_pct = (pnl / self.entry_price) * 100
            
            trade = {
                'entry_time': self.entry_time,
                'exit_time': timestamp,
                'entry_price': self.entry_price,
                'exit_price': price,
                'pnl': pnl,
                'pnl_pct': pnl_pct
            }
            self.trades.append(trade)
            
            logging.info(f"SELL signal at {timestamp}: Price ${price:.4f}, PnL: ${pnl:.4f} ({pnl_pct:.2f}%)")
        
        self.in_position = False
        self.entry_price = None
        self.entry_time = None
    
    def get_trade_summary(self):
        """Get summary of all trades"""
        if not self.trades:
            return "No trades executed"
        
        total_trades = len(self.trades)
        winning_trades = len([t for t in self.trades if t['pnl'] > 0])
        total_pnl = sum(t['pnl'] for t in self.trades)
        total_pnl_pct = sum(t['pnl_pct'] for t in self.trades)
        
        return f"""
Trade Summary:
Total Trades: {total_trades}
Winning Trades: {winning_trades}
Win Rate: {winning_trades/total_trades*100:.1f}%
Total PnL: ${total_pnl:.4f}
Total PnL %: {total_pnl_pct:.2f}%
Average PnL per trade: ${total_pnl/total_trades:.4f}
        """.strip()

class LiveTrader:
    def __init__(self, interval='15M', symbol='XRP_USDT'):
        self.driver = None
        self.symbol = symbol
        self.interval = interval
        self.strategy = HARSIStrategy()
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
        options = Options()
        options.set_preference("browser.privatebrowsing.autostart", False)
        options.set_preference("dom.webdriver.enabled", False)
        options.set_preference("useAutomationExtension", False)
        
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
            
            try:
                WebDriverWait(self.driver, 120).until(
                    lambda d: d.current_url != "https://accounts.pionex.us/en/sign"
                )
                logging.info(f"Login detected. Waiting 30s for page load")
                time.sleep(30)
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
                    actions.move_by_offset(-mid_x, -mid_y)
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
            
            df = df[["Open", "High", "Low", "Close", "Volume"]].astype(float)
            return df
            
        except Exception as e:
            print("Error fetching Pionex data:", e)
            return None

    def fetch_data(self, symbol, start_date, end_date):
        max_retries = 5
        delay = 5
        interval = '30m'
        if isinstance(start_date, datetime):
            start_date = start_date.astimezone(pytz.UTC)
        if isinstance(end_date, datetime):
            end_date = end_date.astimezone(pytz.UTC)
        
        for attempt in range(int(max_retries)):
            try:
                ticker = yf.Ticker(symbol)
                data = ticker.history(start=start_date, end=end_date, interval=interval)
                if not data.empty:
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
        end_idx = num_call + 500
        if end_idx < len(self.allcandles):
            return self.allcandles.iloc[:end_idx]
        else:
            return None

    def run(self):
        # symbol = 'RAY-USD'
        symbol = 'XRP-USD'
        # symbol = 'SOXL'
        end_date = datetime.now()
        start_date = end_date - timedelta(days=59)
        
        print(f"Fetching data for {symbol}...")
        self.allcandles = self.fetch_data(symbol, start_date, end_date)
        
        if self.allcandles is None or self.allcandles.empty:
            logging.error("Failed to fetch initial data")
            return
        
        logging.info("Starting live trading bot...")
        count = 0
        
        while True:
            try:
                # Get current window of data
                data = self.fetch_test_data(count)
                if data is None:
                    logging.info("Reached end of test data")
                    break
                
                count += 1
                
                # Calculate indicators for the full dataset up to this point
                data_with_indicators = self.strategy.calculate_indicators(data)
                
                # Check signals on the last complete candle (not the current forming one)
                current_idx = len(data_with_indicators) - 2  # -2 to get the last complete candle
                if current_idx >= 0:
                    signal = self.strategy.check_signals(data_with_indicators, current_idx)
                    
                    if signal:
                        current_time = data_with_indicators.index[current_idx]
                        current_price = data_with_indicators['Close'].iloc[current_idx]
                        logging.info(f"Signal: {signal} at {current_time}, Price: ${current_price:.4f}")
                
                # Small delay to simulate real-time processing
                time.sleep(0.01)
                
            except Exception as e:
                logging.error(f"Error in main loop: {str(e)}")
                break
        
        # Print trade summary
        print("\n" + "="*50)
        print("BACKTEST RESULTS")
        print("="*50)
        print(self.strategy.get_trade_summary())
        
        if self.strategy.trades:
            print("\nDetailed Trade Log:")
            for i, trade in enumerate(self.strategy.trades, 1):
                print(f"Trade {i}: {trade['entry_time']} -> {trade['exit_time']}")
                print(f"  Entry: ${trade['entry_price']:.4f}, Exit: ${trade['exit_price']:.4f}")
                print(f"  PnL: ${trade['pnl']:.4f} ({trade['pnl_pct']:.2f}%)")

    def cleanup(self):
        logging.info("Cleaning up and closing browser...")
        if self.driver:
            self.driver.quit()

if __name__ == "__main__":
    interval = '30M'
    symbol = 'RAY_USDT'
    
    trader = LiveTrader(interval=interval, symbol=symbol)
    try:
        trader.run()
    except KeyboardInterrupt:
        logging.info("\nShutting down trading bot...")
    finally:
        trader.cleanup()