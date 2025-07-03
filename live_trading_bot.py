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
    filter_order = 1
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

class LiveTrader:
    def __init__(self):
        self.cookies_file = 'pionex_cookies.pkl'
        self.driver = None
        self.symbol = 'XRP_USDT'
        self.interval = '15M'
        self.last_action = None
        self.last_check_time = None
        self.BUY_PERCENTAGE = 0.5
        
    def setup_driver(self):
        """Initialize the Firefox driver with saved cookies if available"""
        # Create firefox profile directory in the user's home directory
        firefox_profile_dir = os.path.join(os.path.expanduser('~'), 'pionex_firefox_profile')
        os.makedirs(firefox_profile_dir, exist_ok=True)
        
        # Add Firefox options for stability
        options = Options()
        options.profile = firefox_profile_dir
        options.set_preference("browser.download.folderList", 2)
        options.set_preference("browser.download.manager.showWhenStarting", False)
        options.set_preference("browser.download.dir", firefox_profile_dir)
        options.set_preference("browser.helperApps.neverAsk.saveToDisk", "application/x-gzip")
        options.set_preference("browser.privatebrowsing.autostart", False)
        options.set_preference("dom.webdriver.enabled", False)
        options.set_preference("useAutomationExtension", False)
       
        gecko_path = "geckodriver.exe"

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
                import base64
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
                logging.info(f"Login detected.")
                self.driver.get("https://www.pionex.us/en-US/trade/XRP_USD/Manual")
                logging.info("Navigated to trading panel.")

            window_size = self.driver.get_window_size()
            mid_x = window_size['width'] // 2
            mid_y = window_size['height'] // 2

            actions = ActionChains(self.driver)
            for _ in range(3):
                actions.move_by_offset(mid_x, mid_y).click().perform()
                actions.move_by_offset(-mid_x, -mid_y)  # Reset mouse position to avoid offset stacking
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
        
        # Get the latest price from the most recent data
        latest_data = self.fetch_live_data()
        if latest_data is not None and not latest_data.empty:
            current_price = latest_data['Close'].iloc[-1]
            logging.info(f"Current XRP price (from fetched): ${current_price:.4f}")
        else:
            logging.warning("Could not fetch current price")
        
        buy_button = WebDriverWait(self.driver, 10).until(
            EC.element_to_be_clickable((By.XPATH, "//div[@role='tab' and text()='Buy']"))
        )
        buy_button.click()
        market_tab = WebDriverWait(self.driver, 10).until(
            EC.element_to_be_clickable((By.XPATH, "//div[@role='tab' and text()='Market']"))
        )
        market_tab.click()
        
        balance_span = WebDriverWait(self.driver, 10).until(
            EC.visibility_of_element_located((
                By.XPATH, "//span[text()='Available balance:']/following-sibling::span"
            ))
        )

        # Extract and clean balance value
        balance_text = balance_span.text.strip().replace("USD", "").replace(",", "")
        balance_value = float(balance_text)

        # Calculate amount
        amount = balance_value * self.BUY_PERCENTAGE
        # amount = 20
        
        input_box = WebDriverWait(self.driver, 10).until(
            EC.element_to_be_clickable((By.XPATH, "//input[@placeholder[contains(., 'Min amount')]]"))
        )
        input_box.clear()
        input_box.send_keys(str(amount))
        
        buy_button = WebDriverWait(self.driver, 10).until(
            EC.element_to_be_clickable((By.XPATH, "//button[.//span[text()='Buy XRP']]"))
        )
        buy_button.click()
        
        # Check for and handle confirmation dialog
        try:
            confirm_button = WebDriverWait(self.driver, 15).until(
                EC.element_to_be_clickable((By.XPATH, "//button[contains(@class, 'pi-btn-primary')]//span[text()='OK']"))
            )
            confirm_button.click()
            logging.info("Confirmed buy order in dialog")
        except:
            logging.info("No confirmation dialog appeared, pressing Enter.")
            self.driver.find_element(By.TAG_NAME, 'body').send_keys(Keys.RETURN)
            
        logging.info(f"Buy Order Placed: Bought ${amount} of XRP at approximately ${current_price:.4f} per XRP")

    def sell(self):
        logging.info("Placing Sell Order...")
        
        # Get the latest price from the most recent data
        latest_data = self.fetch_live_data()
        if latest_data is not None and not latest_data.empty:
            current_price = latest_data['Close'].iloc[-1]
            logging.info(f"Current XRP price (from fetched): ${current_price:.4f}")
        else:
            logging.warning("Could not fetch current price")
            
        sell_button = WebDriverWait(self.driver, 10).until(
            EC.element_to_be_clickable((By.XPATH, "//div[@role='tab' and text()='Sell']"))
        )
        sell_button.click()
        market_tab = WebDriverWait(self.driver, 10).until(
            EC.element_to_be_clickable((By.XPATH, "//div[@role='tab' and text()='Market']"))
        )
        market_tab.click()
        
        balance_span = WebDriverWait(self.driver, 10).until(
            EC.visibility_of_element_located((
                By.XPATH, "//span[text()='Available balance:']/following-sibling::span[contains(text(), 'XRP')]"
            ))
        )

        balance_text = balance_span.text.strip()
        balance_number = balance_text.replace("XRP", "").replace(",", "").strip()
        balance_value = float(balance_number)
        
        input_box = WebDriverWait(self.driver, 10).until(
            EC.element_to_be_clickable((By.XPATH, "//input[@placeholder[contains(., 'Min amount')]]"))
        )
        input_box.clear()
        input_box.send_keys(str(balance_value))
        
        sell_button = WebDriverWait(self.driver, 10).until(
            EC.element_to_be_clickable((By.XPATH, "//button[.//span[text()='Sell XRP']]"))
        )
        sell_button.click()
        
        # Check for and handle confirmation dialog
        try:
            confirm_button = WebDriverWait(self.driver, 15).until(
                EC.element_to_be_clickable((By.XPATH, "//button[contains(@class, 'pi-btn-primary')]//span[text()='OK']"))
            )
            confirm_button.click()
            logging.info("Confirmed sell order in dialog")
        except:
            logging.info("No confirmation dialog appeared, pressing Enter.")
            self.driver.find_element(By.TAG_NAME, 'body').send_keys(Keys.RETURN)
            
        logging.info(f"Sell Order Placed: Sold {balance_value} XRP at approximately ${current_price:.4f} per XRP")

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

    def run(self):
        logging.info("Starting live trading bot...")
        self.setup_driver()  # Initialize driver with cookies
        first_sell_received = False
        while True:
            try:
                # Get current time
                current_time = datetime.now()
                
                # Calculate time until next 15-minute candle open
                minutes_to_next = 15 - (current_time.minute % 15)
                seconds_to_next = minutes_to_next * 60 - current_time.second
                
                if seconds_to_next > 0:
                    logging.info(f"Waiting {seconds_to_next} seconds for next candle open...")
                    time.sleep(seconds_to_next + 2)
                
                # Fetch latest data
                data = self.fetch_live_data()
                if data is None or data.empty:
                    logging.warning("Failed to fetch data, retrying in 5 seconds...")
                    time.sleep(5)
                    continue
                
                # Run strategy
                bt = Backtest(data, HARSIStrategy, cash=1000, commission=.001, trade_on_close=False)
                stats = bt.run()
                bt.plot(filename='backtest_report.html', open_browser=False)
                
                # Get the last action from the strategy
                strat_action = stats['_strategy'].last_action
                
                # Execute trades if needed
                if strat_action == "SELL" and self.last_action != "SELL":
                    if first_sell_received:
                        logging.info("Sell signal detected")
                        self.sell()
                        self.last_action = "SELL"
                    else:
                        first_sell_received = True
                        self.last_action = "SELL"  # Mark sell as the last action to avoid re-triggering
                        logging.info("First sell signal received. The bot is now active and can place buy orders on the next signal.")
                elif strat_action == "BUY" and self.last_action != "BUY":
                    if first_sell_received:
                        logging.info("Buy signal detected")
                        self.buy()
                        self.last_action = "BUY"
                    else:
                        logging.info("Buy signal detected, but waiting for the first sell signal before buying.")
                
                if not first_sell_received:
                    logging.info("Waiting for the first sell signal to activate trading...")

                logging.info(f"Strategy check completed at {current_time}")
                time.sleep(5)                
            except Exception as e:
                logging.error(f"Error in main loop: {str(e)}")
                time.sleep(5)  # Wait 5 seconds before retrying

    def cleanup(self):
        logging.info("Cleaning up and closing browser...")
        if self.driver:
            self.driver.quit()

if __name__ == "__main__":
    trader = LiveTrader()
    try:
        trader.run()
        data = trader.fetch_live_data()
    except KeyboardInterrupt:
        logging.info("\nShutting down trading bot...")
    finally:
        trader.cleanup() 
