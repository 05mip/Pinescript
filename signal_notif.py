import time
import logging
import requests
import pandas as pd
import numpy as np
# from pushbullet import Pushbullet
from scipy.signal import butter, filtfilt
from datetime import datetime, timedelta

# === CONFIGURABLE GLOBALS ===
SYMBOLS = ["XRP_USDT", "RAY_USDT"]
OFFSET = 5  # seconds
PUSHBULLET_API_KEY = "o.IEIo2szuXuYv8yVRwUkJRHQP5QtHnloG"
INTERVAL = "30M"
INTERVAL_MINUTES = 30

# Configure logging to file
logging.basicConfig(
    filename='sender.log',
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Init Pushbullet
# pb = Pushbullet(PUSHBULLET_API_KEY)

class HeikinAshiRSI:
    """HARSI implementation from harsi1.py integrated for signal detection"""
    def __init__(self):
        self.config = {
            # HARSI Candle config
            'harsi_length': 14,
            'open_smoothing': 1,
            
            # RSI Plot config
            'rsi_length': 7,
            'smoothed_mode': True,
            
            # Channel boundaries
            'upper_ob': 20,
            'upper_extreme': 30,
            'lower_os': -20,
            'lower_extreme': -30,
        }
    
    def rsi(self, prices, length=14):
        """Calculate RSI"""
        if isinstance(prices, pd.DataFrame):
            prices = prices.squeeze()
        
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=length).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=length).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def zero_rsi(self, prices, length):
        """Zero median RSI - subtracts 50"""
        rsi_values = self.rsi(prices, length)
        return rsi_values - 50
    
    def smoothed_rsi(self, prices, length):
        """Smoothed RSI similar to HA open calculation"""
        base_rsi = self.zero_rsi(prices, length)
        
        if isinstance(base_rsi, pd.DataFrame):
            base_rsi = base_rsi.squeeze()
            
        smoothed = pd.Series(index=base_rsi.index, dtype=float)
        
        for i in range(len(base_rsi)):
            if pd.isna(base_rsi.iat[i]):
                smoothed.iat[i] = np.nan
                continue
                
            if i == 0:
                smoothed.iat[i] = base_rsi.iat[i]
            else:
                prev_smoothed = smoothed.iat[i-1] 
                if pd.isna(prev_smoothed):
                    smoothed.iat[i] = base_rsi.iat[i]
                else:
                    smoothed.iat[i] = (prev_smoothed + base_rsi.iat[i]) / 2
        
        return smoothed
    
    def mode_selectable_rsi(self, prices, length, smoothed_mode=True):
        """Return either standard or smoothed RSI based on mode"""
        if smoothed_mode:
            return self.smoothed_rsi(prices, length)
        else:
            return self.zero_rsi(prices, length)
    
    def heikin_ashi_rsi(self, ohlc_data, length):
        """Generate RSI Heikin-Ashi OHLC values"""
        high = ohlc_data['High']
        low = ohlc_data['Low']
        close = ohlc_data['Close']
        
        # Get RSI values for each OHLC component
        close_rsi = self.zero_rsi(close, length)
        high_rsi_raw = self.zero_rsi(high, length)
        low_rsi_raw = self.zero_rsi(low, length)
        
        # Ensure all are Series
        if isinstance(close_rsi, pd.DataFrame):
            close_rsi = close_rsi.squeeze()
        if isinstance(high_rsi_raw, pd.DataFrame):
            high_rsi_raw = high_rsi_raw.squeeze()
        if isinstance(low_rsi_raw, pd.DataFrame):
            low_rsi_raw = low_rsi_raw.squeeze()
        
        # Ensure high is highest and low is lowest
        high_rsi = np.maximum(high_rsi_raw, low_rsi_raw)
        low_rsi = np.minimum(high_rsi_raw, low_rsi_raw)
        
        # Initialize series
        ha_open = pd.Series(index=close_rsi.index, dtype=float)
        ha_close = pd.Series(index=close_rsi.index, dtype=float)
        ha_high = pd.Series(index=close_rsi.index, dtype=float)
        ha_low = pd.Series(index=close_rsi.index, dtype=float)
        
        smoothing = self.config['open_smoothing']
        
        for i in range(len(close_rsi)):
            try:
                close_val = close_rsi.iat[i]
                high_val = high_rsi.iat[i] if hasattr(high_rsi, 'iat') else high_rsi[i]
                low_val = low_rsi.iat[i] if hasattr(low_rsi, 'iat') else low_rsi[i]
            except IndexError:
                continue
            
            if pd.isna(close_val) or pd.isna(high_val) or pd.isna(low_val):
                continue
                
            # HA Close calculation
            if i > 0:
                try:
                    prev_close = close_rsi.iat[i-1]
                    open_rsi = prev_close if not pd.isna(prev_close) else close_val
                except IndexError:
                    open_rsi = close_val
            else:
                open_rsi = close_val
                
            ha_close.iat[i] = (open_rsi + high_val + low_val + close_val) / 4
            
            # HA Open calculation with smoothing
            if i < smoothing:
                ha_open.iat[i] = (open_rsi + close_val) / 2
            else:
                try:
                    prev_open = ha_open.iat[i-1]
                    prev_close_ha = ha_close.iat[i-1]
                    if pd.isna(prev_open):
                        ha_open.iat[i] = (open_rsi + close_val) / 2
                    else:
                        ha_open.iat[i] = ((prev_open * smoothing) + prev_close_ha) / (smoothing + 1)
                except IndexError:
                    ha_open.iat[i] = (open_rsi + close_val) / 2
            
            # HA High and Low
            ha_high.iat[i] = max(high_val, ha_open.iat[i], ha_close.iat[i])
            ha_low.iat[i] = min(low_val, ha_open.iat[i], ha_close.iat[i])
        
        return ha_open, ha_high, ha_low, ha_close
    
    def detect_crossovers(self, rsi_line, ha_close):
        """Detect crossovers between RSI line and Heikin-Ashi close with RSI momentum"""
        crossovers = []
        for i in range(2, len(rsi_line)):
            try:
                # Current values
                rsi_curr = rsi_line.iloc[i]
                rsi_prev = rsi_line.iloc[i-1]
                rsi_prev2 = rsi_line.iloc[i-2]
                ha_curr = ha_close.iloc[i]
                ha_prev = ha_close.iloc[i-1]
                
                # Skip if any values are NaN
                if pd.isna(rsi_curr) or pd.isna(rsi_prev) or pd.isna(rsi_prev2) or pd.isna(ha_curr) or pd.isna(ha_prev):
                    continue
                
                # Check if RSI is increasing (momentum condition)
                rsi_increasing = rsi_curr > rsi_prev and rsi_prev > rsi_prev2

                # RSI PEAK DETECTION
                slope1 = rsi_curr - rsi_prev
                slope2 = rsi_prev - rsi_prev2
                
                # Project next slope
                projected_next_slope = slope1 + (slope1 - slope2)
                
                # Local max occurs when slope changes from positive to negative
                current_peak = slope1 > 0 and slope1 < slope2  # momentum decreasing
                next_peak = slope1 > 0 and projected_next_slope < 0  # will turn negative
                
                rsi_peak = current_peak or next_peak
                
                # Check for crossovers
                bullish_cross = (rsi_curr + (rsi_curr - rsi_prev) >= ha_curr and rsi_curr <= ha_curr and rsi_increasing)
                bearish_cross = rsi_peak and rsi_curr - ha_curr < 10 and rsi_curr >= self.config['upper_ob']
                
                if bullish_cross:
                    crossovers.append({
                        'index': i,
                        'type': 'bullish',
                        'rsi_value': rsi_curr,
                        'ha_value': ha_curr,
                        'timestamp': rsi_line.index[i]
                    })
                elif bearish_cross:
                    crossovers.append({
                        'index': i,
                        'type': 'bearish',
                        'rsi_value': rsi_curr,
                        'ha_value': ha_curr,
                        'timestamp': rsi_line.index[i]
                    })
                    
            except (IndexError, AttributeError):
                continue
        
        return crossovers
    
    def calculate_indicator(self, ohlc_data):
        """Calculate all indicator components"""
        # Calculate OHLC4 source
        ohlc4 = (ohlc_data['Open'] + ohlc_data['High'] + ohlc_data['Low'] + ohlc_data['Close']) / 4
        
        # Standard/smoothed RSI for line plot
        rsi_line = self.mode_selectable_rsi(ohlc4, self.config['rsi_length'], self.config['smoothed_mode'])
        
        # Heikin-Ashi RSI candles
        ha_open, ha_high, ha_low, ha_close = self.heikin_ashi_rsi(ohlc_data, self.config['harsi_length'])
        
        # Detect crossovers
        crossovers = self.detect_crossovers(rsi_line, ha_close)
        
        return {
            'rsi_line': rsi_line,
            'ha_open': ha_open,
            'ha_high': ha_high,
            'ha_low': ha_low,
            'ha_close': ha_close,
            'crossovers': crossovers
        }


class SignalGenerator:
    def __init__(self, interval):
        self.interval = interval
        self.harsi = HeikinAshiRSI()

    def fetch_live_data(self, symbol):
        url = "https://api.pionex.com/api/v1/market/klines"
        params = {"symbol": symbol, "interval": self.interval, "limit": 100}

        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()

            klines = data["data"]["klines"]
            df = pd.DataFrame(klines)
            df = df.iloc[::-1].reset_index(drop=True)
            df["timestamp"] = pd.to_datetime(df["time"], unit='ms', utc=True)
            
            # Log info about the most recent candle
            current_candle = df.iloc[-1]
            logging.info(f"Current (potentially unfinished) candle for {symbol}: {current_candle['timestamp']} - Close: {current_candle['close']}")
            
            # Exclude the most recent (current/unfinished) candle for analysis
            df = df.iloc[:-1]  # Remove the last row (most recent candle)
            
            if len(df) == 0:
                logging.warning(f"No completed candles available for {symbol}")
                return None
                
            # Log the most recent completed candle
            last_completed = df.iloc[-1]
            logging.info(f"Most recent completed candle for {symbol}: {last_completed['timestamp']} - Close: {last_completed['close']}")
            
            df.set_index("timestamp", inplace=True)
            df = df.rename(columns={
                "open": "Open", "high": "High", "low": "Low",
                "close": "Close", "volume": "Volume"
            })
            df = df[["Open", "High", "Low", "Close", "Volume"]].astype(float)
            return df

        except Exception as e:
            logging.error(f"Error fetching Pionex data: {e}")
            return None

    def generate_signals(self, symbol):
        df = self.fetch_live_data(symbol)
        if df is None or df.empty:
            return None

        # Check if we have enough data for HARSI calculation
        if len(df) < 30:  # Need enough data for RSI calculation
            logging.warning(f"Insufficient data for HARSI calculation for {symbol}: {len(df)} candles")
            return None

        notes = []

        try:
            # Calculate HARSI indicator
            harsi_results = self.harsi.calculate_indicator(df)
            
            # Check if the most recent completed candle has a bullish signal
            crossovers = harsi_results['crossovers']
            
            if crossovers:
                # Look for bullish crosses
                recent_bullish = [c for c in crossovers if c['type'] == 'bullish']
                
                if recent_bullish:
                    # Get the most recent bullish cross
                    latest_bullish = recent_bullish[-1]
                    
                    # Check if the bullish signal is on the most recent completed candle
                    latest_index = latest_bullish['index']
                    total_periods = len(df)
                    most_recent_candle_index = total_periods - 1
                    
                    # Only alert if the signal is exactly on the most recent completed candle
                    if latest_index == most_recent_candle_index:
                        current_price = df['Close'].iloc[-1]
                        rsi_value = latest_bullish['rsi_value']
                        ha_value = latest_bullish['ha_value']
                        signal_time = latest_bullish['timestamp'].strftime('%Y-%m-%d %H:%M UTC')
                        
                        note = (f"🚀 HARSI Bullish Cross Signal!\n"
                               f"Symbol: {symbol}\n"
                               f"Signal Time: {signal_time}\n"
                               f"Current Price: ${current_price:.4f}\n"
                               f"RSI Value: {rsi_value:.2f}\n"
                               f"HA Close: {ha_value:.2f}\n"
                               f"Signal Strength: {'Strong' if rsi_value < -10 else 'Moderate'}\n"
                               f"Candle Index: {latest_index}/{total_periods-1} (Most Recent)")
                        
                        notes.append(note)
                        logging.info(f"HARSI Bullish cross detected for {symbol} on most recent candle at {signal_time}")
                    else:
                        logging.info(f"HARSI Bullish cross found for {symbol} but not on most recent candle (index {latest_index} vs {most_recent_candle_index})")

            # Log current HARSI values for debugging
            if not harsi_results['rsi_line'].empty and not harsi_results['ha_close'].empty:
                current_rsi = harsi_results['rsi_line'].iloc[-1]
                current_ha = harsi_results['ha_close'].iloc[-1]
                logging.info(f"Current HARSI values for {symbol}: RSI={current_rsi:.2f}, HA_Close={current_ha:.2f}")

        except Exception as e:
            logging.error(f"Error calculating HARSI for {symbol}: {e}")
            return None

        return notes if notes else None

def sleep_until_next_interval(interval_minutes=30, offset=5):
    now = datetime.utcnow()
    # Calculate minutes and seconds until next interval
    next_minute = (now.minute // interval_minutes + 1) * interval_minutes
    if next_minute >= 60:
        next_hour = now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
        next_time = next_hour
    else:
        next_time = now.replace(minute=next_minute, second=0, microsecond=0)
    sleep_seconds = (next_time - now).total_seconds()
    if sleep_seconds < 0:
        sleep_seconds += interval_minutes * 60
    logging.info(f"Sleeping for {sleep_seconds:.2f} seconds until next interval.")
    time.sleep(sleep_seconds + offset)

# === MAIN LOOP ===
def main():
    sg = SignalGenerator(INTERVAL)
    logging.info("HARSI Signal Generator started")
    
    while True:
        try:
            sleep_until_next_interval(INTERVAL_MINUTES, OFFSET)  # Offset to e.g. run at 1:00:05

            for symbol in SYMBOLS:
                signals = sg.generate_signals(symbol)
                if signals:
                    title = f"📈 HARSI Bullish Signal Alert for {symbol}"
                    body = "\n".join(signals)
                    # pb.push_note(title, body)  # Uncomment when ready to send notifications
                    logging.info(f"HARSI bullish signal detected for {symbol}")
                    print(f"SIGNAL DETECTED:\n{title}\n{body}\n" + "="*50)
                else:
                    logging.info(f"No HARSI bullish signal for {symbol}")

        except Exception as e:
            logging.error("Runtime error:", exc_info=e)


if __name__ == "__main__":
    main()