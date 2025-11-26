import pandas as pd
import numpy as np
from backtesting import Strategy, Backtest
from backtesting.lib import crossover
from datetime import datetime, timedelta
import pytz
import time
import yfinance as yf
from ai1.mc_test import mc_optimize, print_optimization_summary
from ai1.mc_test_10 import mc_optimize_composite, print_composite_summary
from scipy.fft import fft, ifft, fftfreq
from scipy.signal import butter, filtfilt
from scipy import stats
import pandas_ta as ta
import warnings
# Method 3: Suppress all FutureWarnings (most broad - use cautiously)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="backtesting.backtesting")

class RSIMACDStrategy(Strategy):
    """
    RSI-based trading strategy with MACD confirmation and multiple entry/exit conditions.
    Designed to match TradingView's ta.rsi() behavior.
    """
    
    # Strategy Parameters
    rsi_length = 7
    rsi_overbought = 58
    rsi_oversold = 40
    rsi_smoothing = 4  # For RSI smoothing (1 = no smoothing)
    
    macd_fast = 9
    macd_slow = 24
    macd_signal = 11
    
    buy_condition = 3   # Which buy condition to use (1-8)
    sell_condition = 3  # Which sell condition to use (1-9, where 9 is TP/SL)
    
    # TP/SL Parameters (as percentages)
    take_profit_pct = 3.5
    stop_loss_pct = 1.01
    
    def init(self):
        close = pd.Series(self.data.Close)
        
        rsi_series = ta.rsi(close, length=self.rsi_length)
        if rsi_series is None or rsi_series.isnull().all():
            rsi_series = pd.Series([50] * len(close))
        else:
            rsi_series = rsi_series.fillna(method='bfill').fillna(50)
        self.rsi = self.I(lambda x: rsi_series.values, self.data.Close)


        # If smoothing > 1, smooth RSI with simple moving average
        if self.rsi_smoothing > 1:
            rsi_smooth = ta.sma(pd.Series(rsi_series), length=self.rsi_smoothing).bfill().ffill().values
            self.rsi = self.I(lambda: rsi_smooth)

        # MACD calculation
        macd_df = ta.macd(close, fast=self.macd_fast, slow=self.macd_slow, signal=self.macd_signal)

        macd_line = macd_df[f"MACD_{self.macd_fast}_{self.macd_slow}_{self.macd_signal}"].bfill().ffill().values
        macd_signal = macd_df[f"MACDs_{self.macd_fast}_{self.macd_slow}_{self.macd_signal}"].bfill().ffill().values
        macd_hist = macd_df[f"MACDh_{self.macd_fast}_{self.macd_slow}_{self.macd_signal}"].bfill().ffill().values

        self.macd_line = self.I(lambda: macd_line)
        self.macd_signal = self.I(lambda: macd_signal)
        self.macd_hist = self.I(lambda: macd_hist)

        self.entry_price = None
    
    def next(self):
        """Execute trading logic directly within next()"""

        if len(self.rsi) < 2 or len(self.macd_hist) < 2:
            return

        current_rsi = self.rsi[-1]
        prev_rsi = self.rsi[-2]
        current_macd_hist = self.macd_hist[-1]
        prev_macd_hist = self.macd_hist[-2]
        current_price = self.data.Close[-1]

        # === BUY LOGIC ===
        buy = False
        if self.buy_condition == 1:
            buy = current_rsi < self.rsi_oversold

        elif self.buy_condition == 2:
            buy = current_rsi < self.rsi_oversold and prev_rsi < self.rsi_oversold

        elif self.buy_condition == 3:
            buy = current_rsi < self.rsi_oversold and prev_rsi < self.rsi_oversold and current_rsi > prev_rsi

        elif self.buy_condition == 4:
            buy = (current_rsi < self.rsi_oversold or prev_rsi < self.rsi_oversold) and current_rsi > prev_rsi

        elif self.buy_condition in [5, 6, 7, 8]:
            macd_improving = current_macd_hist > prev_macd_hist and current_macd_hist < 0
            if macd_improving:
                base = self.buy_condition - 4
                if base == 1:
                    buy = current_rsi < self.rsi_oversold
                elif base == 2:
                    buy = current_rsi < self.rsi_oversold and prev_rsi < self.rsi_oversold
                elif base == 3:
                    buy = current_rsi < self.rsi_oversold and prev_rsi < self.rsi_oversold and current_rsi > prev_rsi
                elif base == 4:
                    buy = (current_rsi < self.rsi_oversold or prev_rsi < self.rsi_oversold) and current_rsi > prev_rsi

        if not self.position and buy:
            self.buy()
            self.entry_price = current_price
            return

        # === SELL LOGIC ===
        sell = False
        if not self.position:
            return

        if self.sell_condition == 9:
            if self.entry_price is not None:
                tp = self.entry_price * (1 + self.take_profit_pct / 100)
                sl = self.entry_price * (1 - self.stop_loss_pct / 100)
                sell = current_price >= tp or current_price <= sl

        elif self.sell_condition == 1:
            sell = current_rsi > self.rsi_overbought

        elif self.sell_condition == 2:
            sell = current_rsi > self.rsi_overbought and prev_rsi > self.rsi_overbought

        elif self.sell_condition == 3:
            sell = current_rsi > self.rsi_overbought and prev_rsi > self.rsi_overbought and current_rsi < prev_rsi

        elif self.sell_condition == 4:
            sell = (current_rsi > self.rsi_overbought or prev_rsi > self.rsi_overbought) and current_rsi < prev_rsi

        elif self.sell_condition in [5, 6, 7, 8]:
            macd_deteriorating = current_macd_hist < prev_macd_hist
            if macd_deteriorating:
                base = self.sell_condition - 4
                if base == 1:
                    sell = current_rsi > self.rsi_overbought
                elif base == 2:
                    sell = current_rsi > self.rsi_overbought and prev_rsi > self.rsi_overbought
                elif base == 3:
                    sell = current_rsi > self.rsi_overbought and prev_rsi > self.rsi_overbought and current_rsi < prev_rsi
                elif base == 4:
                    sell = (current_rsi > self.rsi_overbought or prev_rsi > self.rsi_overbought) and current_rsi < prev_rsi

        if self.position and sell:
            self.position.close()
            self.entry_price = None


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


# Example usage
if __name__ == "__main__":
    strategy = RSIMACDStrategy
    param_space = {
        # RSI Parameters
        'rsi_length': lambda: np.random.randint(5, 25),
        'rsi_overbought': lambda: np.random.randint(55, 80),
        'rsi_oversold': lambda: np.random.randint(20, 45),
        'rsi_smoothing': lambda: np.random.randint(1, 5),  # 1-4 periods smoothing
        
        # MACD Parameters
        'macd_fast': lambda: np.random.randint(5, 20),
        'macd_slow': lambda: np.random.randint(21, 40),
        'macd_signal': lambda: np.random.randint(5, 15),
        
        # Strategy Conditions
        'buy_condition': lambda: np.random.randint(1, 9),   # 1-8
        'sell_condition': lambda: np.random.randint(1, 10), # 1-9 (9 is TP/SL)
        
        # TP/SL Parameters (only used when sell_condition = 9)
        'take_profit_pct': lambda: np.random.uniform(0.5, 5.0),
        'stop_loss_pct': lambda: np.random.uniform(0.5, 3.0),
    }

    if 0:
        results = mc_optimize_composite(
            strategy=RSIMACDStrategy,
            param_space=param_space,
            n_iterations=1000,
            validation_permutations=10,
            n_threads=100,
            output_file='composite_results.json',
            verbose=True
        )
        # print_optimization_summary(results)
        print_composite_summary(results)
    else:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=59)  # 59 days of data
        data = fetch_data("RAY-USD", start_date, end_date, "30m")
        bt = Backtest(data, strategy, cash=100000, commission=.001, finalize_trades=True)
        # stats = bt.run(**results['best_params'])
        stats = bt.run()
        print(stats)
        bt.plot()