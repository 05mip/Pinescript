import pandas as pd
import numpy as np
from backtesting import Backtest, Strategy
import yfinance as yf
from datetime import datetime, timedelta
import pytz

class GeneticHARSIStrategy(Strategy):
    # Best parameters from genetic optimization
    length_rsi = 10
    length_stoch = 12
    smooth_k = 3
    smooth_d = 3
    ha_smooth_period = 1
    k_buy_threshold = -5
    k_sell_threshold = -27

    def init(self):
        # Calculate Heikin Ashi values
        self.ha_close = (self.data.Open + self.data.High + self.data.Low + self.data.Close) / 4
        self.ha_open = self.I(lambda x: x, self.data.Open)
        
        def calc_ha_open(ha_open, ha_close):
            ha_open[0] = self.data.Open[0]
            for i in range(1, len(ha_open)):
                ha_open[i] = (ha_open[i-1] + ha_close[i-1]) / 2
            return ha_open
        self.ha_open = self.I(calc_ha_open, self.ha_open, self.ha_close)
        
        # Apply smoothing to Heikin Ashi values
        def smooth_ha_values(values):
            smoothed = np.zeros_like(values)
            for i in range(len(values)):
                start_idx = max(0, i - self.ha_smooth_period + 1)
                smoothed[i] = np.mean(values[start_idx:i+1])
            return smoothed
        self.ha_open = self.I(smooth_ha_values, self.ha_open)
        self.ha_close = self.I(smooth_ha_values, self.ha_close)

        # Calculate RSI and Stochastic RSI
        def calc_rsi(close, period):
            close = np.array(close, dtype=float)
            delta = np.diff(close)
            gain = np.where(delta > 0, delta, 0)
            loss = np.where(delta < 0, -delta, 0)
            gain_ema = pd.Series(gain).ewm(span=period, adjust=False).mean()
            loss_ema = pd.Series(loss).ewm(span=period, adjust=False).mean()
            rs = gain_ema / loss_ema
            rsi_values = 100 - (100 / (1 + rs))
            rsi = np.full(len(close), np.nan)
            rsi[1:] = rsi_values
            rsi[:period] = np.nan
            return rsi
        rsi = self.I(calc_rsi, self.data.Close, self.length_rsi)

        def calc_stoch_rsi(rsi, length_stoch):
            stoch_rsi = np.zeros_like(rsi)
            for i in range(len(rsi)):
                if i < length_stoch or np.isnan(rsi[i]):
                    stoch_rsi[i] = 0
                else:
                    rsi_min = np.nanmin(rsi[i-length_stoch+1:i+1])
                    rsi_max = np.nanmax(rsi[i-length_stoch+1:i+1])
                    stoch_rsi[i] = (rsi[i] - rsi_min) / (rsi_max - rsi_min) if rsi_max != rsi_min else 0
            return stoch_rsi
        self.stoch_rsi = self.I(calc_stoch_rsi, rsi, self.length_stoch)

        def smooth(values, period):
            smoothed = np.zeros_like(values)
            for i in range(len(values)):
                start_idx = max(0, i - period + 1)
                smoothed[i] = np.mean(values[start_idx:i+1])
            return smoothed
        self.k = self.I(smooth, self.stoch_rsi, self.smooth_k)
        self.d = self.I(smooth, self.k, self.smooth_d)
        self.k_scaled = self.I(lambda k: k * 80 - 40, self.k)
        self.d_scaled = self.I(lambda d: d * 80 - 40, self.d)

    def next(self):
        ha_green = self.ha_close[-1] > self.ha_open[-1]
        ha_red = self.ha_close[-1] < self.ha_open[-1]
        k_rising = self.k_scaled[-1] > self.k_scaled[-2]
        k_falling = self.k_scaled[-1] < self.k_scaled[-2]
        # Buy
        if not self.position and self.k_scaled[-1] > self.k_buy_threshold and ha_green and k_rising:
            self.buy(size=0.7)
        # Sell
        elif self.position and (self.k_scaled[-1] < self.k_sell_threshold and ha_red and k_falling):
            self.position.close()

# --- Test block ---
if __name__ == '__main__':
    symbol = 'XRP-USD'
    end_date = datetime.now()
    start_date = end_date - timedelta(days=59)
    print(f"Fetching data for {symbol}...")
    data = yf.Ticker(symbol).history(start=start_date, end=end_date, interval='15m')
    if not data.empty:
        data.index = data.index.tz_convert('America/Los_Angeles')
        print(f"Successfully fetched {len(data)} data points")
        bt = Backtest(data, GeneticHARSIStrategy, cash=1000, commission=.001)
        stats = bt.run()
        print(stats)
        bt.plot(filename='ai2/genetic_backtest_report.html', open_browser=True)
    else:
        print("Failed to fetch data. Please try again later.") 