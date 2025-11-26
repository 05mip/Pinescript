import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class HARSIIndicator:
    """HARSI Indicator Calculator for Backtesting"""
    
    def __init__(self, len_harsi=14, smoothing=25, len_rsi=7, smooth_mode=True, 
                 smooth_k=3, smooth_d=3, stoch_len=14, stoch_fit=80):
        self.len_harsi = len_harsi
        self.smoothing = smoothing
        self.len_rsi = len_rsi
        self.smooth_mode = smooth_mode
        self.smooth_k = smooth_k
        self.smooth_d = smooth_d
        self.stoch_len = stoch_len
        self.stoch_fit = stoch_fit
    
    def rsi(self, series, period=14):
        """Calculate RSI"""
        delta = series.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def zero_rsi(self, series, length):
        """Zero median RSI helper function - subtracts 50"""
        return self.rsi(series, length) - 50
    
    def smoothed_rsi(self, series, length):
        """Mode selectable RSI function for standard or smoothed output"""
        zrsi = self.zero_rsi(series, length)
        
        # Smoothing similar to HA open
        smoothed = pd.Series(index=zrsi.index, dtype=float)
        smoothed.iloc[0] = zrsi.iloc[0] if not pd.isna(zrsi.iloc[0]) else 0
        
        for i in range(1, len(zrsi)):
            if pd.isna(smoothed.iloc[i-1]) or pd.isna(zrsi.iloc[i]):
                smoothed.iloc[i] = zrsi.iloc[i] if not pd.isna(zrsi.iloc[i]) else smoothed.iloc[i-1]
            else:
                smoothed.iloc[i] = (smoothed.iloc[i-1] + zrsi.iloc[i]) / 2
        
        return smoothed if self.smooth_mode else zrsi
    
    def rsi_heikin_ashi(self, data):
        """Generate RSI Heikin-Ashi OHLC values"""
        # Calculate RSI for each OHLC component
        close_rsi = self.zero_rsi(data['Close'], self.len_harsi)
        high_rsi_raw = self.zero_rsi(data['High'], self.len_harsi)
        low_rsi_raw = self.zero_rsi(data['Low'], self.len_harsi)
        
        # Ensure high is highest and low is lowest
        high_rsi = np.maximum(high_rsi_raw, low_rsi_raw)
        low_rsi = np.minimum(high_rsi_raw, low_rsi_raw)
        
        # Calculate HA Close
        open_rsi = close_rsi.shift(1).fillna(close_rsi)
        ha_close = (open_rsi + high_rsi + low_rsi + close_rsi) / 4
        
        # Calculate HA Open with smoothing
        ha_open = pd.Series(index=close_rsi.index, dtype=float)
        
        for i in range(len(close_rsi)):
            if i == 0 or pd.isna(ha_open.iloc[max(0, i-self.smoothing)]) or pd.isna(close_rsi.iloc[i]):
                if not pd.isna(open_rsi.iloc[i]) and not pd.isna(close_rsi.iloc[i]):
                    ha_open.iloc[i] = (open_rsi.iloc[i] + close_rsi.iloc[i]) / 2
                else:
                    ha_open.iloc[i] = ha_open.iloc[i-1] if i > 0 else 0
            else:
                prev_open = ha_open.iloc[i-1]
                prev_close = ha_close.iloc[i-1]
                if not pd.isna(prev_open) and not pd.isna(prev_close):
                    ha_open.iloc[i] = ((prev_open * self.smoothing) + prev_close) / (self.smoothing + 1)
                else:
                    ha_open.iloc[i] = ha_open.iloc[i-1]
        
        # Calculate HA High and Low
        ha_high = np.maximum(high_rsi, np.maximum(ha_open, ha_close))
        ha_low = np.minimum(low_rsi, np.minimum(ha_open, ha_close))
        
        return pd.DataFrame({
            'HA_Open': ha_open,
            'HA_High': ha_high,
            'HA_Low': ha_low,
            'HA_Close': ha_close
        })
    
    def calculate(self, data):
        """Calculate all HARSI indicator values"""
        # Calculate OHLC4 source
        ohlc4 = (data['Open'] + data['High'] + data['Low'] + data['Close']) / 4
        
        # Standard/smoothed RSI for plot
        rsi_plot = self.smoothed_rsi(ohlc4, self.len_rsi)
        
        # Heikin Ashi RSI candles
        ha_rsi = self.rsi_heikin_ashi(data)
        
        # Combine all indicators
        result = data.copy()
        result['RSI'] = rsi_plot
        result['HA_Open'] = ha_rsi['HA_Open']
        result['HA_High'] = ha_rsi['HA_High']
        result['HA_Low'] = ha_rsi['HA_Low']
        result['HA_Close'] = ha_rsi['HA_Close']
        
        # Add HARSI candle color information
        result['HA_Bullish'] = result['HA_Close'] >= result['HA_Open']
        result['HA_Bearish'] = result['HA_Close'] < result['HA_Open']
        
        return result

class HARSIStrategy(Strategy):
    """HARSI Trading Strategy for Backtesting Library"""
    
    # Strategy parameters (can be optimized)
    len_harsi = 14
    smoothing = 25
    len_rsi = 7
    tp_percent = 1.0  # Take profit %
    sl_percent = 1.0  # Stop loss %
    tp_release = 0.3  # Release 30% of position at TP
    tsl = 1.0         # Trailing stop loss %  

    last_tp = 0       # Track TP levels hit for analysis
    
    def init(self):
        """Initialize strategy with indicators"""
        # Create HARSI indicator
        self.harsi_calc = HARSIIndicator(
            len_harsi=self.len_harsi,
            smoothing=self.smoothing,
            len_rsi=self.len_rsi
        )
        
        # Calculate indicators on the full dataset
        df = pd.DataFrame({
            'Open': self.data.Open,
            'High': self.data.High,
            'Low': self.data.Low,
            'Close': self.data.Close
        })
        
        results = self.harsi_calc.calculate(df)
        
        # Store indicators as strategy attributes for plotting
        self.rsi = self.I(lambda: results['RSI'], name='RSI')
        self.ha_open = self.I(lambda: results['HA_Open'], name='HA_Open')
        self.ha_high = self.I(lambda: results['HA_High'], name='HA_High')
        self.ha_low = self.I(lambda: results['HA_Low'], name='HA_Low')
        self.ha_close = self.I(lambda: results['HA_Close'], name='HA_Close')
        self.ha_bullish = self.I(lambda: results['HA_Bullish'], name='HA_Bullish')
        self.ha_bearish = self.I(lambda: results['HA_Bearish'], name='HA_Bearish')
        
        # Track position state
        self.entry_price = None
        self.tp_price = None
        self.sl_price = None
    
    def next(self):
        """Strategy logic executed on each bar"""
        # Skip if not enough data
        if len(self.data) < max(self.len_harsi, self.len_rsi) + self.smoothing:
            return
        
        # Get current values
        current_price = self.data.Close[-1]
        ha_bullish_curr = self.ha_bullish[-1]
        ha_bearish_curr = self.ha_bearish[-1]
        ha_close_curr = self.ha_close[-1]
        rsi_curr = self.rsi[-1]
        
        # Get previous values (avoid index errors)
        if len(self.data) < 2:
            return
            
        ha_bullish_prev = self.ha_bullish[-2]
        ha_bullish_prev2 = self.ha_bullish[-3]
        ha_bearish_prev = self.ha_bearish[-2]
        ha_bearish_prev2 = self.ha_bearish[-3]
        ha_close_prev = self.ha_close[-2]
        rsi_prev = self.rsi[-2]
        rsi_increasing = rsi_curr > rsi_prev
        rsi_decreasing = rsi_curr < rsi_prev
        
        # Check for NaN values
        if (pd.isna(ha_bullish_curr) or pd.isna(ha_bearish_curr) or 
            pd.isna(ha_close_curr) or pd.isna(rsi_curr) or
            pd.isna(ha_bullish_prev) or pd.isna(ha_bearish_prev) or
            pd.isna(ha_close_prev) or pd.isna(rsi_prev)):
            return
        
        # Position management - check TP/SL first
        gain_pct = (current_price - self.entry_price) / self.entry_price * 100 if self.entry_price else 0

        if self.position:
            # if gain_pct > self.last_tp + 1.0:  # If gain exceeds last TP level
            #     self.sell(size=self.tp_release)
            #     self.last_tp += 1.0
            #     return
            # elif gain_pct < self.last_tp - self.sl_percent:  # Stop loss hit
            #     self.position.close()
            #     self.entry_price = None
            #     self.last_tp = 0
            #     return
            if gain_pct >= self.tp_percent and self.last_tp == 0:  # Take profit hit
                self.sell(size=self.tp_release)
                self.last_tp = 1
        
        buy_condition = ha_bullish_curr and ha_bearish_prev
        sell_condition = ha_bearish_curr and ha_bullish_prev
        # Entry signals
        if not self.position:
            if buy_condition:
                self.buy()
                self.entry_price = current_price
        
        else:  # We have a position
            if sell_condition:
                self.position.close()
                self.entry_price = None
                self.last_tp = 0

def fetch_data_for_backtest(symbol, days=30, timeframe='1h'):
    """Fetch data for backtesting"""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    ticker = yf.Ticker(symbol)
    data = ticker.history(start=start_date, end=end_date, interval=timeframe)
    
    # Clean data
    data = data.dropna()
    
    return data

def run_harsi_backtest(symbol='RAY-USD', days=30, timeframe='1h', cash=10000, commission=0.002):
    """
    Run HARSI strategy backtest with dual candle plotting
    
    Parameters:
    symbol: Trading symbol
    days: Number of days of historical data
    timeframe: Data timeframe ('15m', '30m', '1h', etc.)
    cash: Starting cash
    commission: Trading commission (0.002 = 0.2%)
    """
    
    # Fetch data
    print(f"Fetching {symbol} data for backtesting...")
    data = fetch_data_for_backtest(symbol, days, timeframe)
    
    if data.empty:
        print("No data available for backtesting")
        return None
    
    print(f"Data loaded: {len(data)} bars from {data.index[0]} to {data.index[-1]}")
    
    # Initialize backtest
    bt = Backtest(
        data, 
        HARSIStrategy, 
        cash=cash, 
        commission=commission,
        exclusive_orders=True
    )
    
    # Run backtest
    print("Running backtest...")
    results = bt.run()
    
    # Print results
    print("\n" + "="*50)
    print("HARSI STRATEGY BACKTEST RESULTS")
    print("="*50)
    print(f"Start Date: {data.index[0]}")
    print(f"End Date: {data.index[-1]}")
    print(f"Duration: {days} days")
    print(f"Starting Cash: ${cash:,.2f}")
    print(f"Final Portfolio Value: ${results['Equity Final [$]']:,.2f}")
    print(f"Total Return: {results['Return [%]']:.2f}%")
    print(f"Buy & Hold Return: {results['Buy & Hold Return [%]']:.2f}%")
    print(f"Max Drawdown: {results['Max. Drawdown [%]']:.2f}%")
    print(f"Sharpe Ratio: {results['Sharpe Ratio']:.2f}")
    print(f"Total Trades: {results['# Trades']}")
    print(f"Win Rate: {results['Win Rate [%]']:.1f}%")
    print(f"Avg Trade Duration: {results['Avg. Trade Duration']}")
    
    # Plot results with custom plotting
    print("\nGenerating plots...")
    
    # Create custom plot with both price and HARSI candles
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12), height_ratios=[2, 1])
    
    # Plot 1: Price candles with trades
    ax1.set_title(f'{symbol} Price Chart with HARSI Strategy Signals', fontsize=14, fontweight='bold')
    ax1.plot(data.index, data['Close'], color='blue', linewidth=1, alpha=0.7, label='Price')
    
    # Add buy/sell signals from backtest results if available
    trades_df = results._trades if hasattr(results, '_trades') else pd.DataFrame()
    
    if not trades_df.empty and len(trades_df) > 0:
        # Entry points
        entry_dates = trades_df['EntryTime'] if 'EntryTime' in trades_df.columns else trades_df.index
        entry_prices = trades_df['EntryPrice'] if 'EntryPrice' in trades_df.columns else []
        
        # Exit points  
        exit_dates = trades_df['ExitTime'] if 'ExitTime' in trades_df.columns else []
        exit_prices = trades_df['ExitPrice'] if 'ExitPrice' in trades_df.columns else []
        
        if len(entry_prices) > 0:
            ax1.scatter(entry_dates, entry_prices, color='green', marker='^', s=100, 
                       label='Buy Signal', zorder=5)
        
        if len(exit_prices) > 0:
            ax1.scatter(exit_dates, exit_prices, color='red', marker='v', s=100, 
                       label='Sell Signal', zorder=5)
    
    ax1.set_ylabel('Price ($)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: HARSI candles
    # Calculate HARSI for plotting
    harsi_calc = HARSIIndicator()
    harsi_data = harsi_calc.calculate(data)
    
    ax2.set_title('HARSI Indicator', fontsize=14, fontweight='bold')
    
    # Plot RSI line
    ax2.plot(data.index, harsi_data['RSI'], color='blue', linewidth=1, alpha=0.7, label='RSI')
    
    # Plot HARSI candles as lines (simplified for visibility)
    bullish_mask = harsi_data['HA_Bullish']
    bearish_mask = harsi_data['HA_Bearish']
    
    # Plot bullish candles in green
    ax2.plot(data.index[bullish_mask], harsi_data['HA_Close'][bullish_mask], 
             color='green', linewidth=2, alpha=0.8, label='HA Bullish')
    
    # Plot bearish candles in red
    ax2.plot(data.index[bearish_mask], harsi_data['HA_Close'][bearish_mask], 
             color='red', linewidth=2, alpha=0.8, label='HA Bearish')
    
    # Add horizontal reference lines
    ax2.axhline(y=20, color='red', alpha=0.3, linestyle='--')
    ax2.axhline(y=0, color='orange', alpha=0.5, linestyle=':')
    ax2.axhline(y=-20, color='green', alpha=0.3, linestyle='--')
    
    ax2.set_ylabel('RSI Value')
    ax2.set_xlabel('Date')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Show standard backtesting plot as well
    bt.plot(filename=None)
    
    return results, bt

def optimize_harsi_strategy(symbol='RAY-USD', days=30):
    """
    Optimize HARSI strategy parameters
    """
    print(f"Optimizing HARSI strategy for {symbol}...")
    
    data = fetch_data_for_backtest(symbol, days)
    if data.empty:
        return None
    
    bt = Backtest(data, HARSIStrategy, cash=10000, commission=0.002)
    
    # Optimize key parameters
    optimization_results = bt.optimize(
        len_harsi=range(10, 21, 2),          # RSI period
        smoothing=range(15, 35, 5),          # HA smoothing
        len_rsi=range(5, 12, 2),             # RSI period for signals
        tp_percent=[0.5, 1.0, 1.5, 2.0],    # Take profit %
        sl_percent=[0.5, 1.0, 1.5, 2.0],    # Stop loss %
        maximize='Return [%]',
        constraint=lambda p: p.tp_percent >= p.sl_percent * 0.5  # TP should be reasonable vs SL
    )
    
    print("\nOptimization Results:")
    print(f"Best Return: {optimization_results['Return [%]']:.2f}%")
    print(f"Best Parameters:")
    print(f"  len_harsi: {optimization_results['_strategy'].len_harsi}")
    print(f"  smoothing: {optimization_results['_strategy'].smoothing}")
    print(f"  len_rsi: {optimization_results['_strategy'].len_rsi}")
    print(f"  tp_percent: {optimization_results['_strategy'].tp_percent}")
    print(f"  sl_percent: {optimization_results['_strategy'].sl_percent}")
    
    return optimization_results

# Example usage
if __name__ == "__main__":
    # Run basic backtest
    results, backtest_obj = run_harsi_backtest(
        symbol='RAY-USD', 
        days=30, 
        timeframe='1h',
        cash=10000,
        commission=0.002
    )
    
    # Uncomment to run optimization (takes longer)
    # optimization_results = optimize_harsi_strategy('RAY-USD', days=30)