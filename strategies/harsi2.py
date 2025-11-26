import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib import dates as mdates
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class HeikinAshiRSIOscillator:
    def __init__(self):
        # Default parameters matching Pine Script
        self.len_harsi = 14
        self.smoothing = 1
        self.len_rsi = 7
        self.smooth_mode = True
        self.smooth_k = 3
        self.smooth_d = 3
        self.stoch_len = 14
        self.stoch_fit = 80
        self.upper = 20
        self.upper_x = 30
        self.lower = -20
        self.lower_x = -30
        # TP/SL parameters
        self.tp_percent = 1.0  # Take profit 1%
        self.sl_percent = 1.0  # Stop loss 1%
    
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
        smoothed.iloc[0] = zrsi.iloc[0]
        
        for i in range(1, len(zrsi)):
            if pd.isna(smoothed.iloc[i-1]):
                smoothed.iloc[i] = zrsi.iloc[i]
            else:
                smoothed.iloc[i] = (smoothed.iloc[i-1] + zrsi.iloc[i]) / 2
        
        return smoothed if self.smooth_mode else zrsi
    
    def stochastic(self, series, length, smooth):
        """Calculate Stochastic oscillator"""
        high_n = series.rolling(window=length).max()
        low_n = series.rolling(window=length).min()
        
        k_percent = 100 * ((series - low_n) / (high_n - low_n))
        k_percent = k_percent.rolling(window=smooth).mean()
        
        # Zero median and scale
        return (k_percent - 50) * (self.stoch_fit / 100)
    
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
            if i == 0 or pd.isna(ha_open.iloc[max(0, i-self.smoothing)]):
                ha_open.iloc[i] = (open_rsi.iloc[i] + close_rsi.iloc[i]) / 2
            else:
                prev_open = ha_open.iloc[i-1]
                prev_close = ha_close.iloc[i-1]
                ha_open.iloc[i] = ((prev_open * self.smoothing) + prev_close) / (self.smoothing + 1)
        
        # Calculate HA High and Low
        ha_high = np.maximum(high_rsi, np.maximum(ha_open, ha_close))
        ha_low = np.minimum(low_rsi, np.minimum(ha_open, ha_close))
        
        return pd.DataFrame({
            'Open': ha_open,
            'High': ha_high,
            'Low': ha_low,
            'Close': ha_close
        })
    
    def calculate_signals_with_tpsl(self, rsi, ha_open, ha_high, ha_low, ha_close, price_data):
        """Calculate buy/sell signals with TP/SL tracking"""
        # Determine HARSI candle colors (bullish/bearish)
        harsi_bullish = ha_close >= ha_open  # Green candles
        harsi_bearish = ha_close < ha_open   # Red candles
        
        # Calculate RSI trend (increasing/decreasing)
        rsi_increasing = rsi > rsi.shift(1)
        rsi_decreasing = rsi < rsi.shift(1)
        
        # Initialize signal arrays
        buy_signals = pd.Series(False, index=rsi.index)
        sell_signals = pd.Series(False, index=rsi.index)
        tp_sl_signals = pd.Series(False, index=rsi.index)
        
        # For tracking TP/SL
        open_trades = []  # List to track multiple open positions
        prev_signal = ""
        
        for i in range(1, len(rsi)):
            if pd.isna(rsi.iloc[i]) or pd.isna(ha_close.iloc[i]):
                continue
                
            # Current values
            rsi_curr = rsi.iloc[i]
            rsi_prev = rsi.iloc[i-1]
            current_price = price_data['Close'].iloc[i]
            ha_high_curr = ha_high.iloc[i]
            ha_low_curr = ha_low.iloc[i]
            ha_close_curr = ha_close.iloc[i]
            ha_open_curr = ha_open.iloc[i]


            # Check TP/SL for existing trades
            trades_to_remove = []
            for trade_idx, trade in enumerate(open_trades):
                # Check take profit
                if current_price >= trade['tp_price']:
                    tp_sl_signals.iloc[i] = True
                    trades_to_remove.append(trade_idx)
                # Check stop loss
                elif current_price <= trade['sl_price']:
                    tp_sl_signals.iloc[i] = True
                    trades_to_remove.append(trade_idx)
            
            # Remove closed trades
            for idx in reversed(trades_to_remove):
                open_trades.pop(idx)
            
            # Original signal logic (unchanged)
            # buy_condition = rsi_increasing.iloc[i] and harsi_bullish.iloc[i] and rsi_prev < ha_close.iloc[i-1] and rsi_curr >= ha_close_curr
            # buy_condition = rsi_increasing.iloc[i] and rsi_prev < ha_close.iloc[i-1] and rsi_curr >= ha_close_curr and (rsi_prev < 0 or rsi_curr < 0)
            buy_condition = harsi_bearish.iloc[i-1] and harsi_bullish.iloc[i]
            # sell_condition = rsi_decreasing.iloc[i] and rsi_prev > ha_close.iloc[i-1] and rsi_curr <= ha_close_curr and ha_close_curr > 0
            sell_condition = rsi_decreasing.iloc[i] and ha_close_curr > 0 and prev_signal == "buy"
            # sell_condition = harsi_bullish.iloc[i-1] and harsi_bearish.iloc[i]
            
            # Buy signal: RSI increasing and crosses above HARSI candle
            if buy_condition:
                buy_signals.iloc[i] = True
                # Add new trade to track TP/SL
                open_trades.append({
                    'entry_price': current_price,
                    'tp_price': current_price * (1 + self.tp_percent / 100),
                    'sl_price': current_price * (1 - self.sl_percent / 100)
                })
                prev_signal = "buy"
            
            # Sell signal: RSI decreasing and crosses below HARSI candle
            if sell_condition:
                sell_signals.iloc[i] = True
                prev_signal = "sell"
        
        return buy_signals, sell_signals, tp_sl_signals
    
    def calculate_signals(self, rsi, ha_open, ha_high, ha_low, ha_close):
        """Calculate buy/sell signals"""
        # Determine HARSI candle colors (bullish/bearish)
        harsi_bullish = ha_close >= ha_open  # Green candles
        harsi_bearish = ha_close < ha_open   # Red candles
        
        # Calculate RSI trend (increasing/decreasing)
        rsi_increasing = rsi > rsi.shift(1)
        rsi_decreasing = rsi < rsi.shift(1)
        
        # Initialize signal arrays
        buy_signals = pd.Series(False, index=rsi.index)
        sell_signals = pd.Series(False, index=rsi.index)
        prev_signal = ""
        
        for i in range(1, len(rsi)):
            if pd.isna(rsi.iloc[i]) or pd.isna(ha_close.iloc[i]):
                continue
                
            # Current values
            rsi_curr = rsi.iloc[i]
            rsi_prev = rsi.iloc[i-1]
            ha_high_curr = ha_high.iloc[i]
            ha_low_curr = ha_low.iloc[i]
            ha_close_curr = ha_close.iloc[i]
            ha_open_curr = ha_open.iloc[i]
            
            # buy_condition = rsi_increasing.iloc[i] and harsi_bullish.iloc[i] and rsi_prev < ha_close.iloc[i-1] and rsi_curr >= ha_close_curr
            # buy_condition = rsi_increasing.iloc[i] and rsi_prev < ha_close.iloc[i-1] and rsi_curr >= ha_close_curr and (rsi_prev < 0 or rsi_curr < 0)
            # buy_condition = ha_close_curr >= ha_open_curr
            buy_condition = True
            # sell_condition = rsi_decreasing.iloc[i] and rsi_prev > ha_close.iloc[i-1] and rsi_curr <= ha_close_curr and ha_close_curr > 0
            # sell_condition = rsi_decreasing.iloc[i] and ha_close_curr > 0 and prev_signal == "buy"
            sell_condition = ha_close_curr < ha_open_curr
            # Buy signal: RSI increasing and crosses above green HARSI candle
            if buy_condition:
                buy_signals.iloc[i] = True
                prev_signal = "buy"
            
            # Sell signal: RSI decreasing and crosses below red HARSI candle
            if sell_condition:
                sell_signals.iloc[i] = True
                prev_signal = "sell"

        return buy_signals, sell_signals
    
    def calculate(self, data):
        """Calculate all indicator values"""
        # Calculate OHLC4 source
        ohlc4 = (data['Open'] + data['High'] + data['Low'] + data['Close']) / 4
        
        # Standard/smoothed RSI for plot
        rsi_plot = self.smoothed_rsi(ohlc4, self.len_rsi)
        
        # Stochastic RSI
        stoch_k = self.stochastic(rsi_plot, self.stoch_len, self.smooth_k)
        stoch_d = stoch_k.rolling(window=self.smooth_d).mean()
        
        # Heikin Ashi RSI candles
        ha_rsi = self.rsi_heikin_ashi(data)
        
        # Calculate buy/sell signals with TP/SL
        buy_signals, sell_signals, tp_sl_signals = self.calculate_signals_with_tpsl(
            stoch_k, ha_rsi['Open'], ha_rsi['High'], ha_rsi['Low'], ha_rsi['Close'], data
        )
        
        return {
            'rsi': rsi_plot,
            'stoch_k': stoch_k,
            'stoch_d': stoch_d,
            'ha_open': ha_rsi['Open'],
            'ha_high': ha_rsi['High'],
            'ha_low': ha_rsi['Low'],
            'ha_close': ha_rsi['Close'],
            'buy_signals': buy_signals,
            'sell_signals': sell_signals,
            'tp_sl_signals': tp_sl_signals
        }

def fetch_data(symbol, days=59):
    """Fetch data using yfinance"""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    ticker = yf.Ticker(symbol)
    data = ticker.history(start=start_date, end=end_date, interval='15m')
    
    return data

def draw_candlestick(ax, idx, open_price, high_price, low_price, close_price, candle_width):
    """Draw a single candlestick"""
    # Determine color
    color = 'green' if close_price >= open_price else 'red'
    
    # Calculate candle body
    body_height = abs(close_price - open_price)
    body_bottom = min(open_price, close_price)
    
    # Draw wick (high-low line)
    ax.plot([idx, idx], [low_price, high_price], color='black', linewidth=1, alpha=0.8)
    
    # Draw body
    if body_height > 0:
        rect = Rectangle((idx - candle_width/2, body_bottom), candle_width, body_height,
                        facecolor=color, edgecolor='black', alpha=0.8, linewidth=0.5)
        ax.add_patch(rect)
    else:
        # Doji - draw a line
        ax.plot([idx - candle_width/2, idx + candle_width/2], [close_price, close_price], 
                color='black', linewidth=2)

def draw_harsi_candlestick(ax, idx, open_rsi, high_rsi, low_rsi, close_rsi, candle_width):
    """Draw a single HARSI candlestick"""
    # Determine color
    color = 'teal' if close_rsi >= open_rsi else 'red'
    
    # Calculate candle body
    body_height = abs(close_rsi - open_rsi)
    body_bottom = min(open_rsi, close_rsi)
    
    # Draw wick
    ax.plot([idx, idx], [low_rsi, high_rsi], color='gray', linewidth=1, alpha=0.7)
    
    # Draw body
    if body_height > 0:
        rect = Rectangle((idx - candle_width/2, body_bottom), candle_width, body_height,
                        facecolor=color, edgecolor=color, alpha=0.8, linewidth=0.5)
        ax.add_patch(rect)
    else:
        # Doji - draw a line
        ax.plot([idx - candle_width/2, idx + candle_width/2], [close_rsi, close_rsi], 
                color=color, linewidth=2)

def plot_harsi_indicator(symbol='AAPL', days=59):
    """Main function to plot price and HARSI indicator"""
    # Fetch data
    print(f"Fetching {symbol} data for last {days} days...")
    data = fetch_data(symbol, days)
    
    if data.empty:
        print("No data found!")
        return
    
    # Initialize indicator
    harsi = HeikinAshiRSIOscillator()
    
    # Calculate indicator values
    print("Calculating HARSI indicator...")
    results = harsi.calculate(data)
    
    # Create plots with shared x-axis
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12), gridspec_kw={'height_ratios': [2, 3]}, sharex=True)
    
    # Calculate candle width based on time interval
    time_diff = (data.index[1] - data.index[0]).total_seconds() / 86400  # Convert to days
    candle_width = timedelta(days=time_diff * 0.8)  # 80% of the interval
    
    # Plot 1: Price Chart with OHLC Candlesticks
    for idx, row in data.iterrows():
        draw_candlestick(ax1, idx, row['Open'], row['High'], row['Low'], row['Close'], candle_width)
    
    ax1.set_title(f'{symbol} Price Chart (15min, Last {days} Days)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Price ($)', fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: HARSI Indicator
    # Background fills
    ax2.axhspan(harsi.upper, harsi.upper_x, alpha=0.1, color='red', label='OB Extreme')
    ax2.axhspan(harsi.upper, harsi.lower, alpha=0.1, color='blue', label='Channel')
    ax2.axhspan(harsi.lower, harsi.lower_x, alpha=0.1, color='green', label='OS Extreme')
    
    # Horizontal lines
    ax2.axhline(y=harsi.upper_x, color='silver', alpha=0.4, linestyle='-', linewidth=0.8)
    ax2.axhline(y=harsi.upper, color='silver', alpha=0.6, linestyle='-', linewidth=0.8)
    ax2.axhline(y=0, color='orange', alpha=0.8, linestyle=':', linewidth=1)
    ax2.axhline(y=harsi.lower, color='silver', alpha=0.6, linestyle='-', linewidth=0.8)
    ax2.axhline(y=harsi.lower_x, color='silver', alpha=0.4, linestyle='-', linewidth=0.8)
    
    # RSI Histogram (using bar with proper width)
    bar_width = time_diff * 0.6  # Slightly narrower than candles
    ax2.bar(data.index, results['rsi'], alpha=0.2, color='silver', 
            width=bar_width, label='RSI Histogram', align='center')
    
    # Heikin Ashi RSI Candles
    valid_indices = ~(pd.isna(results['ha_open']) | pd.isna(results['ha_high']) | 
                     pd.isna(results['ha_low']) | pd.isna(results['ha_close']))
    
    for idx in data.index[valid_indices]:
        if idx not in results['ha_open'].index:
            continue
            
        o = results['ha_open'][idx]
        h = results['ha_high'][idx]
        l = results['ha_low'][idx]
        c = results['ha_close'][idx]
        
        if pd.isna(o) or pd.isna(h) or pd.isna(l) or pd.isna(c):
            continue
            
        draw_harsi_candlestick(ax2, idx, o, h, l, c, candle_width)
    
    # RSI line plot
    # ax2.plot(data.index, results['rsi'], color='gold', linewidth=1.5, alpha=0.8, label='RSI Line')
    
    # Plot buy/sell signals
    buy_indices = data.index[results['buy_signals']]
    sell_indices = data.index[results['sell_signals']]
    tp_sl_indices = data.index[results['tp_sl_signals']]
    
    if len(buy_indices) > 0:
        buy_rsi_values = results['rsi'][results['buy_signals']]
        ax2.scatter(buy_indices, buy_rsi_values, color='lime', s=100, marker='^', 
                   edgecolors='darkgreen', linewidth=2, label='BUY Signal', zorder=10)
        
        # Also plot buy signals on price chart
        buy_prices = data.loc[buy_indices, 'Close']
        ax1.scatter(buy_indices, buy_prices, color='lime', s=100, marker='^', 
                   edgecolors='darkgreen', linewidth=2, label='BUY Signal', zorder=10)
    
    if len(sell_indices) > 0:
        sell_rsi_values = results['rsi'][results['sell_signals']]
        ax2.scatter(sell_indices, sell_rsi_values, color='red', s=100, marker='v', 
                   edgecolors='darkred', linewidth=2, label='SELL Signal', zorder=10)
        
        # Also plot sell signals on price chart
        sell_prices = data.loc[sell_indices, 'Close']
        ax1.scatter(sell_indices, sell_prices, color='red', s=100, marker='v', 
                   edgecolors='darkred', linewidth=2, label='SELL Signal', zorder=10)

    if len(tp_sl_indices) > 0:
        tp_sl_rsi_values = results['rsi'][results['tp_sl_signals']]
        ax2.scatter(tp_sl_indices, tp_sl_rsi_values, color='purple', s=100, marker='x', 
                   edgecolors='blue', linewidth=2, label='TP/SL Signal', zorder=10)
        
        # Also plot TP/SL signals on price chart
        tp_sl_prices = data.loc[tp_sl_indices, 'Close']
        ax1.scatter(tp_sl_indices, tp_sl_prices, color='blue', s=100, marker='x', 
                   edgecolors='blue', linewidth=2, label='TP/SL Signal', zorder=10)
    
    # Add legends
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper left')
    
    # Stochastic RSI (optional)
    ax2.plot(data.index, results['stoch_k'], color='#0094FF', linewidth=1, alpha=0.7, label='Stoch K')
    ax2.plot(data.index, results['stoch_d'], color='#FF6A00', linewidth=1, alpha=0.7, label='Stoch D')
    
    ax2.set_title('Heikin Ashi RSI Oscillator (HARSI)', fontsize=14, fontweight='bold')
    ax2.set_ylabel('RSI Value', fontsize=12)
    ax2.set_xlabel('Date/Time', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(-60, 60)
    
    # Format x-axis for both subplots
    # Set major ticks and labels
    ax1.xaxis.set_major_locator(mdates.HourLocator(interval=6))  # Every 6 hours
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d %H:%M'))
    ax1.xaxis.set_minor_locator(mdates.HourLocator(interval=2))  # Every 2 hours
    
    ax2.xaxis.set_major_locator(mdates.HourLocator(interval=6))  # Every 6 hours
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d %H:%M'))
    ax2.xaxis.set_minor_locator(mdates.HourLocator(interval=2))  # Every 2 hours
    
    # Rotate labels for better readability
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Add grid lines for better alignment visualization
    ax1.grid(True, alpha=0.3, which='both')
    ax2.grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    plt.show()
    
    # Print some statistics
    buy_count = results['buy_signals'].sum()
    sell_count = results['sell_signals'].sum()
    tp_sl_count = results['tp_sl_signals'].sum()
    
    print(f"\nHARSI Statistics:")
    print(f"RSI Range: {results['rsi'].min():.2f} to {results['rsi'].max():.2f}")
    print(f"Current RSI: {results['rsi'].iloc[-1]:.2f}")
    print(f"Current HA Close: {results['ha_close'].iloc[-1]:.2f}")
    print(f"Total candles: {len(data)}")
    print(f"TP: {harsi.tp_percent}% | SL: {harsi.sl_percent}%")
    print(f"\nSignals Generated:")
    print(f"Buy signals: {buy_count}")
    print(f"Sell signals: {sell_count}")
    print(f"TP/SL hits: {tp_sl_count}")
    
    if buy_count > 0:
        print(f"\nBuy Signal Times:")
        for idx in data.index[results['buy_signals']]:
            price = data.loc[idx, 'Close']
            rsi_val = results['rsi'][idx]
            print(f"  {idx.strftime('%Y-%m-%d %H:%M')} - Price: ${price:.2f}, RSI: {rsi_val:.2f}")
    
    if sell_count > 0:
        print(f"\nSell Signal Times:")
        for idx in data.index[results['sell_signals']]:
            price = data.loc[idx, 'Close']
            rsi_val = results['rsi'][idx]
            print(f"  {idx.strftime('%Y-%m-%d %H:%M')} - Price: ${price:.2f}, RSI: {rsi_val:.2f}")
    
    if tp_sl_count > 0:
        print(f"\nTP/SL Hit Times:")
        for idx in data.index[results['tp_sl_signals']]:
            price = data.loc[idx, 'Close']
            rsi_val = results['rsi'][idx]
            print(f"  {idx.strftime('%Y-%m-%d %H:%M')} - Price: ${price:.2f}, RSI: {rsi_val:.2f}")

# Example usage
if __name__ == "__main__":
    # Plot AAPL by default
    plot_harsi_indicator('RAY-USD', 7)
    
    # You can also try other symbols
    # plot_harsi_indicator('MSFT', 59)
    # plot_harsi_indicator('TSLA', 59)