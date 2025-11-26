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
        self.smoothing = 25
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
        self.tp_percent = 1.5  # Take profit 1%
        self.sl_percent = 2.5  # Stop loss 1%
        self.tp_amount = 0.25  # Take profit amount in percentage
    
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
    
    def calculate_signals_with_tpsl(self, rsi, ha_open, ha_high, ha_low, ha_close, price_data, 
                                  ha_open_2h, ha_high_2h, ha_low_2h, ha_close_2h):
        """Calculate buy/sell signals with TP/SL tracking and 2-hour HARSI data"""
        # Determine HARSI candle colors (bullish/bearish)
        harsi_bullish = ha_close >= ha_open  # Green candles
        harsi_bearish = ha_close < ha_open   # Red candles
        
        # Determine 2-hour HARSI candle colors
        harsi_bullish_2h = ha_close_2h >= ha_open_2h  # Green candles
        harsi_bearish_2h = ha_close_2h < ha_open_2h   # Red candles
        
        # Calculate RSI trend (increasing/decreasing)
        rsi_increasing = rsi > rsi.shift(1)
        rsi_decreasing = rsi < rsi.shift(1)
        
        # Initialize signal arrays
        buy_signals = pd.Series(False, index=rsi.index)
        sell_signals = pd.Series(False, index=rsi.index)
        tp_sl_signals = pd.Series(False, index=rsi.index)
        
        # For tracking TP/SL
        prev_signal = ""
        last_tp = 0
        entry_price = 0
        sl_price = 0
        
        for i in range(1, len(rsi)):
            if pd.isna(rsi.iloc[i]) or pd.isna(ha_close.iloc[i]):
                continue
                
            # Current values
            current_price = price_data['Close'].iloc[i]
            rsi_curr = rsi.iloc[i]
            rsi_prev = rsi.iloc[i-1]
            rsi_inc = rsi_increasing.iloc[i]
            rsi_dec = rsi_decreasing.iloc[i]
            ha_high_curr = ha_high.iloc[i]
            ha_low_curr = ha_low.iloc[i]
            ha_close_curr = ha_close.iloc[i]
            ha_open_curr = ha_open.iloc[i]
            ha_green = harsi_bullish.iloc[i]
            ha_red = harsi_bearish.iloc[i]
            ha_height = ha_close_curr - ha_open_curr
            ha_green_prev = harsi_bullish.iloc[i]
            ha_green_prev2 = harsi_bullish.iloc[i-1]
            ha_close_prev = ha_close.iloc[i-1]

            # 2-hour HARSI values (align with current index if available)
            ha_close_2h_curr = None
            ha_open_2h_curr = None
            ha_green_2h = None
            ha_red_2h = None
            
            if i < len(ha_close_2h) and pd.notna(ha_close_2h.iloc[i]):
                ha_close_2h_curr = ha_close_2h.iloc[i]
                ha_open_2h_curr = ha_open_2h.iloc[i]
                ha_green_2h = harsi_bullish_2h.iloc[i]
                ha_red_2h = harsi_bearish_2h.iloc[i]
            
            buy_condition = ha_green and ha_green_prev and not ha_green_prev2 and ha_close_curr > ha_close_prev and prev_signal != "buy"
            sell_condition = ha_red and harsi_bearish.iloc[i-1] and ha_close_curr < ha_close_prev and prev_signal == "buy"
            
            # TP/SL condition
            if prev_signal == "buy":
                gain_pct = (current_price - entry_price) / entry_price * 100
                if gain_pct >= last_tp + self.tp_percent:
                    tp_sl_signals.iloc[i] = True
                    last_tp += self.tp_percent
                    sl_price = current_price * (1 - self.sl_percent / 100)
                if current_price <= sl_price:
                    sell_condition = True

            # Buy signal: RSI increasing and crosses above HARSI candle
            if buy_condition:
                buy_signals.iloc[i] = True
                prev_signal = "buy"
                entry_price = current_price  # Set entry price on buy signal
                sl_price = entry_price * (100 - self.sl_percent) / 100
            
            # Sell signal: RSI decreasing and crosses below HARSI candle
            if sell_condition:
                sell_signals.iloc[i] = True
                prev_signal = "sell"
                entry_price = 0 
                last_tp = 0
                sl_price = 0
        
        return buy_signals, sell_signals, tp_sl_signals
    
    def calculate(self, data, data_2h):
        """Calculate all indicator values"""
        # Calculate OHLC4 source for 1-hour data
        ohlc4 = (data['Open'] + data['High'] + data['Low'] + data['Close']) / 4
        
        # Standard/smoothed RSI for plot
        rsi_plot = self.smoothed_rsi(ohlc4, self.len_rsi)
        
        # Stochastic RSI
        stoch_k = self.stochastic(rsi_plot, self.stoch_len, self.smooth_k)
        stoch_d = stoch_k.rolling(window=self.smooth_d).mean()
        
        # Heikin Ashi RSI candles for 1-hour data
        ha_rsi = self.rsi_heikin_ashi(data)
        
        # Heikin Ashi RSI candles for 2-hour data
        ha_rsi_2h = self.rsi_heikin_ashi(data_2h)
        
        # Calculate buy/sell signals with TP/SL
        buy_signals, sell_signals, tp_sl_signals = self.calculate_signals_with_tpsl(
            rsi_plot, ha_rsi['Open'], ha_rsi['High'], ha_rsi['Low'], ha_rsi['Close'], data,
            ha_rsi_2h['Open'], ha_rsi_2h['High'], ha_rsi_2h['Low'], ha_rsi_2h['Close']
        )
        
        return {
            'rsi': rsi_plot,
            'stoch_k': stoch_k,
            'stoch_d': stoch_d,
            'rsi_2h': ha_rsi_2h['Close'],
            'ha_open': ha_rsi['Open'],
            'ha_high': ha_rsi['High'],
            'ha_low': ha_rsi['Low'],
            'ha_close': ha_rsi['Close'],
            'ha_open_2h': ha_rsi_2h['Open'],
            'ha_high_2h': ha_rsi_2h['High'],
            'ha_low_2h': ha_rsi_2h['Low'],
            'ha_close_2h': ha_rsi_2h['Close'],
            'buy_signals': buy_signals,
            'sell_signals': sell_signals,
            'tp_sl_signals': tp_sl_signals
        }

def setup_dynamic_scaling(ax, data_dict, data_index):
    """Setup dynamic y-axis scaling that responds to zoom changes"""
    def on_xlim_changed(ax):
        # Get current x-axis limits
        xlim = ax.get_xlim()
        
        # Convert matplotlib dates back to datetime for filtering
        start_date = mdates.num2date(xlim[0])
        end_date = mdates.num2date(xlim[1])
        
        # Filter data to visible range
        mask = (data_index >= start_date) & (data_index <= end_date)
        
        if not mask.any():
            return
            
        # Collect all visible data values
        visible_values = []
        for key, series in data_dict.items():
            if series is not None and not series.empty:
                visible_data = series[mask].dropna()
                if not visible_data.empty:
                    visible_values.extend(visible_data.tolist())
        
        if visible_values:
            y_min = min(visible_values)
            y_max = max(visible_values)
            y_range = y_max - y_min
            
            # Add padding (10% or minimum of 0.1 if range is very small)
            padding = max(y_range * 0.1, 0.1) if y_range > 0 else 0.1
            
            # Update y-axis limits
            ax.set_ylim(y_min - padding, y_max + padding)
            ax.figure.canvas.draw_idle()
    
    # Connect the callback to x-axis limit changes
    ax.callbacks.connect('xlim_changed', on_xlim_changed)
    
    return on_xlim_changed

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
    """Main function to plot price, HARSI indicator, and 2-hour HARSI"""
    # Fetch data
    print(f"Fetching {symbol} data for last {days} days...")
    data = fetch_data(symbol, days, '90m')
    data_2h = fetch_data(symbol, days, '4h')
    
    if data.empty or data_2h.empty:
        print("No data found!")
        return
    
    # Initialize indicator
    harsi = HeikinAshiRSIOscillator()
    
    # Calculate indicator values
    print("Calculating HARSI indicators...")
    results = harsi.calculate(data, data_2h)
    
    # Create plots with shared x-axis (3 subplots)
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 16), 
                                        gridspec_kw={'height_ratios': [2, 2, 1.5]}, 
                                        sharex=True)
    
    # Calculate candle width based on time interval
    time_diff = (data.index[1] - data.index[0]).total_seconds() / 86400  # Convert to days
    candle_width = timedelta(days=time_diff * 0.8)  # 80% of the interval
    
    # Calculate 2-hour candle width
    time_diff_2h = (data_2h.index[1] - data_2h.index[0]).total_seconds() / 86400
    candle_width_2h = timedelta(days=time_diff_2h * 0.8)
    
    # Plot 1: Price Chart with OHLC Candlesticks
    for idx, row in data.iterrows():
        draw_candlestick(ax1, idx, row['Open'], row['High'], row['Low'], row['Close'], candle_width)
    
    ax1.set_title(f'{symbol} Price Chart (1h, Last {days} Days)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Price ($)', fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    # Auto-scale price chart y-axis based on data range with some padding
    price_min = data[['Open', 'High', 'Low', 'Close']].min().min()
    price_max = data[['Open', 'High', 'Low', 'Close']].max().max()
    price_range = price_max - price_min
    price_padding = price_range * 0.05  # 5% padding for price chart
    ax1.set_ylim(price_min - price_padding, price_max + price_padding)
    
    # Setup dynamic scaling for price chart
    price_data_dict = {
        'Open': data['Open'],
        'High': data['High'], 
        'Low': data['Low'],
        'Close': data['Close']
    }
    setup_dynamic_scaling(ax1, price_data_dict, data.index)
    
    # Plot 2: 1-Hour HARSI Indicator
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
        ax1.scatter(buy_indices, buy_prices * 0.99, color='lime', s=100, marker='^', 
                   edgecolors='darkgreen', linewidth=2, label='BUY Signal', zorder=10)
    
    if len(sell_indices) > 0:
        sell_rsi_values = results['rsi'][results['sell_signals']]
        ax2.scatter(sell_indices, sell_rsi_values, color='red', s=100, marker='v', 
                   edgecolors='darkred', linewidth=2, label='SELL Signal', zorder=10)
        
        # Also plot sell signals on price chart
        sell_prices = data.loc[sell_indices, 'Close']
        ax1.scatter(sell_indices, sell_prices * 1.01, color='red', s=100, marker='v', 
                   edgecolors='darkred', linewidth=2, label='SELL Signal', zorder=10)

    if len(tp_sl_indices) > 0:
        tp_sl_rsi_values = results['rsi'][results['tp_sl_signals']]
        ax2.scatter(tp_sl_indices, tp_sl_rsi_values, color='purple', s=100, marker='x', 
                   edgecolors='blue', linewidth=2, label='TP/SL Signal', zorder=10)
        
        # Also plot TP/SL signals on price chart
        tp_sl_prices = data.loc[tp_sl_indices, 'Close']
        ax1.scatter(tp_sl_indices, tp_sl_prices, color='blue', s=100, marker='x', 
                   edgecolors='blue', linewidth=2, label='TP/SL Signal', zorder=10)
    
    # RSI line
    ax2.plot(data.index, results['rsi'], color="#0011FF", linewidth=1, alpha=0.7, label='RSI')
    
    ax2.set_title('1-Hour Heikin Ashi RSI Oscillator (HARSI)', fontsize=14, fontweight='bold')
    ax2.set_ylabel('RSI Value', fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    # Auto-scale HARSI y-axis based on data range with some padding
    valid_rsi_data = results['rsi'].dropna()
    valid_stoch_k = results['stoch_k'].dropna()
    valid_stoch_d = results['stoch_d'].dropna()
    valid_ha_data = pd.concat([results['ha_open'], results['ha_high'], results['ha_low'], results['ha_close']]).dropna()
    
    all_harsi_values = pd.concat([valid_rsi_data, valid_stoch_k, valid_stoch_d, valid_ha_data])
    if not all_harsi_values.empty:
        harsi_min = all_harsi_values.min()
        harsi_max = all_harsi_values.max()
        harsi_range = harsi_max - harsi_min
        harsi_padding = harsi_range * 0.1  # 10% padding
        ax2.set_ylim(harsi_min - harsi_padding, harsi_max + harsi_padding)
    
    # Setup dynamic scaling for HARSI chart
    harsi_data_dict = {
        'rsi': results['rsi'],
        'stoch_k': results['stoch_k'],
        'stoch_d': results['stoch_d'],
        'ha_open': results['ha_open'],
        'ha_high': results['ha_high'],
        'ha_low': results['ha_low'],
        'ha_close': results['ha_close']
    }
    setup_dynamic_scaling(ax2, harsi_data_dict, data.index)
    
    # Plot 3: 2-Hour HARSI Indicator
    # Background fills for 2-hour chart
    ax3.axhspan(harsi.upper, harsi.upper_x, alpha=0.1, color='red', label='OB Extreme')
    ax3.axhspan(harsi.upper, harsi.lower, alpha=0.1, color='blue', label='Channel')
    ax3.axhspan(harsi.lower, harsi.lower_x, alpha=0.1, color='green', label='OS Extreme')
    
    # Horizontal lines
    ax3.axhline(y=harsi.upper_x, color='silver', alpha=0.4, linestyle='-', linewidth=0.8)
    ax3.axhline(y=harsi.upper, color='silver', alpha=0.6, linestyle='-', linewidth=0.8)
    ax3.axhline(y=0, color='orange', alpha=0.8, linestyle=':', linewidth=1)
    ax3.axhline(y=harsi.lower, color='silver', alpha=0.6, linestyle='-', linewidth=0.8)
    ax3.axhline(y=harsi.lower_x, color='silver', alpha=0.4, linestyle='-', linewidth=0.8)
    
    # 2-Hour Heikin Ashi RSI Candles
    valid_indices_2h = ~(pd.isna(results['ha_open_2h']) | pd.isna(results['ha_high_2h']) | 
                        pd.isna(results['ha_low_2h']) | pd.isna(results['ha_close_2h']))
    
    for idx in data_2h.index[valid_indices_2h]:
        if idx not in results['ha_open_2h'].index:
            continue
            
        o = results['ha_open_2h'][idx]
        h = results['ha_high_2h'][idx]
        l = results['ha_low_2h'][idx]
        c = results['ha_close_2h'][idx]
        
        if pd.isna(o) or pd.isna(h) or pd.isna(l) or pd.isna(c):
            continue
            
        draw_harsi_candlestick(ax3, idx, o, h, l, c, candle_width_2h)
    
    # Plot 2-hour RSI line
    ax3.plot(data_2h.index, results['rsi_2h'], color="#FF6600", linewidth=2, alpha=0.8, label='2H RSI')
    
    ax3.set_title('2-Hour Heikin Ashi RSI Oscillator (HARSI)', fontsize=14, fontweight='bold')
    ax3.set_ylabel('RSI Value', fontsize=12)
    ax3.set_xlabel('Date/Time', fontsize=12)
    ax3.grid(True, alpha=0.3)
    
    # Auto-scale 2-hour HARSI y-axis
    valid_ha_data_2h = pd.concat([results['ha_open_2h'], results['ha_high_2h'], 
                                  results['ha_low_2h'], results['ha_close_2h']]).dropna()
    valid_rsi_2h = results['rsi_2h'].dropna()
    
    # Combine all 2-hour data for scaling
    all_2h_values = pd.concat([valid_ha_data_2h, valid_rsi_2h])
    
    if not all_2h_values.empty:
        harsi_2h_min = all_2h_values.min()
        harsi_2h_max = all_2h_values.max()
        harsi_2h_range = harsi_2h_max - harsi_2h_min
        harsi_2h_padding = harsi_2h_range * 0.1 if harsi_2h_range > 0 else 0.1
        ax3.set_ylim(harsi_2h_min - harsi_2h_padding, harsi_2h_max + harsi_2h_padding)
    
    # Setup dynamic scaling for 2-hour HARSI chart
    harsi_2h_data_dict = {
        'rsi_2h': results['rsi_2h'],
        'ha_open_2h': results['ha_open_2h'],
        'ha_high_2h': results['ha_high_2h'],
        'ha_low_2h': results['ha_low_2h'],
        'ha_close_2h': results['ha_close_2h']
    }
    setup_dynamic_scaling(ax3, harsi_2h_data_dict, data_2h.index)
    
    # Format x-axis for all subplots
    for ax in [ax1, ax2, ax3]:
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=6))  # Every 6 hours
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d %H:%M'))
        ax.xaxis.set_minor_locator(mdates.HourLocator(interval=2))  # Every 2 hours
        ax.grid(True, alpha=0.3, which='both')
    
    # Rotate labels for better readability (only for bottom chart)
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    plt.show()
    
    # Print some statistics
    buy_count = results['buy_signals'].sum()
    sell_count = results['sell_signals'].sum()
    tp_sl_count = results['tp_sl_signals'].sum()
    
    print(f"\nHARSI Statistics:")
    print(f"1H RSI Range: {results['rsi'].min():.2f} to {results['rsi'].max():.2f}")
    print(f"2H RSI Range: {results['rsi_2h'].min():.2f} to {results['rsi_2h'].max():.2f}")
    print(f"Current 1H RSI: {results['rsi'].iloc[-1]:.2f}")
    print(f"Current 2H RSI: {results['rsi_2h'].iloc[-1]:.2f}")
    print(f"Current 1H HA Close: {results['ha_close'].iloc[-1]:.2f}")
    print(f"Current 2H HA Close: {results['ha_close_2h'].iloc[-1]:.2f}")
    print(f"Total 1H candles: {len(data)}")
    print(f"Total 2H candles: {len(data_2h)}")
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

def fetch_data(symbol, days=7, interval='1h'):
    """Fetch data using yfinance"""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    ticker = yf.Ticker(symbol)
    data = ticker.history(start=start_date, end=end_date, interval=interval)
    
    return data

# Example usage
if __name__ == "__main__":
    # Plot RAY-USD by default
    plot_harsi_indicator('RAY-USD', 20)
    
    # You can also try other symbols
    # plot_harsi_indicator('AAPL', 59)
    # plot_harsi_indicator('MSFT', 59)
    # plot_harsi_indicator('TSLA', 59)