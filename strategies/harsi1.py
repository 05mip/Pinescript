import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import yfinance as yf
from datetime import datetime, timedelta
import matplotlib.dates as mdates

class HeikinAshiRSI:
    def __init__(self):
        self.config = {
            # HARSI Candle config
            'harsi_length': 14,
            'open_smoothing': 2,
            'color_up': '#00CED1',  # dark turquoise
            'color_down': '#FF4500',  # orangered
            'color_wick': '#808080',  # gray
            
            # RSI Plot config
            'rsi_length': 7,
            'smoothed_mode': True,
            'show_rsi_plot': True,
            'show_rsi_histogram': True,
            
            # Channel boundaries
            'upper_ob': 20,
            'upper_extreme': 30,
            'lower_os': -20,
            'lower_extreme': -30,
            
            # Crossover marker config
            'show_crossover_markers': True,
            'crossover_marker_size': 60,
            'bullish_cross_color': '#00FF00',  # bright green
            'bearish_cross_color': '#FF69B4',  # hot pink
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

                # RSI PEAK DETECTION ########################################
                # Calculate slopes
                slope1 = rsi_curr - rsi_prev
                slope2 = rsi_prev - rsi_prev2
                
                # Project next slope
                projected_next_slope = slope1 + (slope1 - slope2)
                
                # Local max occurs when slope changes from positive to negative
                current_peak = slope1 > 0 and slope1 < slope2  # momentum decreasing
                next_peak = slope1 > 0 and projected_next_slope < 0  # will turn negative
                
                rsi_peak = current_peak or next_peak
                
                # Check for crossovers
                # bullish_cross = ha_curr > ha_prev and rsi_increasing
                # bullish_cross = (rsi_prev < ha_prev and rsi_curr >= ha_curr and rsi_increasing)
                bullish_cross = (rsi_curr + (rsi_curr - rsi_prev) >= ha_curr and rsi_curr <= ha_curr and rsi_increasing)
                # bullish_cross = (rsi_curr + (rsi_curr - rsi_prev) >= ha_curr and rsi_curr <= ha_curr and rsi_increasing) or (rsi_prev < ha_prev and rsi_curr >= ha_curr and rsi_increasing and ha_curr < 0 and not (crossovers[-1]['index'] == i-1) and crossovers[-1]['type'] == 'bullish')
                # bearish_cross = (rsi_prev > ha_prev and rsi_curr <= ha_curr and not rsi_increasing)
                bearish_cross = rsi_curr >= self.config['upper_ob'] and (rsi_curr < rsi_prev) and not crossovers[-1]['type'] == 'bearish'
                # bearish_cross = rsi_peak and rsi_curr >= self.config['upper_ob'
                # bearish_cross = (rsi_curr + (rsi_curr - rsi_prev) <= ha_curr and rsi_curr >= ha_curr and not rsi_increasing)
                if bullish_cross:
                    crossovers.append({
                        'index': i,
                        'type': 'bullish',
                        'rsi_value': rsi_curr,
                        'ha_value': ha_curr,
                        'price_idx': i  # For plotting on price chart
                    })
                elif bearish_cross:
                    crossovers.append({
                        'index': i,
                        'type': 'bearish',
                        'rsi_value': rsi_curr,
                        'ha_value': ha_curr,
                        'price_idx': i  # For plotting on price chart
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

def plot_ohlc_candles(ax, ohlc_data, crossovers=None, width=0.8):
    """Plot regular OHLC candlesticks with crossover markers"""
    # Convert datetime index to numeric for plotting
    x_values = range(len(ohlc_data))
    
    for i, (idx, row) in enumerate(ohlc_data.iterrows()):
        open_price = row['Open']
        high_price = row['High']
        low_price = row['Low']
        close_price = row['Close']
        
        # Determine candle color
        color = '#00FF00' if close_price.iloc[0] > open_price.iloc[0] else '#FF0000'
        
        # Draw candle body
        body_height = abs(close_price.iloc[0] - open_price.iloc[0])
        body_bottom = min(open_price.iloc[0], close_price.iloc[0])
        
        rect = Rectangle((i - width/2, body_bottom), width, body_height, 
                        facecolor=color, edgecolor=color, alpha=0.8)
        ax.add_patch(rect)
        
        # Draw wicks
        ax.plot([i, i], [low_price, high_price], color='black', alpha=0.8, linewidth=1)
    
    # Add crossover markers on price chart
    if crossovers:
        for cross in crossovers:
            i = cross['index']
            if i < len(ohlc_data):
                high_price = ohlc_data['High'].iloc[i]
                low_price = ohlc_data['Low'].iloc[i]
                
                if cross['type'] == 'bullish':
                    # Place marker below the candle
                    y_pos = low_price - (high_price - low_price) * 0.05
                    ax.scatter(i, y_pos, marker='^', s=80, c='#00FF00', 
                             edgecolors='black', linewidth=1, zorder=5, label='Bullish Cross' if i == crossovers[0]['index'] else "")
                else:
                    # Place marker above the candle
                    y_pos = high_price + (high_price - low_price) * 0.05
                    ax.scatter(i, y_pos, marker='v', s=80, c='#FF69B4', 
                             edgecolors='black', linewidth=1, zorder=5, label='Bearish Cross' if i == crossovers[0]['index'] else "")
    
    # Set x-axis labels
    n_ticks = min(10, len(ohlc_data))
    tick_indices = np.linspace(0, len(ohlc_data)-1, n_ticks, dtype=int)
    ax.set_xticks(tick_indices)
    ax.set_xticklabels([ohlc_data.index[i].strftime('%m-%d %H:%M') for i in tick_indices], 
                       rotation=45, ha='right')
    
    ax.set_xlim(-0.5, len(ohlc_data) - 0.5)

def plot_harsi_indicator(ax, ohlc_data, harsi_results, config):
    """Plot the HARSI indicator on given axis with crossover markers"""
    # Convert datetime index to numeric for plotting
    x_values = range(len(ohlc_data))
    
    # Plot channel boundaries
    ax.axhline(y=config['upper_extreme'], color='silver', alpha=0.4, linestyle='-', linewidth=0.5)
    ax.axhline(y=config['upper_ob'], color='silver', alpha=0.6, linestyle='-', linewidth=0.5)
    ax.axhline(y=0, color='orange', alpha=0.8, linestyle=':', linewidth=1)
    ax.axhline(y=config['lower_os'], color='silver', alpha=0.6, linestyle='-', linewidth=0.5)
    ax.axhline(y=config['lower_extreme'], color='silver', alpha=0.4, linestyle='-', linewidth=0.5)
    
    # Fill channel areas
    ax.fill_between(x_values, config['upper_ob'], config['upper_extreme'], 
                   color='red', alpha=0.1, label='OB Extreme')
    ax.fill_between(x_values, config['upper_ob'], config['lower_os'], 
                   color='blue', alpha=0.1, label='Normal Channel')
    ax.fill_between(x_values, config['lower_os'], config['lower_extreme'], 
                   color='green', alpha=0.1, label='OS Extreme')
    
    # Plot RSI histogram (if enabled)
    if config['show_rsi_histogram']:
        rsi_values = [harsi_results['rsi_line'].iloc[i] if i < len(harsi_results['rsi_line']) 
                     and not pd.isna(harsi_results['rsi_line'].iloc[i]) else 0 for i in x_values]
        ax.bar(x_values, rsi_values, alpha=0.2, color='silver', width=0.8)
    
    # Plot Heikin-Ashi candles
    for i in x_values:
        try:
            ha_o = harsi_results['ha_open'].iloc[i]
            ha_h = harsi_results['ha_high'].iloc[i]
            ha_l = harsi_results['ha_low'].iloc[i]
            ha_c = harsi_results['ha_close'].iloc[i]
        except IndexError:
            continue
        
        if pd.isna(ha_o) or pd.isna(ha_c) or pd.isna(ha_h) or pd.isna(ha_l):
            continue
        
        # Determine candle color
        color = config['color_up'] if ha_c > ha_o else config['color_down']
        
        # Draw candle body
        body_height = abs(ha_c - ha_o)
        body_bottom = min(ha_o, ha_c)
        
        rect = Rectangle((i - 0.3, body_bottom), 0.6, body_height, 
                       facecolor=color, edgecolor=color, alpha=0.8)
        ax.add_patch(rect)
        
        # Draw wicks
        ax.plot([i, i], [ha_l, ha_h], color=config['color_wick'], alpha=0.8, linewidth=1)
    
    # Plot RSI line (if enabled)
    if config['show_rsi_plot']:
        rsi_values = [harsi_results['rsi_line'].iloc[i] if i < len(harsi_results['rsi_line']) 
                     and not pd.isna(harsi_results['rsi_line'].iloc[i]) else np.nan for i in x_values]
        ax.plot(x_values, rsi_values, color='#FAC832', linewidth=2, label='RSI Line')
    
    # Add crossover markers on RSI chart
    if config['show_crossover_markers'] and harsi_results['crossovers']:
        for cross in harsi_results['crossovers']:
            i = cross['index']
            rsi_val = cross['rsi_value']
            
            if cross['type'] == 'bullish':
                ax.scatter(i, rsi_val, marker='o', s=config['crossover_marker_size'], 
                         c=config['bullish_cross_color'], edgecolors='black', linewidth=2, 
                         zorder=10, label='RSI Bullish Cross' if i == harsi_results['crossovers'][0]['index'] else "")
            else:
                ax.scatter(i, rsi_val, marker='o', s=config['crossover_marker_size'], 
                         c=config['bearish_cross_color'], edgecolors='black', linewidth=2, 
                         zorder=10, label='RSI Bearish Cross' if i == harsi_results['crossovers'][0]['index'] else "")
    
    # Set x-axis labels
    n_ticks = min(10, len(ohlc_data))
    tick_indices = np.linspace(0, len(ohlc_data)-1, n_ticks, dtype=int)
    ax.set_xticks(tick_indices)
    ax.set_xticklabels([ohlc_data.index[i].strftime('%m-%d %H:%M') for i in tick_indices], 
                       rotation=45, ha='right')
    
    ax.set_xlim(-0.5, len(ohlc_data) - 0.5)
    
    # Set y-axis limits
    rsi_valid = harsi_results['rsi_line'].dropna()
    if not rsi_valid.empty:
        y_min = min(config['lower_extreme'] - 5, rsi_valid.min() - 5)
        y_max = max(config['upper_extreme'] + 5, rsi_valid.max() + 5)
        ax.set_ylim(y_min, y_max)

def create_dual_panel_chart(ticker="XRP-USD", days=59):
    """Create dual panel chart with OHLC on top and HARSI on bottom"""
    
    # Download 1-hour data from Yahoo Finance
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    try:
        print(f"Downloading 1-hour data for {ticker}...")
        data = yf.download(ticker, start=start_date, end=end_date, interval='60m', auto_adjust=True)
        
        # Clean the data
        data = data.dropna()
        data = data.sort_index()
        
        # Limit data for better visualization
        if len(data) > 300:
            data = data.tail(300)
            
        print(f"Using {len(data)} hours of data from {data.index.min()} to {data.index.max()}")
        
    except Exception as e:
        print(f"Error downloading data: {e}")
        return None, None
    
    if len(data) < 50:
        print("Insufficient data for analysis")
        return None, None
    
    # Create HARSI indicator
    harsi = HeikinAshiRSI()
    harsi_results = harsi.calculate_indicator(data)
    
    # Create dual panel chart
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12), height_ratios=[2, 1], 
                                   gridspec_kw={'hspace': 0.3})
    
    # Top panel: OHLC Candles with crossover markers
    plot_ohlc_candles(ax1, data, harsi_results['crossovers'])
    ax1.set_title(f'{ticker} - 1 Hour OHLC Candles with RSI-HA Crossover Signals', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Price ($)', fontsize=12)
    ax1.grid(True, alpha=0.3)
    if harsi_results['crossovers']:
        # ax1.legend(loc='upper left')
        pass
    
    # Bottom panel: HARSI Indicator with crossover markers
    plot_harsi_indicator(ax2, data, harsi_results, harsi.config)
    ax2.set_title('Heikin Ashi RSI Oscillator (HARSI) with Crossover Signals', fontsize=14, fontweight='bold')
    ax2.set_ylabel('RSI Value', fontsize=12)
    ax2.set_xlabel('Date & Time', fontsize=12)
    ax2.grid(True, alpha=0.3)
    if harsi_results['crossovers']:
        # ax2.legend(loc='upper right')
        pass
    
    plt.tight_layout()
    
    # Print statistics and crossover information
    rsi_valid = harsi_results['rsi_line'].dropna()
    ha_close_valid = harsi_results['ha_close'].dropna()
    
    if not rsi_valid.empty and not ha_close_valid.empty:
        print(f"\nHARSI Statistics for {ticker}:")
        print(f"RSI Line - Mean: {rsi_valid.mean():.2f}, Std: {rsi_valid.std():.2f}")
        print(f"RSI Line - Min: {rsi_valid.min():.2f}, Max: {rsi_valid.max():.2f}")
        print(f"HA Close - Mean: {ha_close_valid.mean():.2f}, Std: {ha_close_valid.std():.2f}")
        print(f"Current RSI: {rsi_valid.iloc[-1]:.2f}")
        print(f"Current HA Close: {ha_close_valid.iloc[-1]:.2f}")
    
    # Print crossover signals
    if harsi_results['crossovers']:
        print(f"\nFound {len(harsi_results['crossovers'])} crossover signals:")
        for i, cross in enumerate(harsi_results['crossovers'][-10:]):  # Show last 10
            timestamp = data.index[cross['index']].strftime('%Y-%m-%d %H:%M')
            cross_type = cross['type'].upper()
            rsi_val = cross['rsi_value']
            ha_val = cross['ha_value']
            print(f"  {i+1}. {timestamp} - {cross_type} Cross: RSI={rsi_val:.2f}, HA={ha_val:.2f}")
    else:
        print("\nNo crossover signals found in the current data range.")
    
    return fig, (harsi, data)

# Run the analysis
if __name__ == "__main__":
    # You can change these parameters
    TICKER = "RAY-USD"  # Change to any stock symbol
    DAYS = 59        # Number of days of hourly data to fetch
    
    fig, results = create_dual_panel_chart(TICKER, DAYS)
    
    if fig is not None:
        plt.show()
        
        # You can also try other tickers
        # create_dual_panel_chart("TSLA", 14)
        # create_dual_panel_chart("SPY", 14)
    else:
        print("Failed to create chart")