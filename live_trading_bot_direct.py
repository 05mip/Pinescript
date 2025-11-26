# ==== ANIMATED SLIDING WINDOW HARSI BACKTEST ====

from dataclasses import dataclass
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle
from matplotlib import dates as mdates
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')
from strategies.harsi2_macd import *

def animate_sliding_window(
    symbol='RAY-USD',
    days=59,
    interval='1h',
    window=500,
    initial_capital=1000.0,
    tp_fraction=0.30,
    animation_speed=50  # milliseconds between frames
):
    """
    Create an animated visualization of the sliding window backtest
    """
    print(f"Fetching data for {symbol}...")
    data = fetch_data(symbol, days, interval)
    data_2h = fetch_data(symbol, days, '4h')
    
    if data.empty or data_2h.empty:
        raise ValueError("No data returned.")

    price = data['Close']
    
    # Initialize trading variables
    cash = initial_capital
    position = 0.0
    in_trade = False
    entry_price = None
    equity_curve = []
    
    # Store all calculation results for animation
    animation_data = []
    
    print("Calculating sliding window data...")
    for i in range(window, len(data)):
        # Slice last `window` bars
        window_data_1h = data.iloc[i - window:i]
        window_data_2h = data_2h.iloc[i - window:i]
        
        # Recalculate HARSI fresh on the slice
        harsi = HeikinAshiRSIOscillator()
        res = harsi.calculate(window_data_1h, window_data_2h)
        
        buy_sig = res['buy_signals'].iloc[-1]
        sell_sig = res['sell_signals'].iloc[-1]
        tp_sig = res['tp_sl_signals'].iloc[-1]
        
        t = price.index[i]
        px = float(price.iloc[i])
        
        # Trading logic
        trade_action = None
        if buy_sig and position == 0.0 and cash > 0.0:
            position = cash / px
            cash = 0.0
            entry_price = px
            in_trade = True
            trade_action = 'BUY'
            
        elif tp_sig and position > 0.0:
            sell_units = position * tp_fraction
            cash += sell_units * px
            position -= sell_units
            trade_action = 'TP'
            
        elif sell_sig and position > 0.0:
            cash += position * px
            position = 0.0
            in_trade = False
            entry_price = None
            trade_action = 'SELL'
        
        equity = cash + position * px
        equity_curve.append((t, equity))
        
        # Store data for this frame
        animation_data.append({
            'window_start': i - window,
            'window_end': i,
            'current_time': t,
            'current_price': px,
            'window_data': window_data_1h.copy(),
            'harsi_results': res.copy(),
            'trade_action': trade_action,
            'equity': equity,
            'position': position,
            'cash': cash,
            'buy_sig': buy_sig,
            'sell_sig': sell_sig,
            'tp_sig': tp_sig
        })
    
    print(f"Prepared {len(animation_data)} frames for animation")
    
    # Create the animated plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12), 
                                   gridspec_kw={'height_ratios': [1.5, 2]})
    
    def animate_frame(frame_idx):
        if frame_idx >= len(animation_data):
            return
            
        ax1.clear()
        ax2.clear()
        
        frame_data = animation_data[frame_idx]
        window_data = frame_data['window_data']
        results = frame_data['harsi_results']
        current_time = frame_data['current_time']
        current_price = frame_data['current_price']
        trade_action = frame_data['trade_action']
        equity = frame_data['equity']
        
        # Calculate candle width
        if len(window_data) > 1:
            time_diff = (window_data.index[1] - window_data.index[0]).total_seconds() / 86400
            candle_width = timedelta(days=time_diff * 0.8)
        else:
            candle_width = timedelta(hours=1)
        
        # Plot 1: Price Chart
        for idx, row in window_data.iterrows():
            color = 'green' if row['Close'] >= row['Open'] else 'red'
            
            # Draw wick
            ax1.plot([idx, idx], [row['Low'], row['High']], color='black', linewidth=1, alpha=0.8)
            
            # Draw body
            body_height = abs(row['Close'] - row['Open'])
            body_bottom = min(row['Open'], row['Close'])
            
            if body_height > 0:
                rect = Rectangle((idx - candle_width/2, body_bottom), candle_width, body_height,
                               facecolor=color, edgecolor='black', alpha=0.8, linewidth=0.5)
                ax1.add_patch(rect)
            else:
                ax1.plot([idx - candle_width/2, idx + candle_width/2], [row['Close'], row['Close']], 
                        color='black', linewidth=2)
        
        # Highlight current price point
        ax1.scatter([current_time], [current_price], color='yellow', s=100, 
                   edgecolors='black', linewidth=2, zorder=10, label='Current Price')
        
        # Show trade action
        if trade_action == 'BUY':
            ax1.scatter([current_time], [current_price * 0.99], color='lime', s=200, marker='^', 
                       edgecolors='darkgreen', linewidth=3, zorder=11, label='BUY')
        elif trade_action == 'SELL':
            ax1.scatter([current_time], [current_price * 1.01], color='red', s=200, marker='v', 
                       edgecolors='darkred', linewidth=3, zorder=11, label='SELL')
        elif trade_action == 'TP':
            ax1.scatter([current_time], [current_price], color='blue', s=200, marker='x', 
                       edgecolors='blue', linewidth=3, zorder=11, label='TP')
        
        ax1.set_title(f'{symbol} - Sliding Window (Frame {frame_idx + 1}/{len(animation_data)}) - '
                     f'{current_time.strftime("%Y-%m-%d %H:%M")} - Equity: ${equity:.2f}', 
                     fontsize=14, fontweight='bold')
        ax1.set_ylabel('Price ($)', fontsize=12)
        ax1.grid(True, alpha=0.3)
        # ax1.legend()
        
        # Format price chart x-axis
        ax1.xaxis.set_major_locator(mdates.HourLocator(interval=24))
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
        
        # Plot 2: HARSI Indicator
        harsi = HeikinAshiRSIOscillator()  # Get parameters
        
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
        
        # RSI line
        ax2.plot(window_data.index, results['rsi'], color="#0011FF", linewidth=1.5, alpha=0.8, label='RSI')
        
        # HARSI Candlesticks
        for idx in window_data.index:
            if (idx in results['ha_open'].index and 
                not pd.isna(results['ha_open'][idx]) and 
                not pd.isna(results['ha_close'][idx])):
                
                o = results['ha_open'][idx]
                h = results['ha_high'][idx]
                l = results['ha_low'][idx]
                c = results['ha_close'][idx]
                
                color = 'teal' if c >= o else 'red'
                body_height = abs(c - o)
                body_bottom = min(o, c)
                
                # Draw wick
                ax2.plot([idx, idx], [l, h], color='gray', linewidth=1, alpha=0.7)
                
                # Draw body
                if body_height > 0:
                    rect = Rectangle((idx - candle_width/2, body_bottom), candle_width, body_height,
                                   facecolor=color, edgecolor=color, alpha=0.8, linewidth=0.5)
                    ax2.add_patch(rect)
                else:
                    ax2.plot([idx - candle_width/2, idx + candle_width/2], [c, c], 
                            color=color, linewidth=2)
        
        # Plot signals
        buy_indices = window_data.index[results['buy_signals']]
        sell_indices = window_data.index[results['sell_signals']]
        tp_indices = window_data.index[results['tp_sl_signals']]
        
        if len(buy_indices) > 0:
            buy_rsi_values = results['rsi'][results['buy_signals']]
            ax2.scatter(buy_indices, buy_rsi_values, color='lime', s=100, marker='^', 
                       edgecolors='darkgreen', linewidth=2, label='BUY Signal', zorder=10)
        
        if len(sell_indices) > 0:
            sell_rsi_values = results['rsi'][results['sell_signals']]
            ax2.scatter(sell_indices, sell_rsi_values, color='red', s=100, marker='v', 
                       edgecolors='darkred', linewidth=2, label='SELL Signal', zorder=10)
        
        if len(tp_indices) > 0:
            tp_rsi_values = results['rsi'][results['tp_sl_signals']]
            ax2.scatter(tp_indices, tp_rsi_values, color='blue', s=100, marker='x', 
                       edgecolors='blue', linewidth=2, label='TP Signal', zorder=10)
        
        # Highlight current point on indicator
        if not pd.isna(results['rsi'].iloc[-1]):
            ax2.scatter([current_time], [results['rsi'].iloc[-1]], color='yellow', s=100, 
                       edgecolors='black', linewidth=2, zorder=11)
        
        ax2.set_title('Heikin Ashi RSI Oscillator (HARSI) - Current Window', fontsize=14, fontweight='bold')
        ax2.set_ylabel('RSI Value', fontsize=12)
        ax2.set_xlabel('Time', fontsize=12)
        ax2.grid(True, alpha=0.3)
        # ax2.legend(loc='upper right')
        
        # Format HARSI chart x-axis
        ax2.xaxis.set_major_locator(mdates.HourLocator(interval=24))
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
    
    # Create animation
    print("Creating animation...")
    anim = animation.FuncAnimation(fig, animate_frame, frames=len(animation_data), 
                                 interval=animation_speed, repeat=False, blit=False)
    
    # Show the animation
    plt.show()
    
    # Option to save as GIF or MP4
    save_animation = input("Save animation? (y/n): ").lower().strip() == 'y'
    if save_animation:
        format_choice = input("Save as (1) GIF or (2) MP4? Enter 1 or 2: ").strip()
        filename = f"harsi_sliding_window_{symbol}_{window}bars"
        
        if format_choice == '1':
            print("Saving as GIF (this may take a while)...")
            anim.save(f"{filename}.gif", writer='pillow', fps=20, 
                     savefig_kwargs={'bbox_inches': 'tight', 'dpi': 100})
            print(f"Animation saved as {filename}.gif")
        elif format_choice == '2':
            print("Saving as MP4...")
            try:
                anim.save(f"{filename}.mp4", writer='ffmpeg', fps=20, 
                         savefig_kwargs={'bbox_inches': 'tight', 'dpi': 100})
                print(f"Animation saved as {filename}.mp4")
            except Exception as e:
                print(f"Could not save MP4 (ffmpeg required): {e}")
                print("Try saving as GIF instead")
    
    return anim

def fetch_data(symbol, days=7, interval='1h'):
    """Fetch data using yfinance"""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    ticker = yf.Ticker(symbol)
    data = ticker.history(start=start_date, end=end_date, interval=interval)
    
    return data

if __name__ == "__main__":
    # Run the animated sliding window
    anim = animate_sliding_window(
        symbol='RAY-USD',
        days=59,
        interval='1h',
        window=500,
        initial_capital=1000.0,
        tp_fraction=0.30,
        animation_speed=10
    )