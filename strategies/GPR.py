import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from scipy.linalg import pinv
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

class GaussianProcessRegression:
    """
    Gaussian Process Regression implementation based on the PineScript indicator
    """
    def __init__(self, window=100, forecast=20, length=20.0, sigma=0.01):
        self.window = window
        self.forecast = forecast
        self.length = length
        self.sigma = sigma
        
    def rbf_kernel(self, x1, x2, l):
        """Radial Basis Function kernel"""
        return np.exp(-np.power(x1 - x2, 2) / (2.0 * np.power(l, 2)))
    
    def kernel_matrix(self, X1, X2, l):
        """Create kernel matrix between two sets of points"""
        km = np.zeros((len(X1), len(X2)))
        for i, x1 in enumerate(X1):
            for j, x2 in enumerate(X2):
                km[i, j] = self.rbf_kernel(x1, x2, l)
        return km
    
    def predict(self, prices):
        """
        Predict future prices using Gaussian Process Regression
        """
        if len(prices) < self.window:
            return None, None
        
        # Use the last 'window' prices for training
        ytrain = prices[-self.window:].values
        mean_price = float(np.mean(ytrain))
        ytrain_centered = ytrain - mean_price
        
        # Create training indices
        xtrain = np.arange(self.window)
        xtest = np.arange(self.window + self.forecast)
        
        # Build kernel matrices
        identity = np.eye(self.window)
        Ktrain = self.kernel_matrix(xtrain, xtrain, self.length)
        Ktrain += identity * (self.sigma ** 2)  # Add noise term
        
        # Compute inverse (with regularization for numerical stability)
        try:
            K_inv = pinv(Ktrain)
        except:
            return None, None
        
        K_star = self.kernel_matrix(xtrain, xtest, self.length)
        K_source = K_star.T @ K_inv
        
        # Make predictions
        mu = K_source @ ytrain_centered
        
        # Add back the mean
        predictions = mu + mean_price
        
        # Return fitted values and forecasts
        fitted = predictions[:self.window]
        forecast = predictions[self.window:]
        
        return fitted, forecast

class GPRTradingStrategy:
    """
    Trading strategy based on GPR predictions
    """
    def __init__(self, gpr_model, min_prediction_strength=0.002, stop_loss_pct=0.02, take_profit_pct=0.04, min_hold_days=3):
        self.gpr = gpr_model
        self.min_prediction_strength = min_prediction_strength
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.min_hold_days = min_hold_days  # Minimum days to hold position
        
    def generate_signals(self, data):
        """
        Generate trading signals based on GPR predictions
        """
        signals = pd.DataFrame(index=data.index)
        signals['price'] = data['Close']
        signals['signal'] = 0
        signals['predicted_direction'] = 0
        signals['prediction_strength'] = 0.0
        signals['model_confidence'] = 0.0  # New column for model confidence
        signals['position'] = 0
        signals['entry_price'] = np.nan
        signals['exit_price'] = np.nan
        signals['returns'] = 0.0
        
        current_position = 0
        entry_price = None
        entry_day = None
        last_prediction_direction = 0  # Track previous prediction for reversal detection
        
        # Generate predictions for each day (starting from window size)
        for i in range(self.gpr.window, len(data)):
            try:
                # Get historical prices up to current point
                historical_prices = data['Close'].iloc[:i+1]
                
                # Make prediction
                fitted, forecast = self.gpr.predict(historical_prices)
                
                if forecast is not None and len(forecast) > 0:
                    current_price = float(historical_prices.iloc[-1])
                    
                    # Use first few forecast points to determine direction
                    short_term_forecast = forecast[:min(5, len(forecast))]
                    predicted_price = float(short_term_forecast[0])
                    
                    # Calculate prediction strength (relative change) - ensure it's a scalar
                    prediction_strength = float((predicted_price - current_price) / current_price)
                    
                    # Calculate model confidence based on forecast consistency
                    if len(short_term_forecast) > 1:
                        forecast_trend = np.mean(np.diff(short_term_forecast))
                        model_confidence = min(abs(forecast_trend) * 1000, 1.0)  # Scale to 0-1
                    else:
                        model_confidence = abs(prediction_strength) * 100
                    
                    signals.iloc[i, signals.columns.get_loc('prediction_strength')] = prediction_strength
                    signals.iloc[i, signals.columns.get_loc('model_confidence')] = model_confidence
                    
                    # Determine predicted direction
                    current_prediction_direction = 0
                    if prediction_strength > self.min_prediction_strength:
                        current_prediction_direction = 1  # Bullish
                    elif prediction_strength < -self.min_prediction_strength:
                        current_prediction_direction = -1  # Bearish
                    
                    signals.iloc[i, signals.columns.get_loc('predicted_direction')] = current_prediction_direction
                    
                    # Check for prediction reversal (reduces whipsaws)
                    prediction_reversal = (last_prediction_direction != 0 and 
                                         current_prediction_direction != 0 and
                                         last_prediction_direction != current_prediction_direction)
                    
                    # Strong prediction condition (combines strength and confidence)
                    strong_bullish = (prediction_strength > self.min_prediction_strength and 
                                    model_confidence > 0.3)
                    strong_bearish = (prediction_strength < -self.min_prediction_strength and 
                                    model_confidence > 0.3)
                    
                    # Generate trading signals (only long positions as requested)
                    if current_position == 0:  # No position
                        # Enter only on strong bullish signals or prediction reversals to bullish
                        if strong_bullish and (prediction_reversal or last_prediction_direction <= 0):
                            # Enter long position
                            signals.iloc[i, signals.columns.get_loc('signal')] = 1
                            signals.iloc[i, signals.columns.get_loc('position')] = 1
                            signals.iloc[i, signals.columns.get_loc('entry_price')] = current_price
                            current_position = 1
                            entry_price = current_price
                            entry_day = i
                            
                    elif current_position == 1:  # Long position
                        # Check for exit conditions
                        pnl_pct = (current_price - entry_price) / entry_price
                        days_held = i - entry_day
                        
                        # Exit conditions: stop loss, take profit, strong bearish signal, or minimum hold time + weak signal
                        exit_condition = (pnl_pct <= -self.stop_loss_pct or 
                                        pnl_pct >= self.take_profit_pct or 
                                        strong_bearish or
                                        (days_held >= self.min_hold_days and prediction_strength < 0))
                        
                        if exit_condition:
                            # Exit long position
                            signals.iloc[i, signals.columns.get_loc('signal')] = -1
                            signals.iloc[i, signals.columns.get_loc('position')] = 0
                            signals.iloc[i, signals.columns.get_loc('exit_price')] = current_price
                            signals.iloc[i, signals.columns.get_loc('returns')] = pnl_pct
                            current_position = 0
                            entry_price = None
                            entry_day = None
                        else:
                            # Continue holding
                            signals.iloc[i, signals.columns.get_loc('position')] = 1
                    
                    # Update last prediction direction
                    last_prediction_direction = current_prediction_direction
            except Exception as e:
                print(f"Error at index {i}: {e}")
                continue
        
        return signals

def backtest_strategy(signals):
    """
    Backtest the trading strategy and calculate performance metrics
    """
    # Calculate cumulative returns
    signals['strategy_returns'] = signals['returns'].fillna(0)
    signals['cumulative_returns'] = (1 + signals['strategy_returns']).cumprod()
    
    # Calculate buy and hold returns for comparison
    if len(signals) > 1:
        signals['buy_hold_returns'] = signals['price'].pct_change().fillna(0)
        signals['buy_hold_cumulative'] = (1 + signals['buy_hold_returns']).cumprod()
    else:
        signals['buy_hold_returns'] = 0
        signals['buy_hold_cumulative'] = 1
    
    # Performance metrics
    total_return = signals['cumulative_returns'].iloc[-1] - 1 if len(signals) > 0 else 0
    buy_hold_return = signals['buy_hold_cumulative'].iloc[-1] - 1 if len(signals) > 0 else 0
    
    # Calculate additional metrics
    trades = signals[signals['signal'] != 0]
    winning_trades = signals[signals['returns'] > 0]
    losing_trades = signals[signals['returns'] < 0]
    
    num_trades = len(trades[trades['signal'] == 1])  # Count entry signals
    win_rate = len(winning_trades) / num_trades if num_trades > 0 else 0
    
    # Handle case where no trades were made
    strategy_trade_returns = signals['strategy_returns'][signals['strategy_returns'] != 0]
    avg_return = strategy_trade_returns.mean() if len(strategy_trade_returns) > 0 else 0
    
    # Calculate volatility and Sharpe ratio
    if len(signals) > 1 and signals['strategy_returns'].std() > 0:
        volatility = signals['strategy_returns'].std() * np.sqrt(252)
        annualized_return = total_return * 252 / len(signals)
        sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
    else:
        volatility = 0
        sharpe_ratio = 0
    
    # Calculate maximum drawdown
    if len(signals) > 1:
        running_max = signals['cumulative_returns'].expanding().max()
        drawdown = (running_max - signals['cumulative_returns']) / running_max
        max_drawdown = drawdown.max()
    else:
        max_drawdown = 0
    
    return {
        'total_return': total_return,
        'buy_hold_return': buy_hold_return,
        'num_trades': num_trades,
        'win_rate': win_rate,
        'avg_return_per_trade': avg_return,
        'volatility': volatility,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown
    }

def plot_results(data, signals):
    """
    Plot the results of the backtest
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12), sharex=True)
    
    # Plot 1: Candlestick chart with signals
    # Create OHLC data for candlesticks
    from matplotlib.patches import Rectangle
    from matplotlib.lines import Line2D
    
    # Plot candlesticks
    for i in range(len(data)):
        date = data.index[i]
        open_price = float(data['Open'].iloc[i])
        high_price = float(data['High'].iloc[i])
        low_price = float(data['Low'].iloc[i])
        close_price = float(data['Close'].iloc[i])
        
        # Color: green if close > open, red otherwise
        color = 'green' if close_price >= open_price else 'red'
        
        # Draw the high-low line
        ax1.plot([date, date], [low_price, high_price], color='black', linewidth=0.8, alpha=0.8)
        
        # Draw the body (rectangle)
        body_height = abs(close_price - open_price)
        body_bottom = min(open_price, close_price)
        
        # Make candles much thinner
        candle_width = pd.Timedelta(days=0.8)  # Reduced from 0.6
        
        if body_height > 0:  # Avoid zero-height rectangles
            rect = Rectangle((date - candle_width/2, body_bottom), width=candle_width, 
                           height=body_height, facecolor=color, alpha=0.8, 
                           edgecolor='black', linewidth=0.3)
            ax1.add_patch(rect)
        else:
            # For doji candles (open = close), draw a thin line
            ax1.plot([date - candle_width/2, date + candle_width/2], 
                    [close_price, close_price], color='black', linewidth=1)
    
    # Mark entry and exit points
    entries = signals[signals['signal'] == 1]
    exits = signals[signals['signal'] == -1]
    
    ax1.scatter(entries.index, entries['price'], color='lime', marker='^', 
                s=150, label='Buy Signal', zorder=10, edgecolors='black')
    ax1.scatter(exits.index, exits['price'], color='red', marker='v', 
                s=150, label='Sell Signal', zorder=10, edgecolors='black')
    
    ax1.set_title('Price (Candlesticks) and Trading Signals')
    ax1.set_ylabel('Price ($)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Prediction strength
    ax2.plot(signals.index, signals['prediction_strength'], label='Prediction Strength', 
             alpha=0.8, linewidth=1.5, color='blue')
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5, linewidth=1)
    ax2.axhline(y=0.001, color='green', linestyle='--', alpha=0.7, 
                linewidth=2, label='Bullish Threshold')
    ax2.axhline(y=-0.001, color='red', linestyle='--', alpha=0.7, 
                linewidth=2, label='Bearish Threshold')
    
    # Fill areas above/below thresholds
    ax2.fill_between(signals.index, 0.001, signals['prediction_strength'].max(), 
                     where=(signals['prediction_strength'] > 0.001), 
                     alpha=0.1, color='green', label='Bullish Zone')
    ax2.fill_between(signals.index, -0.001, signals['prediction_strength'].min(), 
                     where=(signals['prediction_strength'] < -0.001), 
                     alpha=0.1, color='red', label='Bearish Zone')
    
    ax2.set_title('GPR Prediction Strength')
    ax2.set_ylabel('Prediction Strength (%)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Cumulative returns
    ax3.plot(signals.index, signals['cumulative_returns'], label='GPR Strategy', 
             linewidth=3, color='blue', alpha=0.9)
    ax3.plot(signals.index, signals['buy_hold_cumulative'], label='Buy & Hold', 
             linewidth=2, color='orange', alpha=0.8)
    
    # Fill between curves to show outperformance
    ax3.fill_between(signals.index, signals['cumulative_returns'], 
                     signals['buy_hold_cumulative'], 
                     where=(signals['cumulative_returns'] >= signals['buy_hold_cumulative']), 
                     alpha=0.2, color='green', label='Outperformance')
    ax3.fill_between(signals.index, signals['cumulative_returns'], 
                     signals['buy_hold_cumulative'], 
                     where=(signals['cumulative_returns'] < signals['buy_hold_cumulative']), 
                     alpha=0.2, color='red', label='Underperformance')
    
    ax3.set_title('Cumulative Returns Comparison')
    ax3.set_ylabel('Cumulative Returns')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Model Prediction Direction and Confidence
    # Create prediction direction visualization
    bullish_periods = signals['predicted_direction'] == 1
    bearish_periods = signals['predicted_direction'] == -1
    neutral_periods = signals['predicted_direction'] == 0
    
    # Plot prediction direction as colored areas
    ax4.fill_between(signals.index, 0, 1, where=bullish_periods, 
                     alpha=0.4, color='green', label='Bullish Prediction')
    ax4.fill_between(signals.index, -1, 0, where=bearish_periods, 
                     alpha=0.4, color='red', label='Bearish Prediction')
    ax4.fill_between(signals.index, -0.1, 0.1, where=neutral_periods, 
                     alpha=0.2, color='gray', label='Neutral Prediction')
    
    # Overlay model confidence as line intensity
    confidence_line = signals['model_confidence'].fillna(0)
    ax4.plot(signals.index, confidence_line, color='black', linewidth=2, 
             alpha=0.7, label='Model Confidence')
    
    # Mark prediction reversals
    prediction_changes = signals['predicted_direction'].diff() != 0
    reversal_points = signals[prediction_changes & (signals['predicted_direction'] != 0)]
    
    for idx in reversal_points.index:
        direction = reversal_points.loc[idx, 'predicted_direction']
        color = 'green' if direction == 1 else 'red'
        marker = '▲' if direction == 1 else '▼'
        ax4.scatter(idx, direction * 0.8, color=color, s=100, marker=marker, 
                   zorder=10, edgecolor='black', linewidth=1)
    
    # Add horizontal reference lines
    ax4.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=1)
    ax4.axhline(y=1, color='green', linestyle='--', alpha=0.5, linewidth=1)
    ax4.axhline(y=-1, color='red', linestyle='--', alpha=0.5, linewidth=1)
    
    ax4.set_title('Model Predictions & Confidence')
    ax4.set_ylabel('Prediction Direction\n(+1=Bull, -1=Bear)')
    ax4.set_xlabel('Date')
    ax4.set_ylim(-1.2, 1.2)
    ax4.legend(loc='upper right')
    ax4.grid(True, alpha=0.3)
    
    # Format x-axis for better readability
    import matplotlib.dates as mdates
    for ax in [ax1, ax2, ax3, ax4]:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    plt.tight_layout()
    plt.show()

# Main execution
if __name__ == "__main__":
    # Download data
    print("Downloading stock data...")
    symbol = "SPY"  # S&P 500 ETF
    data = yf.download(symbol, start="2020-01-01", end="2024-01-01")
    
    print(f"Downloaded {len(data)} days of data for {symbol}")
    
    # Initialize GPR model with parameters adjusted for 15-minute data
    gpr = GaussianProcessRegression(
        window=50,     # Reduced for shorter timeframes (was 100)
        forecast=10,   # Reduced forecast length (was 20)
        length=10.0,   # Reduced smoothing (was 20.0)
        sigma=0.01     # Keep noise parameter same
    )
    
    # Initialize trading strategy with parameters adjusted for crypto/short timeframes
    strategy = GPRTradingStrategy(
        gpr_model=gpr,
        min_prediction_strength=0.001,  # Reduced threshold for more sensitivity (was 0.002)
        stop_loss_pct=0.01,            # Tighter stop loss for short timeframes (was 0.02)
        take_profit_pct=0.02,          # Tighter take profit (was 0.04)
        min_hold_days=1                # Reduced minimum hold time (was 3)
    )
    
    print("Generating trading signals...")
    signals = strategy.generate_signals(data)
    
    print("Backtesting strategy...")
    performance = backtest_strategy(signals)
    
    # Print results
    print("\n" + "="*50)
    print("BACKTEST RESULTS")
    print("="*50)
    print(f"Symbol: {symbol}")
    print(f"Period: {data.index[0].strftime('%Y-%m-%d')} to {data.index[-1].strftime('%Y-%m-%d')}")
    print(f"Total Return: {performance['total_return']:.2%}")
    print(f"Buy & Hold Return: {performance['buy_hold_return']:.2%}")
    print(f"Excess Return: {performance['total_return'] - performance['buy_hold_return']:.2%}")
    print(f"Number of Trades: {performance['num_trades']}")
    print(f"Win Rate: {performance['win_rate']:.2%}")
    print(f"Average Return per Trade: {performance['avg_return_per_trade']:.2%}")
    print(f"Volatility (Annualized): {performance['volatility']:.2%}")
    print(f"Sharpe Ratio: {performance['sharpe_ratio']:.2f}")
    print(f"Maximum Drawdown: {performance['max_drawdown']:.2%}")
    
    # Plot results
    print("\nGenerating plots...")
    plot_results(data, signals)
    
    # Additional analysis
    print("\n" + "="*50)
    print("ADDITIONAL ANALYSIS")
    print("="*50)
    
    # Trade distribution
    trade_returns = signals[signals['returns'] != 0]['returns']
    if len(trade_returns) > 0:
        print(f"Best Trade: {trade_returns.max():.2%}")
        print(f"Worst Trade: {trade_returns.min():.2%}")
        print(f"Median Trade: {trade_returns.median():.2%}")
        
        # Monthly performance
        monthly_returns = signals.resample('M')['strategy_returns'].sum()
        positive_months = (monthly_returns > 0).sum()
        total_months = len(monthly_returns[monthly_returns != 0])
        print(f"Positive Months: {positive_months}/{total_months} ({positive_months/total_months:.1%})")