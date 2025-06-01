import numpy as np
import pandas as pd
import yfinance as yf
from stable_baselines3 import PPO
from trading_ai import TradingEnvironment, fetch_data
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import time

def load_model(model_path="ai/trading_model"):
    """Load the trained model"""
    return PPO.load(model_path)

def backtest(model, start_date, end_date):
    """Run backtest on the model"""
    # Fetch data for backtest period
    df = fetch_data(start_date=start_date, end_date=end_date)
    print(f"\nBacktest Period: {df.index[0]} to {df.index[-1]}")
    print(f"Number of candles: {len(df)}")
    
    # Create environment
    env = TradingEnvironment(df)
    
    # Initialize tracking variables
    total_trades = 0
    winning_trades = 0
    total_profit = 0
    max_drawdown = 0
    current_drawdown = 0
    peak_value = env.initial_balance
    
    # Initialize lists for plotting
    portfolio_values = []
    returns = []
    prices = []
    actions = []
    
    # Run backtest
    obs, _ = env.reset()  # Unpack observation and info
    done = False
    
    while not done:
        action, _ = model.predict(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        # Store data for plotting
        portfolio_values.append(env.total_value)
        returns.append(env.returns)
        prices.append(env.df['Close'].iloc[env.current_step])
        actions.append(action)
        
        # Update statistics
        if info.get('trade_executed', False):
            total_trades += 1
            if info.get('trade_profit', 0) > 0:
                winning_trades += 1
            total_profit += info.get('trade_profit', 0)
        
        # Update drawdown
        current_value = env.total_value
        if current_value > peak_value:
            peak_value = current_value
        current_drawdown = (peak_value - current_value) / peak_value
        max_drawdown = max(max_drawdown, current_drawdown)
    
    # Calculate statistics
    win_rate = winning_trades / total_trades if total_trades > 0 else 0
    total_return = (env.total_value - env.initial_balance) / env.initial_balance
    
    # Print results
    print("\nBacktest Results:")
    print(f"Total Trades: {total_trades}")
    print(f"Winning Trades: {winning_trades}")
    print(f"Win Rate: {win_rate:.2%}")
    print(f"Total Profit: ${total_profit:.2f}")
    print(f"Total Return: {total_return:.2%}")
    print(f"Maximum Drawdown: {max_drawdown:.2%}")
    print(f"Final Portfolio Value: ${env.total_value:.2f}")
    
    # Plot results
    plt.figure(figsize=(15, 12))
    
    # Plot portfolio value
    plt.subplot(3, 1, 1)
    plt.plot(df.index[:len(portfolio_values)], portfolio_values)
    plt.title('Portfolio Value Over Time')
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value ($)')
    plt.grid(True)
    
    # Plot returns
    plt.subplot(3, 1, 2)
    plt.plot(df.index[:len(returns)], [r * 100 for r in returns])
    plt.title('Returns Over Time')
    plt.xlabel('Date')
    plt.ylabel('Returns (%)')
    plt.grid(True)
    
    # Plot price and actions
    plt.subplot(3, 1, 3)
    plt.plot(df.index[:len(prices)], prices, label='Price')
    
    # Plot buy/sell signals
    buy_signals = [i for i, a in enumerate(actions) if a == 1]
    sell_signals = [i for i, a in enumerate(actions) if a == 2]
    
    if buy_signals:
        plt.scatter(df.index[buy_signals], [prices[i] for i in buy_signals], 
                   color='green', marker='^', label='Buy Signal')
    if sell_signals:
        plt.scatter(df.index[sell_signals], [prices[i] for i in sell_signals], 
                   color='red', marker='v', label='Sell Signal')
    
    plt.title('XRP Price and Trading Signals')
    plt.xlabel('Date')
    plt.ylabel('Price ($)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    return {
        'total_trades': total_trades,
        'winning_trades': winning_trades,
        'win_rate': win_rate,
        'total_profit': total_profit,
        'total_return': total_return,
        'max_drawdown': max_drawdown,
        'final_value': env.total_value,
        'portfolio_values': portfolio_values,
        'returns': returns,
        'prices': prices,
        'actions': actions
    }

def live_trading(model, check_interval_minutes=15):
    """Run live trading with the model"""
    print("Starting live trading...")
    print("Press Ctrl+C to stop")
    
    try:
        while True:
            # Fetch latest data
            ticker = yf.Ticker('XRP-USD')
            df = ticker.history(period='1d', interval='15m')
            df = df.dropna()
            
            # Create environment with latest data
            env = TradingEnvironment(df)
            obs = env.reset()
            
            # Get model's prediction
            action, _ = model.predict(obs)
            
            # Execute trade based on action
            if action == 1:  # Buy
                print(f"[{datetime.now()}] BUY Signal")
            elif action == 2:  # Sell
                print(f"[{datetime.now()}] SELL Signal")
            else:
                print(f"[{datetime.now()}] HOLD Signal")
            
            # Wait for next interval
            time.sleep(check_interval_minutes * 60)
            
    except KeyboardInterrupt:
        print("\nStopping live trading...")

if __name__ == "__main__":
    # Load the trained model
    model = load_model()
    
    # Example: Run backtest for the last 30 days
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    results = backtest(model, start_date, end_date)
    
    # Uncomment to run live trading
    # live_trading(model) 