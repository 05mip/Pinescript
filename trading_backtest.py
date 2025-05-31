import numpy as np
import pandas as pd
import yfinance as yf
from stable_baselines3 import PPO
from trading_ai import TradingEnvironment
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import time

def load_model(model_path="trading_model"):
    """Load the trained model"""
    return PPO.load(model_path)

def backtest(model, start_date, end_date=None, initial_balance=10000):
    """Backtest the model on historical data"""
    # Fetch data for backtesting
    ticker = yf.Ticker('XRP-USD')
    df = ticker.history(start=start_date, end=end_date, interval='15m')
    df = df.dropna()
    
    print(f"\nBacktest Period: {df.index[0]} to {df.index[-1]}")
    print(f"Number of candles: {len(df)}")
    
    # Create environment
    env = TradingEnvironment(df, initial_balance=initial_balance)
    
    # Run backtest
    obs = env.reset()
    done = False
    actions = []
    portfolio_values = []
    returns = []
    prices = []
    
    while not done:
        action, _ = model.predict(obs)
        obs, reward, done, _ = env.step(action)
        
        actions.append(action)
        portfolio_values.append(env.total_value)
        returns.append(env.returns)
        prices.append(env.df['Close'].iloc[env.current_step])
    
    # Calculate performance metrics
    portfolio_values = np.array(portfolio_values)
    returns = np.array(returns)
    prices = np.array(prices)
    
    # Print action distribution
    action_counts = np.bincount(actions)
    print("\nAction Distribution:")
    print(f"Hold: {action_counts[0] if 0 < len(action_counts) else 0}")
    print(f"Buy: {action_counts[1] if 1 < len(action_counts) else 0}")
    print(f"Sell: {action_counts[2] if 2 < len(action_counts) else 0}")
    
    # Calculate metrics
    total_return = (portfolio_values[-1] - initial_balance) / initial_balance * 100
    
    # Handle edge cases for Sharpe ratio calculation
    if len(returns) > 1 and np.std(returns) != 0:
        sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252 * 24 * 4)  # Annualized
    else:
        sharpe_ratio = 0
        print("\nWarning: Could not calculate Sharpe ratio (insufficient data or zero variance)")
    
    max_drawdown = np.min(returns) * 100 if len(returns) > 0 else 0
    
    # Plot results
    plt.figure(figsize=(15, 15))
    
    # Plot portfolio value
    plt.subplot(3, 1, 1)
    plt.plot(df.index[:len(portfolio_values)], portfolio_values)
    plt.title('Portfolio Value Over Time')
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value ($)')
    
    # Plot returns
    plt.subplot(3, 1, 2)
    plt.plot(df.index[:len(returns)], returns * 100)
    plt.title('Returns Over Time')
    plt.xlabel('Date')
    plt.ylabel('Returns (%)')
    
    # Plot price
    plt.subplot(3, 1, 3)
    plt.plot(df.index[:len(prices)], prices)
    plt.title('XRP Price Over Time')
    plt.xlabel('Date')
    plt.ylabel('Price ($)')
    
    plt.tight_layout()
    plt.savefig('backtest_results.png')
    
    print(f"\nBacktest Results:")
    print(f"Total Return: {total_return:.2f}%")
    print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
    print(f"Maximum Drawdown: {max_drawdown:.2f}%")
    print(f"Final Portfolio Value: ${portfolio_values[-1]:.2f}")
    print(f"Initial Portfolio Value: ${portfolio_values[0]:.2f}")
    
    return {
        'total_return': total_return,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'final_value': portfolio_values[-1],
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