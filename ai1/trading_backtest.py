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
    executed_actions = []  # Track actual executed actions
    attempted_actions = []  # Track what the model tried to do
    
    # Run backtest
    obs, _ = env.reset()
    done = False
    
    while not done:
        # Get model's prediction
        predicted_action, _ = model.predict(obs)
        attempted_actions.append(predicted_action)
        
        # Store previous state to detect actual trades
        prev_position = env.position
        prev_shares = env.shares_owned
        prev_trade_count = env.trade_count
        
        # Execute step
        obs, reward, terminated, truncated, info = env.step(predicted_action)
        done = terminated or truncated
        
        # Determine what action was actually executed
        actual_action = 0  # Default to hold
        if info.get('trade_executed', False):
            if env.trade_count > prev_trade_count:  # A trade was executed
                # Check if it was a buy or sell
                if prev_position is None and env.position == 'long':
                    actual_action = 1  # Buy was executed
                elif prev_position == 'long' and env.position is None:
                    actual_action = 2  # Sell was executed
        
        executed_actions.append(actual_action)
        
        # Store data for plotting
        portfolio_values.append(env.total_value)
        returns.append(env.returns)
        prices.append(env.df['Close'].iloc[env.current_step])
        
        # Update statistics
        if info.get('trade_executed', False):
            total_trades += 1
            # Calculate profit from closed trades
            if len(env.closed_trades) > 0:
                last_trade = env.closed_trades[-1]
                trade_profit = last_trade['profit']
                if trade_profit > 0:
                    winning_trades += 1
                total_profit += trade_profit
        
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
    
    # Print action statistics
    attempted_buys = sum(1 for a in attempted_actions if a == 1)
    attempted_sells = sum(1 for a in attempted_actions if a == 2)
    executed_buys = sum(1 for a in executed_actions if a == 1)
    executed_sells = sum(1 for a in executed_actions if a == 2)
    
    print(f"\nAction Statistics:")
    print(f"Attempted Buy Signals: {attempted_buys}")
    print(f"Executed Buy Trades: {executed_buys}")
    print(f"Buy Execution Rate: {executed_buys/attempted_buys*100:.1f}%" if attempted_buys > 0 else "Buy Execution Rate: N/A")
    print(f"Attempted Sell Signals: {attempted_sells}")
    print(f"Executed Sell Trades: {executed_sells}")
    print(f"Sell Execution Rate: {executed_sells/attempted_sells*100:.1f}%" if attempted_sells > 0 else "Sell Execution Rate: N/A")
    
    # Plot results
    plt.figure(figsize=(15, 15))
    
    # Plot portfolio value
    plt.subplot(4, 1, 1)
    plt.plot(df.index[:len(portfolio_values)], portfolio_values)
    plt.title('Portfolio Value Over Time')
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value ($)')
    plt.grid(True)
    
    # Plot returns
    plt.subplot(4, 1, 2)
    plt.plot(df.index[:len(returns)], [r * 100 for r in returns])
    plt.title('Returns Over Time')
    plt.xlabel('Date')
    plt.ylabel('Returns (%)')
    plt.grid(True)
    
    # Plot price and EXECUTED actions (this is the key fix)
    plt.subplot(4, 1, 3)
    plt.plot(df.index[:len(prices)], prices, label='Price', alpha=0.7)
    
    # Plot only executed buy/sell signals
    executed_buy_signals = [i for i, a in enumerate(executed_actions) if a == 1]
    executed_sell_signals = [i for i, a in enumerate(executed_actions) if a == 2]
    
    if executed_buy_signals:
        plt.scatter([df.index[i] for i in executed_buy_signals], 
                   [prices[i] for i in executed_buy_signals], 
                   color='green', marker='^', s=100, label='Executed Buy', zorder=5)
    if executed_sell_signals:
        plt.scatter([df.index[i] for i in executed_sell_signals], 
                   [prices[i] for i in executed_sell_signals], 
                   color='red', marker='v', s=100, label='Executed Sell', zorder=5)
    
    plt.title('XRP Price and EXECUTED Trading Signals')
    plt.xlabel('Date')
    plt.ylabel('Price ($)')
    plt.legend()
    plt.grid(True)
    
    # Plot comparison of attempted vs executed actions
    plt.subplot(4, 1, 4)
    time_indices = range(len(attempted_actions))
    
    # Plot attempted actions as light colored background
    attempted_buy_indices = [i for i, a in enumerate(attempted_actions) if a == 1]
    attempted_sell_indices = [i for i, a in enumerate(attempted_actions) if a == 2]
    
    if attempted_buy_indices:
        plt.scatter(attempted_buy_indices, [1] * len(attempted_buy_indices), 
                   color='lightgreen', marker='^', s=30, alpha=0.5, label='Attempted Buy')
    if attempted_sell_indices:
        plt.scatter(attempted_sell_indices, [2] * len(attempted_sell_indices), 
                   color='lightcoral', marker='v', s=30, alpha=0.5, label='Attempted Sell')
    
    # Plot executed actions as solid colored foreground
    if executed_buy_signals:
        plt.scatter(executed_buy_signals, [1] * len(executed_buy_signals), 
                   color='green', marker='^', s=60, label='Executed Buy')
    if executed_sell_signals:
        plt.scatter(executed_sell_signals, [2] * len(executed_sell_signals), 
                   color='red', marker='v', s=60, label='Executed Sell')
    
    plt.title('Attempted vs Executed Actions')
    plt.xlabel('Time Step')
    plt.ylabel('Action Type')
    plt.yticks([0, 1, 2], ['Hold', 'Buy', 'Sell'])
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
        'attempted_actions': attempted_actions,
        'executed_actions': executed_actions
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
            obs, _ = env.reset()
            
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