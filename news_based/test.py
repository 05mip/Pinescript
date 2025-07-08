import pandas as pd
import numpy as np
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
import yfinance as yf

class IntradayStrategy(Strategy):
    """
    Intraday strategy with MACD filter:
    - Only buys when MACD is trending up
    - Buys at the beginning of each day (at open)
    - Sells at the end of each day (at close)
    """
    
    def init(self):
        # Calculate MACD using pandas
        close_prices = self.data.Close
        self.macd = self.I(self.calculate_macd, close_prices)
    
    def calculate_macd(self, prices, fast=12, slow=26, signal=9):
        """Calculate MACD using pandas"""
        exp1 = prices.ewm(span=fast).mean()
        exp2 = prices.ewm(span=slow).mean()
        macd_line = exp1 - exp2
        signal_line = macd_line.ewm(span=signal).mean()
        return macd_line
    
    def next(self):
        # Get current position
        position = self.position
        
        # Check if MACD is trending up (current MACD > previous MACD)
        macd_trending_up = self.macd[-1] > self.macd[-2] if len(self.macd) > 1 and not np.isnan(self.macd[-1]) and not np.isnan(self.macd[-2]) else False
        
        # If we don't have a position and MACD is trending up, buy at the beginning of the day
        if not position and macd_trending_up:
            # Buy at the current bar's open price
            self.buy()
        
        # If we have a position, sell at the end of the day
        elif position:
            # Sell at the current bar's close price
            self.sell()

def load_and_prepare_data():
    """Load and prepare the XRP price data from yfinance"""
    print("Fetching XRP data from yfinance...")
    
    # Get XRP data from yfinance for the last year
    ticker = yf.Ticker("XRP-USD")
    df = ticker.history(period="1y", interval="1d")
    
    # Keep only the columns needed for backtesting
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
    
    # Drop any rows with NaN values
    df = df.dropna()
    
    return df

def main():
    print("Loading and preparing data...")
    data = load_and_prepare_data()
    
    print(f"Data loaded: {len(data)} days of XRP price data")
    print(f"Date range: {data.index.min()} to {data.index.max()}")
    print(f"Price range: ${data['Close'].min():.2f} to ${data['Close'].max():.2f}")
    
    print("\nRunning backtest...")
    
    # Run the backtest
    bt = Backtest(
        data, 
        IntradayStrategy, 
        cash=10000,  # Starting cash
        commission=0.001,  # 0.1% commission
        exclusive_orders=True
    )
    
    # Run the backtest
    stats = bt.run()
    
    # Print results
    print("\n=== BACKTEST RESULTS ===")
    print(f"Total Return: {stats['Return [%]']:.2f}%")
    print(f"Buy & Hold Return: {stats['Buy & Hold Return [%]']:.2f}%")
    print(f"Max Drawdown: {stats['Max. Drawdown [%]']:.2f}%")
    print(f"Sharpe Ratio: {stats['Sharpe Ratio']:.2f}")
    print(f"Total Trades: {stats['# Trades']}")
    print(f"Win Rate: {stats['Win Rate [%]']:.2f}%")
    print(f"Profit Factor: {stats['Profit Factor']:.2f}")
    print(f"Final Portfolio Value: ${stats['Equity Final [$]']:.2f}")
    
    # Plot the results
    print("\nGenerating plot...")
    bt.plot()
    
    return stats

if __name__ == "__main__":
    main()
