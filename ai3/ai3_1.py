import pandas as pd
import numpy as np
from backtesting import Backtest, Strategy
import ai3 as ai3_module

# Enhanced strategy with multiple exit conditions
class ImprovedMLStrategy(Strategy):
    # Strategy parameters
    profit_target = 0.05  # 5% profit target
    stop_loss = 0.02      # 2% stop loss
    trailing_stop = 0.03  # 3% trailing stop
    max_hold_days = 10    # Maximum holding period
    
    def init(self):
        self.pred = self.data.df['Prediction'].values
        self.entry_price = None
        self.entry_time = None
        self.highest_price = None
        
    def next(self):
        i = len(self.data) - 1
        current_price = self.data.Close[i]
        signal = self.pred[i]
        
        # Entry logic
        if signal == 2 and not self.position:  # Buy signal
            self.buy()
            self.entry_price = current_price
            self.entry_time = i
            self.highest_price = current_price
            
        # Exit logic when in position
        elif self.position:
            # Update highest price for trailing stop
            if current_price > self.highest_price:
                self.highest_price = current_price
            
            # Calculate returns and time held
            unrealized_return = (current_price - self.entry_price) / self.entry_price
            time_held = i - self.entry_time
            
            # Exit conditions (in order of priority)
            should_exit = False
            exit_reason = ""
            
            # 1. Model says sell
            if signal == 0:
                should_exit = True
                exit_reason = "model_sell"
            
            # 2. Profit target hit
            elif unrealized_return >= self.profit_target:
                should_exit = True
                exit_reason = "profit_target"
            
            # 3. Stop loss hit
            elif unrealized_return <= -self.stop_loss:
                should_exit = True
                exit_reason = "stop_loss"
            
            # 4. Trailing stop hit
            elif (self.highest_price - current_price) / self.highest_price >= self.trailing_stop:
                should_exit = True
                exit_reason = "trailing_stop"
            
            # 5. Maximum holding period reached
            elif time_held >= self.max_hold_days:
                should_exit = True
                exit_reason = "max_hold"
            
            if should_exit:
                self.position.close()
                # Optional: print exit reason for debugging
                # print(f"Exit at {current_price:.2f}, reason: {exit_reason}, return: {unrealized_return:.2%}")


# Alternative: Multi-timeframe strategy
class MultiTimeframeStrategy(Strategy):
    def init(self):
        self.pred = self.data.df['Prediction'].values
        self.rsi = self.data.df['RSI'].values if 'RSI' in self.data.df.columns else None
        self.macd = self.data.df['MACD'].values if 'MACD' in self.data.df.columns else None
        
    def next(self):
        i = len(self.data) - 1
        current_price = self.data.Close[i]
        signal = self.pred[i]
        
        # Entry with confirmation
        if signal == 2 and not self.position:
            # Additional confirmation filters
            rsi_ok = self.rsi is None or self.rsi[i] < 70  # Not overbought
            macd_ok = self.macd is None or self.macd[i] > 0  # MACD positive
            
            if rsi_ok and macd_ok:
                self.buy()
        
        # Exit logic
        elif self.position:
            # Quick exit on strong sell signal + technical confirmation
            if signal == 0:
                rsi_sell = self.rsi is None or self.rsi[i] > 70  # Overbought
                if rsi_sell:
                    self.position.close()
            
            # Or use the improved exit logic from above
            # (You can combine both approaches)


# Strategy with prediction smoothing
class SmoothedMLStrategy(Strategy):
    lookback = 3  # Number of periods to look back for signal confirmation
    
    def init(self):
        self.pred = self.data.df['Prediction'].values
        
    def get_smoothed_signal(self, i):
        """Get smoothed signal based on recent predictions"""
        if i < self.lookback:
            return 1  # Hold if not enough history
        
        recent_preds = self.pred[i-self.lookback:i+1]
        
        # Count buy/sell signals in recent periods
        buy_count = sum(1 for p in recent_preds if p == 2)
        sell_count = sum(1 for p in recent_preds if p == 0)
        
        # Only act on strong signals
        if buy_count >= 2:  # At least 2 buy signals in last 3 periods
            return 2
        elif sell_count >= 2:  # At least 2 sell signals in last 3 periods
            return 0
        else:
            return 1  # Hold
    
    def next(self):
        i = len(self.data) - 1
        signal = self.get_smoothed_signal(i)
        
        if signal == 2 and not self.position:
            self.buy()
        elif signal == 0 and self.position:
            self.position.close()


# Advanced strategy with dynamic position sizing
class DynamicPositionStrategy(Strategy):
    def init(self):
        self.pred = self.data.df['Prediction'].values
        # Get prediction probabilities if available
        self.buy_prob = self.data.df.get('Buy_Prob', pd.Series([0.33] * len(self.data))).values
        self.sell_prob = self.data.df.get('Sell_Prob', pd.Series([0.33] * len(self.data))).values
        
    def next(self):
        i = len(self.data) - 1
        signal = self.pred[i]
        buy_confidence = self.buy_prob[i]
        sell_confidence = self.sell_prob[i]
        
        # Entry with confidence-based position sizing
        if signal == 2 and not self.position:
            # Size position based on model confidence
            if buy_confidence > 0.7:  # High confidence
                size = 1.0
            elif buy_confidence > 0.5:  # Medium confidence
                size = 0.7
            else:  # Low confidence
                size = 0.5
            
            self.buy(size=size)
        
        # Exit with confidence threshold
        elif self.position and signal == 0:
            if sell_confidence > 0.4:  # Only exit if reasonably confident
                self.position.close()

class MLStrategy(Strategy):
    def init(self):
        self.pred = self.data.df['Prediction'].values
        self.time_ = self.data.df['Close'].values
    def next(self):
        # print("here")
        i = len(self.data) - 1
        signal = self.pred[i]
        # print(signal)
        if signal == 2.0 and not self.position:  # Buy
                self.buy()
        elif signal == 0.0 and self.position:  # Sell
            if not self.position.is_short:
                self.position.close()


# Function to run backtests with different strategies
def run_multiple_backtests(bt_df, strategies, cash=1000000, commission=0.001):
    """Run backtests with different strategies and compare results"""
    results = {}
    
    for name, strategy_class in strategies.items():
        print(f"\nRunning {name} strategy...")
        try:
            bt = Backtest(bt_df, strategy_class, cash=cash, commission=commission)
            stats = bt.run()
            results[name] = stats
            
            print(stats)
            
        except Exception as e:
            print(f"Error running {name}: {e}")
            
    return results

import pandas as pd
import numpy as np
from backtesting import Backtest, Strategy
import ai3 as ai3_module

def deep_data_diagnostic():
    """Deep dive into the data to find the issue"""
    
    print("=== STEP 1: Loading and generating signals ===")
    
    # Load your data (same as original)
    trader, best_model, results = ai3_module.main()
    df = trader.load_all_datasets()
    X, y, dates = trader.prepare_training_data(df)
    feature_cols = X.columns.tolist()
    
    print(f"Original df shape: {df.shape}")
    print(f"Original df columns: {df.columns.tolist()}")
    print(f"Original df index type: {type(df.index)}")
    
    # Generate signals
    signals = trader.predict_trading_signals(df, best_model, feature_cols)
    print(f"Signals shape: {signals.shape}")
    print(f"Signals columns: {signals.columns.tolist()}")
    print(f"Signals index type: {type(signals.index)}")
    
    # Check signals content
    print(f"\nSignals prediction distribution:")
    print(signals['Prediction'].value_counts().sort_index())
    
    print(f"\nFirst 10 signals:")
    print(signals.head(10))
    
    print(f"\nLast 10 signals:")
    print(signals.tail(10))
    
    # Set index for joining
    signals = signals.set_index('Datetime')
    df = df.set_index('Datetime')
    
    print(f"\n=== STEP 2: Joining data ===")
    print(f"DF index range: {df.index.min()} to {df.index.max()}")
    print(f"Signals index range: {signals.index.min()} to {signals.index.max()}")
    
    # Join the data
    df = df.join(signals[['Prediction']], how='left')
    print(f"After join - df shape: {df.shape}")
    print(f"After join - prediction nulls: {df['Prediction'].isnull().sum()}")
    
    # Fill NaN with 1 (hold)
    df['Prediction'] = df['Prediction'].fillna(1)
    print(f"After fillna - prediction distribution:")
    print(df['Prediction'].value_counts().sort_index())
    
    print(f"\n=== STEP 3: Filtering for symbol ===")
    symbol = 'BTC-USD'
    df_symbol = df[df['Symbol'] == symbol].copy()
    print(f"Symbol filtered shape: {df_symbol.shape}")
    print(f"Symbol prediction distribution:")
    print(df_symbol['Prediction'].value_counts().sort_index())
    
    print(f"\n=== STEP 4: Preparing backtest data ===")
    print(f"Available columns: {df_symbol.columns.tolist()}")
    
    # Check if we have the required columns
    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'Prediction']
    missing_cols = [col for col in required_cols if col not in df_symbol.columns]
    if missing_cols:
        print(f"MISSING COLUMNS: {missing_cols}")
    
    # Create backtest DataFrame
    bt_df = df_symbol[['Open', 'High', 'Low', 'Close', 'Volume', 'Prediction']].copy()
    print(f"Before dropna: {len(bt_df)} rows")
    
    # Check for NaN values before dropping
    print(f"\nNaN counts before dropna:")
    for col in bt_df.columns:
        nan_count = bt_df[col].isnull().sum()
        print(f"  {col}: {nan_count}")
    
    bt_df = bt_df.dropna()
    print(f"After dropna: {len(bt_df)} rows")
    
    if len(bt_df) == 0:
        print("ERROR: All data was dropped! Investigating...")
        # Don't dropna, just show what's happening
        bt_df_debug = df_symbol[['Open', 'High', 'Low', 'Close', 'Volume', 'Prediction']].copy()
        print(f"Sample of data with NaN:")
        print(bt_df_debug.head(20))
        return None
    
    print(f"\nFinal backtest data prediction distribution:")
    print(bt_df['Prediction'].value_counts().sort_index())
    
    print(f"\nBacktest data date range: {bt_df.index.min()} to {bt_df.index.max()}")
    
    # Show sample of actual data going into backtest
    print(f"\nSample of backtest data:")
    print(bt_df.head(10))
    
    # Check data types
    print(f"\nData types:")
    print(bt_df.dtypes)
    
    return bt_df

# Simple test strategy with maximum debugging
class MaxDebugStrategy(Strategy):
    def init(self):
        print(f"=== STRATEGY INIT ===")
        print(f"Data shape: {self.data.df.shape}")
        print(f"Data columns: {self.data.df.columns.tolist()}")
        print(f"Data index: {self.data.df.index}")
        
        if 'Prediction' in self.data.df.columns:
            self.pred = self.data.df['Prediction'].values
            print(f"Predictions loaded: {len(self.pred)} values")
            print(f"Prediction distribution: {np.bincount(self.pred.astype(int))}")
            print(f"First 10 predictions: {self.pred[:10]}")
        else:
            print("ERROR: No Prediction column found!")
            self.pred = np.ones(len(self.data.df))  # Default to hold
    
    def next(self):
        i = len(self.data) - 1
        signal = self.pred[i]
        current_price = self.data.Close[i]
        
        # Print everything for first 20 bars
        if i < 20:
            print(f"Bar {i}: Date={self.data.index[i]}, Price={current_price:.2f}, Signal={signal}, Position={bool(self.position)}")
        
        # Log all trading decisions
        if signal == 2 and not self.position:
            print(f">>> BUY DECISION: Bar {i}, Date={self.data.index[i]}, Price={current_price:.2f}")
            self.buy()
        elif signal == 0 and self.position:
            print(f">>> SELL DECISION: Bar {i}, Date={self.data.index[i]}, Price={current_price:.2f}")
            self.position.close()

def run_data_diagnostic():
    """Run the complete data diagnostic"""
    
    bt_df = deep_data_diagnostic()
    
    if bt_df is None:
        print("Cannot proceed with backtest - data issue found")
        return
    
    print(f"\n=== RUNNING DEBUG BACKTEST ===")
    
    try:
        bt = Backtest(bt_df, MaxDebugStrategy, cash=1000000, commission=0.001)
        stats = bt.run()
        
        print(f"\n=== BACKTEST RESULTS ===")
        print(f"# Trades: {stats['# Trades']}")
        print(f"Return: {stats['Return [%]']:.2f}%")
        print(f"Exposure Time: {stats['Exposure Time [%]']:.1f}%")
        
        if stats['# Trades'] == 0:
            print("\nSTILL NO TRADES - Let's check the strategy received the right data...")
            # The MaxDebugStrategy will have printed what it received
        
    except Exception as e:
        print(f"Backtest failed: {e}")
        import traceback
        traceback.print_exc()

# Alternative: Let's also test with completely artificial data
def test_with_artificial_data():
    """Test backtest with artificial data to verify the strategy works"""
    
    print(f"\n=== TESTING WITH ARTIFICIAL DATA ===")
    
    # Create artificial data with known buy/sell signals
    dates = pd.date_range('2025-01-01', periods=100, freq='H')
    
    # Create price data
    prices = 100 + np.cumsum(np.random.randn(100) * 0.1)
    
    # Create artificial signals: buy every 10 bars, sell after 5 bars
    signals = []
    for i in range(100):
        if i % 10 == 0:  # Buy signal
            signals.append(2)
        elif i % 10 == 5:  # Sell signal
            signals.append(0)
        else:  # Hold
            signals.append(1)
    
    # Create DataFrame
    artificial_df = pd.DataFrame({
        'Open': prices,
        'High': prices * 1.01,
        'Low': prices * 0.99,
        'Close': prices,
        'Volume': np.random.randint(1000, 10000, 100),
        'Prediction': signals
    }, index=dates)
    
    print(f"Artificial data shape: {artificial_df.shape}")
    print(f"Artificial prediction distribution:")
    print(artificial_df['Prediction'].value_counts().sort_index())
    
    # Test with artificial data
    bt = Backtest(artificial_df, MaxDebugStrategy, cash=1000000, commission=0.001)
    stats = bt.run()
    
    print(f"\n=== ARTIFICIAL DATA RESULTS ===")
    print(f"# Trades: {stats['# Trades']}")
    print(f"Return: {stats['Return [%]']:.2f}%")
    
    if stats['# Trades'] > 0:
        print("SUCCESS: Strategy works with artificial data!")
        print("The issue is with your real data preparation.")
    else:
        print("PROBLEM: Strategy doesn't work even with artificial data!")

# if __name__ == "__main__":
#     # Run full diagnostic
#     run_data_diagnostic()
    
#     # Test with artificial data
#     test_with_artificial_data()
# Usage example
if __name__ == "__main__":
    # Load your data (same as your original code)
    trader, best_model, results = ai3_module.main()
    df = trader.load_all_datasets()
    X, y, dates = trader.prepare_training_data(df)
    feature_cols = X.columns.tolist()
    
    signals = trader.predict_trading_signals(df, best_model, feature_cols)
    signals = signals.set_index('Datetime')
    
    df = df.set_index('Datetime')
    df = df.join(signals[['Prediction', 'Buy_Prob', 'Sell_Prob']], how='left')
    df['Prediction'] = df['Prediction'].fillna(1)
    
    # Filter for symbol
    symbol = 'BTC-USD'
    df_symbol = df[df['Symbol'] == symbol].copy()
    
    # Add technical indicators for multi-timeframe strategy
    if 'RSI' in df_symbol.columns:
        bt_df = df_symbol[['Open', 'High', 'Low', 'Close', 'Volume', 'Prediction', 'RSI', 'MACD']].copy()
    else:
        bt_df = df_symbol[['Open', 'High', 'Low', 'Close', 'Volume', 'Prediction']].copy()
    
    # Add probability columns if available
    if 'Buy_Prob' in df_symbol.columns:
        bt_df['Buy_Prob'] = df_symbol['Buy_Prob']
        bt_df['Sell_Prob'] = df_symbol['Sell_Prob']
    
    bt_df = bt_df.dropna()

    
    # Run all strategies
    strategies = {
        'Original': MLStrategy,
        'Improved': ImprovedMLStrategy,
        'Multi-timeframe': MultiTimeframeStrategy,
        'Smoothed': SmoothedMLStrategy,
        'Dynamic': DynamicPositionStrategy
    }
    all_results = run_multiple_backtests(bt_df, strategies)
    
    # Find best strategy
    best_strategy = max(all_results.keys(), key=lambda k: all_results[k]['Return [%]'])
    print(f"\nBest strategy: {best_strategy}")
    
    # Plot best strategy
    bt = Backtest(bt_df, strategies[best_strategy], cash=1000000, commission=0.001)
    stats = bt.run()
    bt.plot()