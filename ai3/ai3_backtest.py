import pandas as pd
import numpy as np
from backtesting import Backtest, Strategy
import ai3 as ai3_module  # Import your CryptoTradingModel and main

# 1. Load data and model (retrain for now)
trader, best_model, results = ai3_module.main()

# 2. Prepare the data (use the same as in training)
df = trader.load_all_datasets()
X, y, dates = trader.prepare_training_data(df)
feature_cols = X.columns.tolist()

# 3. Generate signals for the backtest period
signals = trader.predict_trading_signals(df, best_model, feature_cols)
signals = signals.set_index('Datetime')

# 4. Merge signals with price data for backtesting
df = df.set_index('Datetime')
df = df.join(signals[['Prediction']], how='left')
df['Prediction'] = df['Prediction'].fillna(1)  # Default to 'hold' if missing

# 5. Filter for a single symbol
symbol = 'BTC-USD'  # Change to any symbol you want to backtest
df_symbol = df[df['Symbol'] == symbol].copy()

# 6. Prepare Backtesting.py DataFrame
bt_df = df_symbol[['Open', 'High', 'Low', 'Close', 'Volume', 'Prediction']].copy()
bt_df = bt_df.dropna()

# 7. Define a Backtesting.py strategy
class MLStrategy(Strategy):
    def init(self):
        self.pred = self.data.df['Prediction'].values

    def next(self):
        i = len(self.data) - 1
        signal = self.pred[i]
        print(f"Bar {i}, Signal: {signal}, Position: {self.position}")
        if signal == 2 and not self.position:  # Buy
            self.buy()
        elif signal == 0 and self.position:  # Sell
            if not self.position.is_short:
                self.position.close()
        # Hold (1): do nothing

# 8. Run the backtest
bt = Backtest(bt_df, MLStrategy, cash=1000000, commission=.001)
stats = bt.run()
bt.plot()

print(stats) 

print("Prediction value counts in backtest data:")
print(bt_df['Prediction'].value_counts())
print("First 20 predictions:", bt_df['Prediction'].head(20).tolist()) 
