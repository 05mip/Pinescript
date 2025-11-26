import pandas as pd
import numpy as np
import pandas_ta as ta
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
import os
import warnings
warnings.filterwarnings('ignore')

# RL and backtesting imports
import gym
from gym import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
import matplotlib.pyplot as plt
from collections import deque
import random

class CryptoTradingModel:
    def __init__(self, data_folder=r'C:\Users\micha\OneDrive\Documents\CodeProjects\Pinescript\data_30m', 
                 label_folder=r'C:\Users\micha\OneDrive\Documents\CodeProjects\Pinescript\labelled_30'):
        self.data_folder = data_folder
        self.label_folder = label_folder
        self.scaler = StandardScaler()
        self.model = None
        
    def load_all_datasets(self):
        """Load and combine all 9 datasets"""
        all_data = []
        
        symbols = ['ADA-USD', 'AVAX-USD', 'BNB-USD', 'BTC-USD', 'DOGE-USD', 
                  'DOT-USD', 'ETH-USD', 'SOL-USD', 'XRP-USD']
        
        for symbol in symbols:
            try:
                print(f"Loading {symbol}...")
                df = self.load_and_prepare_data(symbol)
                df = self.create_features(df)
                df['Symbol'] = symbol
                all_data.append(df)
                print(f"  - Loaded {len(df)} rows")
            except Exception as e:
                print(f"  - Error loading {symbol}: {e}")
                continue
        
        if not all_data:
            raise ValueError("No datasets could be loaded!")
        
        combined_df = pd.concat(all_data, ignore_index=True)
        print(f"\nCombined dataset: {len(combined_df)} total rows")
        
        return combined_df
    
    def load_and_prepare_data(self, symbol):
        """Load price data and labels, merge them"""
        price_file = f"{symbol}_30m.csv"
        price_path = os.path.join(self.data_folder, price_file)
        df = pd.read_csv(price_path, header=0, skiprows=[1,2])
        df = df.rename(columns={'Price': 'Datetime'})
        df['Datetime'] = pd.to_datetime(df['Datetime'])
        
        price_cols = ['Close', 'High', 'Low', 'Open', 'Volume']
        for col in price_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        label_file = f"{symbol.replace('-', '_')}_labelled.csv"
        label_path = os.path.join(self.label_folder, label_file)
        labels = pd.read_csv(label_path)
        labels['Datetime'] = pd.to_datetime(labels['Datetime'])
        
        df = pd.merge(df, labels, on='Datetime', how='left')
        
        return df
    
    def create_features(self, df):
        """Create technical indicators and features"""
        macd_data = ta.macd(df['Close'])
        df['MACD'] = macd_data['MACD_12_26_9']
        df['MACD_Signal'] = macd_data['MACDs_12_26_9']
        df['MACD_Histogram'] = macd_data['MACDh_12_26_9']
        
        df['RSI'] = ta.rsi(df['Close'])
        
        df['MACD_Signal_Distance'] = df['MACD'] - df['MACD_Signal']
        df['MACD_Histogram_Direction'] = np.where(
            df['MACD_Histogram'] > df['MACD_Histogram'].shift(1), 1, 0
        )
        
        df['Volume_SMA'] = ta.sma(df['Volume'], length=10)
        df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA']
        df['High_Low_Ratio'] = df['High'] / df['Low']
        
        # Extended lag features
        lag_periods = [1, 2, 3, 5, 10, 15, 20, 30, 40]
        
        for lag in lag_periods:
            if lag <= len(df):
                df[f'Close_lag_{lag}'] = df['Close'].shift(lag)
                df[f'RSI_lag_{lag}'] = df['RSI'].shift(lag)
                df[f'MACD_lag_{lag}'] = df['MACD'].shift(lag)
                df[f'MACD_Signal_lag_{lag}'] = df['MACD_Signal'].shift(lag)
                df[f'MACD_Histogram_lag_{lag}'] = df['MACD_Histogram'].shift(lag)
                df[f'Volume_Ratio_lag_{lag}'] = df['Volume_Ratio'].shift(lag)
        
        df['RSI_Change'] = df['RSI'] - df['RSI'].shift(1)
        df['MACD_Signal_Cross'] = np.where(df['MACD'] > df['MACD_Signal'], 1, 0)
        
        return df
    
    def prepare_training_data(self, df):
        """Prepare features and labels for training"""
        lag_periods = [1, 2, 3, 5, 10, 15, 20, 30, 40]
        
        feature_cols = [
            'MACD', 'MACD_Signal', 'MACD_Histogram', 'RSI',
            'High_Low_Ratio', 'Volume_Ratio',
            'MACD_Signal_Distance', 'MACD_Histogram_Direction',
            'RSI_Change', 'MACD_Signal_Cross'
        ]
        
        for lag in lag_periods:
            lag_features = [
                f'Close_lag_{lag}', f'RSI_lag_{lag}', f'MACD_lag_{lag}',
                f'MACD_Signal_lag_{lag}', f'MACD_Histogram_lag_{lag}', f'Volume_Ratio_lag_{lag}'
            ]
            for feature in lag_features:
                if feature in df.columns:
                    feature_cols.append(feature)
        
        labeled_data = df.dropna(subset=['Label']).copy()
        labeled_data = labeled_data.dropna(subset=feature_cols)
        
        X = labeled_data[feature_cols]
        y = labeled_data['Label'].astype(int)
        
        print(f"Training data shape: {X.shape}")
        print(f"Available features: {len(feature_cols)}")
        print(f"Label distribution: {y.value_counts().sort_index()}")
        
        return X, y, labeled_data['Datetime']
    
    def train_models(self, X, y):
        """Train multiple models and compare performance"""
        models = {
            'Random Forest': RandomForestClassifier(
                n_estimators=100, 
                max_depth=15, 
                min_samples_split=5,
                min_samples_leaf=3,
                random_state=42, 
                class_weight='balanced'
            ),
            'XGBoost': xgb.XGBClassifier(
                n_estimators=100, 
                max_depth=8, 
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=0.1,
                random_state=42, 
                eval_metric='mlogloss'
            ),
            'Logistic Regression': LogisticRegression(
                random_state=42, 
                max_iter=2000, 
                class_weight='balanced', 
                C=0.01,
                penalty='l2'
            ),
            'Naive Bayes': GaussianNB()
        }
        
        results = {}
        
        for name, model in models.items():
            print(f"\nTraining {name}...")
            
            if name in ['Logistic Regression']:
                X_scaled = self.scaler.fit_transform(X)
                X_train, X_test, y_train, y_test = train_test_split(
                    X_scaled, y, test_size=0.2, random_state=42, stratify=y
                )
            else:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42, stratify=y
                )
            
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            results[name] = {
                'model': model,
                'accuracy': (y_pred == y_test).mean(),
                'predictions': y_pred,
                'actual': y_test
            }
            
            print(f"{name} Accuracy: {results[name]['accuracy']:.4f}")
        
        return results


class TradingEnvironment(gym.Env):
    """Custom Trading Environment for RL"""
    
    def __init__(self, df, base_models, feature_cols, initial_balance=10000):
        super(TradingEnvironment, self).__init__()
        
        self.df = df.copy()
        self.base_models = base_models
        self.feature_cols = feature_cols
        self.initial_balance = initial_balance
        
        # Remove NaN values
        self.df = self.df.dropna(subset=feature_cols)
        self.df = self.df.reset_index(drop=True)
        
        # Action space: 0=Sell, 1=Hold, 2=Buy
        self.action_space = spaces.Discrete(3)
        
        # Observation space: model predictions + portfolio state + market features
        n_models = len(base_models)
        n_features = len(feature_cols)
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(n_models * 3 + 3 + n_features,),  # model probs + portfolio + features
            dtype=np.float32
        )
        
        self.reset()
    
    def reset(self):
        self.current_step = 0
        self.balance = self.initial_balance
        self.position = 0  # 0=no position, 1=long, -1=short
        self.position_size = 0
        self.entry_price = 0
        self.total_reward = 0
        self.trade_history = []
        
        return self._get_observation()
    
    def _get_observation(self):
        if self.current_step >= len(self.df):
            return np.zeros(self.observation_space.shape)
        
        # Get current row
        current_row = self.df.iloc[self.current_step]
        
        # Get model predictions
        model_predictions = []
        features = current_row[self.feature_cols].values.reshape(1, -1)
        
        for name, model_info in self.base_models.items():
            model = model_info['model']
            if hasattr(model, 'predict_proba'):
                probs = model.predict_proba(features)[0]
                model_predictions.extend(probs)
            else:
                pred = model.predict(features)[0]
                # Convert to one-hot-like format
                pred_one_hot = [0, 0, 0]
                pred_one_hot[pred] = 1
                model_predictions.extend(pred_one_hot)
        
        # Portfolio state
        portfolio_state = [
            self.balance / self.initial_balance,  # Normalized balance
            self.position,  # Current position
            self.position_size / self.initial_balance  # Normalized position size
        ]
        
        # Market features (normalized)
        market_features = features.flatten()
        
        # Combine all observations
        observation = np.concatenate([
            model_predictions,
            portfolio_state,
            market_features
        ]).astype(np.float32)
        
        return observation
    
    def step(self, action):
        if self.current_step >= len(self.df) - 1:
            return self._get_observation(), 0, True, {}
        
        current_price = self.df.iloc[self.current_step]['Close']
        next_price = self.df.iloc[self.current_step + 1]['Close']
        
        reward = 0
        
        # Execute action
        if action == 2:  # Buy
            if self.position <= 0:  # Close short or enter long
                if self.position < 0:  # Close short position
                    profit = (self.entry_price - current_price) * abs(self.position_size)
                    self.balance += profit
                    reward += profit / self.initial_balance
                
                # Enter long position
                self.position = 1
                self.position_size = self.balance * 0.95  # Use 95% of balance
                self.entry_price = current_price
                
        elif action == 0:  # Sell
            if self.position >= 0:  # Close long or enter short
                if self.position > 0:  # Close long position
                    profit = (current_price - self.entry_price) * self.position_size / self.entry_price
                    self.balance += profit
                    reward += profit / self.initial_balance
                
                # Enter short position
                self.position = -1
                self.position_size = self.balance * 0.95
                self.entry_price = current_price
        
        # Calculate unrealized P&L for current position
        if self.position > 0:  # Long position
            unrealized_pnl = (next_price - self.entry_price) * self.position_size / self.entry_price
            reward += unrealized_pnl / self.initial_balance * 0.1  # Small weight for unrealized gains
        elif self.position < 0:  # Short position
            unrealized_pnl = (self.entry_price - next_price) * self.position_size / self.entry_price
            reward += unrealized_pnl / self.initial_balance * 0.1
        
        # Penalty for excessive trading
        if len(self.trade_history) > 0 and self.trade_history[-1] != action:
            reward -= 0.001
        
        self.trade_history.append(action)
        self.current_step += 1
        self.total_reward += reward
        
        done = self.current_step >= len(self.df) - 1
        
        return self._get_observation(), reward, done, {
            'balance': self.balance,
            'position': self.position,
            'total_reward': self.total_reward
        }


class RLStackedModel:
    """Reinforcement Learning Stacked Model"""
    
    def __init__(self, base_models, feature_cols):
        self.base_models = base_models
        self.feature_cols = feature_cols
        self.rl_model = None
        self.env = None
        
    def create_environment(self, df, initial_balance=10000):
        """Create trading environment"""
        self.env = TradingEnvironment(df, self.base_models, self.feature_cols, initial_balance)
        return self.env
    
    def train_rl_model(self, df, total_timesteps=50000):
        """Train the RL model"""
        print("Creating trading environment...")
        env = self.create_environment(df)
        env = DummyVecEnv([lambda: env])
        
        print("Training RL model...")
        self.rl_model = PPO(
            "MlpPolicy", 
            env, 
            verbose=1,
            learning_rate=0.0003,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            tensorboard_log="./ppo_crypto_tensorboard/"
        )
        
        self.rl_model.learn(total_timesteps=total_timesteps)
        return self.rl_model
    
    def predict(self, df):
        """Make predictions using the trained RL model"""
        if self.rl_model is None:
            raise ValueError("RL model not trained yet!")
        
        env = self.create_environment(df)
        obs = env.reset()
        
        actions = []
        confidences = []
        
        done = False
        while not done:
            action, _states = self.rl_model.predict(obs, deterministic=True)
            actions.append(action)
            
            # Get action probabilities for confidence
            if hasattr(self.rl_model, 'policy'):
                action_probs = self.rl_model.policy.predict(obs.reshape(1, -1), deterministic=False)[1]
                confidences.append(np.max(action_probs))
            else:
                confidences.append(1.0)
            
            obs, reward, done, info = env.step(action)
        
        return np.array(actions), np.array(confidences)


class MLStackedStrategy(Strategy):
    """Backtesting Strategy using ML Stacked Model"""
    
    def __init__(self):
        super().__init__()
        self.rl_model = None
        self.signals = None
        self.signal_index = 0
    
    def init(self):
        # This will be set externally
        pass
    
    def next(self):
        if self.signal_index >= len(self.signals):
            return
        
        signal = self.signals[self.signal_index]
        
        if signal == 2:  # Buy signal
            if not self.position:
                self.buy(size=0.95)
        elif signal == 0:  # Sell signal
            if self.position:
                self.position.close()
        # signal == 1 is hold, do nothing
        
        self.signal_index += 1


def run_backtest(df, rl_model, base_models, feature_cols):
    """Run backtest using the RL stacked model"""
    print("Generating trading signals...")
    
    # Create stacked model
    stacked_model = RLStackedModel(base_models, feature_cols)
    stacked_model.rl_model = rl_model
    
    # Generate signals
    signals, confidences = stacked_model.predict(df)
    
    # Prepare data for backtesting
    backtest_df = df[['Datetime', 'Open', 'High', 'Low', 'Close', 'Volume']].copy()
    backtest_df = backtest_df.dropna()
    backtest_df = backtest_df.set_index('Datetime')
    
    # Limit signals to available data
    min_len = min(len(signals), len(backtest_df))
    signals = signals[:min_len]
    backtest_df = backtest_df.iloc[:min_len]
    
    print(f"Running backtest with {len(signals)} signals...")
    
    # Create strategy instance
    strategy = MLStackedStrategy
    
    # Run backtest
    bt = Backtest(backtest_df, strategy, cash=10000, commission=0.002)
    
    # Set signals in strategy
    def init_with_signals(self):
        self.signals = signals
        self.signal_index = 0
    
    strategy.init = init_with_signals
    
    # Run the backtest
    result = bt.run()
    
    print("Backtest Results:")
    print(f"Start: {result['Start']}")
    print(f"End: {result['End']}")
    print(f"Duration: {result['Duration']}")
    print(f"Exposure Time [%]: {result['Exposure Time [%]']:.2f}")
    print(f"Equity Final [$]: {result['Equity Final [$]']:.2f}")
    print(f"Equity Peak [$]: {result['Equity Peak [$]']:.2f}")
    print(f"Return [%]: {result['Return [%]']:.2f}")
    print(f"Buy & Hold Return [%]: {result['Buy & Hold Return [%]']:.2f}")
    print(f"Return (Ann.) [%]: {result['Return (Ann.) [%]']:.2f}")
    print(f"Volatility (Ann.) [%]: {result['Volatility (Ann.) [%]']:.2f}")
    print(f"Sharpe Ratio: {result['Sharpe Ratio']:.2f}")
    print(f"Sortino Ratio: {result['Sortino Ratio']:.2f}")
    print(f"Calmar Ratio: {result['Calmar Ratio']:.2f}")
    print(f"Max. Drawdown [%]: {result['Max. Drawdown [%]']:.2f}")
    print(f"Avg. Drawdown [%]: {result['Avg. Drawdown [%]']:.2f}")
    print(f"Max. Drawdown Duration: {result['Max. Drawdown Duration']}")
    print(f"Avg. Drawdown Duration: {result['Avg. Drawdown Duration']}")
    print(f"# Trades: {result['# Trades']}")
    print(f"Win Rate [%]: {result['Win Rate [%]']:.2f}")
    print(f"Best Trade [%]: {result['Best Trade [%]']:.2f}")
    print(f"Worst Trade [%]: {result['Worst Trade [%]']:.2f}")
    print(f"Avg. Trade [%]: {result['Avg. Trade [%]']:.2f}")
    print(f"Max. Trade Duration: {result['Max. Trade Duration']}")
    print(f"Avg. Trade Duration: {result['Avg. Trade Duration']}")
    print(f"Profit Factor: {result['Profit Factor']:.2f}")
    print(f"Expectancy [%]: {result['Expectancy [%]']:.2f}")
    print(f"SQN: {result['SQN']:.2f}")
    
    # Plot results
    bt.plot()
    
    return result, bt


def main():
    """Main function to run the complete pipeline"""
    print("Starting RL Stacked Model Training and Backtesting...")
    
    # Initialize the base model
    trader = CryptoTradingModel()
    
    # Load and combine all datasets
    print("Loading all datasets...")
    df = trader.load_all_datasets()
    
    # Prepare training data
    print("Preparing training data...")
    X, y, dates = trader.prepare_training_data(df)
    
    # Train base models
    print("Training base models...")
    base_models = trader.train_models(X, y)
    
    # Get feature columns
    feature_cols = X.columns.tolist()
    
    # Create and train RL stacked model
    print("Creating RL stacked model...")
    rl_stacked = RLStackedModel(base_models, feature_cols)
    
    # Train RL model (use a subset of data for faster training)
    train_df = df.head(10000)  # Use first 10k rows for training
    rl_model = rl_stacked.train_rl_model(train_df, total_timesteps=20000)
    
    # Run backtest on remaining data
    test_df = df.tail(5000)  # Use last 5k rows for testing
    result, bt = run_backtest(test_df, rl_model, base_models, feature_cols)
    
    return trader, rl_stacked, result, bt


if __name__ == "__main__":
    # You'll need to install these packages:
    # pip install stable-baselines3 backtesting gym
    
    trader, rl_stacked, result, bt = main()