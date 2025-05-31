import numpy as np
import pandas as pd
import yfinance as yf
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from gym import spaces
import gym
from datetime import datetime, timedelta

def calculate_heikin_ashi(df):
    """Calculate Heikin-Ashi candles"""
    ha_df = pd.DataFrame(index=df.index)
    
    # Calculate Heikin-Ashi values
    ha_df['HA_Close'] = (df['Open'] + df['High'] + df['Low'] + df['Close']) / 4
    
    # Initialize HA_Open with a vectorized operation
    ha_open = np.zeros(len(df))
    ha_open[0] = df['Open'].iloc[0]
    
    # Calculate HA_Open using vectorized operations
    for i in range(1, len(df)):
        ha_open[i] = (ha_open[i-1] + ha_df['HA_Close'].iloc[i-1]) / 2
    
    ha_df['HA_Open'] = ha_open
    ha_df['HA_High'] = df[['High', 'Open', 'Close']].max(axis=1)
    ha_df['HA_Low'] = df[['Low', 'Open', 'Close']].min(axis=1)
    
    return ha_df

def calculate_rsi(df, period=14):
    """Calculate RSI indicator"""
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

class TradingEnvironment(gym.Env):
    def __init__(self, df, initial_balance=10000, transaction_fee=0.001):
        super(TradingEnvironment, self).__init__()
        
        # Calculate indicators
        self.ha_df = calculate_heikin_ashi(df)
        self.rsi = calculate_rsi(df)
        
        # Combine all data
        self.df = df.copy()
        self.df['RSI'] = self.rsi
        self.df['HA_Open'] = self.ha_df['HA_Open']
        self.df['HA_Close'] = self.ha_df['HA_Close']
        self.df['HA_High'] = self.ha_df['HA_High']
        self.df['HA_Low'] = self.ha_df['HA_Low']
        
        # Drop NaN values that result from indicator calculations
        self.df = self.df.dropna().reset_index(drop=True)
        
        self.initial_balance = initial_balance
        self.transaction_fee = transaction_fee
        
        # Define action space (0: hold, 1: buy, 2: sell)
        self.action_space = spaces.Discrete(3)
        
        # Define observation space with additional indicators
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(10,),  # [price, volume, balance, position, returns, RSI, HA_Open, HA_Close, HA_High, HA_Low]
            dtype=np.float32
        )
        
        self.reset()
    
    def reset(self):
        self.current_step = 0
        self.balance = float(self.initial_balance)
        self.position = 0.0
        self.total_value = float(self.balance)
        self.returns = 0.0
        self.last_trade_price = 0.0
        self.trades = []
        self.trade_count = 0
        
        return self._get_observation()
    
    def _get_observation(self):
        if self.current_step >= len(self.df):
            self.current_step = len(self.df) - 1
            
        price = float(self.df['Close'].iloc[self.current_step])
        volume = float(self.df['Volume'].iloc[self.current_step])
        rsi = float(self.df['RSI'].iloc[self.current_step])
        ha_open = float(self.df['HA_Open'].iloc[self.current_step])
        ha_close = float(self.df['HA_Close'].iloc[self.current_step])
        ha_high = float(self.df['HA_High'].iloc[self.current_step])
        ha_low = float(self.df['HA_Low'].iloc[self.current_step])
        
        # Handle NaN values
        rsi = rsi if not np.isnan(rsi) else 50.0
        
        observation = np.array([
            price / 1000.0,  # Normalize price
            volume / 1e6,    # Normalize volume
            self.balance / self.initial_balance,  # Balance ratio
            self.position * price / self.initial_balance,  # Position value ratio
            self.returns,
            rsi / 100.0,     # Normalize RSI to 0-1
            ha_open / 1000.0,
            ha_close / 1000.0,
            ha_high / 1000.0,
            ha_low / 1000.0
        ], dtype=np.float32)
        
        # Replace any remaining NaN or inf values
        observation = np.nan_to_num(observation, nan=0.0, posinf=1.0, neginf=-1.0)
        
        return observation
    
    def step(self, action):
        if self.current_step >= len(self.df) - 1:
            return self._get_observation(), 0, True, {}
            
        current_price = float(self.df['Close'].iloc[self.current_step])
        previous_value = self.total_value
        
        # Debug: Print trade attempts
        trade_executed = False
        
        # Execute action
        if action == 1:  # Buy
            if self.balance > 0 and self.position == 0:  # Only buy if no position
                # Calculate how much we can afford including fees
                max_spend = self.balance / (1 + self.transaction_fee)
                shares_to_buy = max_spend / current_price
                
                if shares_to_buy > 0:
                    cost = shares_to_buy * current_price * (1 + self.transaction_fee)
                    
                    if cost <= self.balance:
                        self.position = shares_to_buy
                        self.balance -= cost
                        self.last_trade_price = current_price
                        self.trades.append(('BUY', current_price, shares_to_buy, self.current_step))
                        self.trade_count += 1
                        trade_executed = True
                        print(f"Step {self.current_step}: BUY {shares_to_buy:.4f} shares at ${current_price:.4f}")
                        
        elif action == 2:  # Sell
            if self.position > 0:  # Only sell if holding position
                shares_to_sell = self.position
                revenue = shares_to_sell * current_price * (1 - self.transaction_fee)
                
                self.balance += revenue
                self.position = 0
                self.last_trade_price = current_price
                self.trades.append(('SELL', current_price, shares_to_sell, self.current_step))
                self.trade_count += 1
                trade_executed = True
                print(f"Step {self.current_step}: SELL {shares_to_sell:.4f} shares at ${current_price:.4f}")
        
        # Calculate total value and returns
        position_value = self.position * current_price
        self.total_value = self.balance + position_value
        self.returns = (self.total_value - self.initial_balance) / self.initial_balance
        
        # Calculate reward - improved version
        reward = 0.0
        
        if previous_value > 0:
            # Portfolio change reward
            portfolio_change = (self.total_value - previous_value) / previous_value
            reward += portfolio_change * 100  # Scale up the reward
            
            # Reward for taking action when appropriate
            if trade_executed:
                reward += 0.1  # Small bonus for trading
            
            # RSI-based reward
            rsi = self.df['RSI'].iloc[self.current_step]
            if not np.isnan(rsi):
                if action == 1 and rsi < 30:  # Buy when oversold
                    reward += 0.5
                elif action == 2 and rsi > 70:  # Sell when overbought
                    reward += 0.5
                elif action == 1 and rsi > 70:  # Penalty for buying overbought
                    reward -= 0.2
                elif action == 2 and rsi < 30:  # Penalty for selling oversold
                    reward -= 0.2
        
        # Move to next step
        self.current_step += 1
        done = self.current_step >= len(self.df) - 1
        
        info = {
            'total_value': self.total_value,
            'balance': self.balance,
            'position': self.position,
            'trade_executed': trade_executed,
            'trade_count': self.trade_count
        }
        
        return self._get_observation(), reward, done, info

def fetch_data(symbol='XRP-USD', period='max', interval='15m'):
    """Fetch data from yfinance with 15-minute candles"""
    ticker = yf.Ticker(symbol)
    df = ticker.history(start=datetime.now() - timedelta(days=59), end=datetime.now(), interval=interval)
    
    # Remove any rows with NaN values
    df = df.dropna()
    
    print(f"Fetched {len(df)} candles from {df.index[0]} to {df.index[-1]}")
    return df

def train_model(df, total_timesteps=100000):
    """Train the trading model"""
    print(f"Training with {len(df)} data points")
    
    env = TradingEnvironment(df)
    env = DummyVecEnv([lambda: env])
    
    # Better PPO parameters
    model = PPO(
        'MlpPolicy', 
        env, 
        verbose=1,
        learning_rate=0.0003,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        ent_coef=0.01  # Encourage exploration
    )
    
    model.learn(total_timesteps=total_timesteps)
    
    return model

def main():
    # Fetch data
    print("Fetching XRP/USD data...")
    df = fetch_data()
    
    # Train model
    print("Training model...")
    model = train_model(df)
    
    # Save model
    model.save("trading_model")
    print("Model saved as 'trading_model'")
    
    # Quick test to verify trades
    print("\n=== TESTING MODEL ===")
    test_env = TradingEnvironment(df[-100:])  # Test on last 100 steps
    obs = test_env.reset()
    
    for i in range(50):
        action, _ = model.predict(obs)
        obs, reward, done, info = test_env.step(action)
        if done:
            break
    
    print(f"\nTest Results:")
    print(f"Trades executed: {len(test_env.trades)}")
    print(f"Final portfolio value: ${test_env.total_value:.2f}")
    print(f"Final balance: ${test_env.balance:.2f}")
    print(f"Final position: {test_env.position:.4f}")
    
    for trade in test_env.trades:
        print(f"Trade: {trade[0]} {trade[2]:.4f} shares at ${trade[1]:.4f}")

if __name__ == "__main__":
    main()