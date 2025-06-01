import numpy as np
import pandas as pd
import yfinance as yf
from stable_baselines3 import PPO
from gymnasium.vector import SyncVectorEnv
import gymnasium as gym
from gymnasium import spaces
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
    def __init__(self, df, initial_balance=10000, transaction_fee=0.001,
                 portfolio_change_multiplier=82.378,
                 stop_loss_penalty=40.106,
                 take_profit_reward=20.007,
                 excessive_trade_base_penalty=225.611,
                 high_win_rate_bonus=24.45,
                 win_rate_window=15,
                 holding_reward_multiplier=2.5):
        super().__init__()
        
        # Reward parameters
        self.portfolio_change_multiplier = portfolio_change_multiplier
        self.stop_loss_penalty = stop_loss_penalty
        self.take_profit_reward = take_profit_reward
        self.excessive_trade_base_penalty = excessive_trade_base_penalty
        self.high_win_rate_bonus = high_win_rate_bonus
        self.win_rate_window = win_rate_window
        self.holding_reward_multiplier = holding_reward_multiplier
        
        # Trade tracking
        self.closed_trades = []  # List to track closed trades
        
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
        
        # Add trend indicators
        self.df['HA_Trend'] = np.where(self.df['HA_Close'] > self.df['HA_Open'], 1, -1)
        self.df['RSI_Trend'] = np.where(self.df['RSI'] > 50, 1, -1)
        
        # Drop NaN values that result from indicator calculations
        self.df = self.df.dropna().reset_index(drop=True)
        
        self.initial_balance = initial_balance
        self.transaction_fee = transaction_fee
        self.min_trade_interval = 24  # Increased from 12 to 24 (6 hours between trades)
        self.max_position_size = 0.5  # Maximum 50% of portfolio in one position
        self.stop_loss_pct = 0.02  # 2% stop loss
        self.take_profit_pct = 0.03  # 3% take profit
        self.max_daily_trades = 4  # Reduced from 6 to 4 trades per day
        self.daily_trades = 0  # Counter for trades in current day
        self.current_day = None  # Track current day for trade counting
        
        # New trading restrictions
        self.min_hold_period = 12  # Minimum 3 hours holding period
        self.last_trade_profit = 0  # Track last trade's profit
        self.loss_cooldown = 0  # Cooldown period after losses
        self.min_profit_threshold = 0.005  # 0.5% minimum profit target
        
        # Define action space (0: hold, 1: buy, 2: sell)
        self.action_space = spaces.Discrete(3)
        
        # Define observation space with additional indicators
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(12,),  # Keep 12 dimensions to match trained model
            dtype=np.float32
        )
        
        self.reset()
    
    def reset(self, seed=None, options=None):
        """Reset the environment to initial state"""
        super().reset(seed=seed)  # Required by Gymnasium
        
        self.current_step = 0
        self.balance = float(self.initial_balance)
        self.position = None  # None: no position, 'long': in long position
        self.total_value = float(self.balance)
        self.returns = 0.0
        self.last_trade_price = 0.0
        self.trades = []
        self.closed_trades = []  # Clear closed trades
        self.trade_count = 0
        self.last_trade_step = -self.min_trade_interval
        self.entry_price = 0.0
        self.daily_trades = 0
        
        # Safely get the current day from the index
        if len(self.df) > 0 and isinstance(self.df.index[0], (pd.Timestamp, datetime)):
            self.current_day = self.df.index[0].date()
        else:
            self.current_day = None
        
        return self._get_observation(), {}  # Return observation and empty info dict as required by Gymnasium
    
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
        ha_trend = float(self.df['HA_Trend'].iloc[self.current_step])
        rsi_trend = float(self.df['RSI_Trend'].iloc[self.current_step])
        
        # Convert position to numeric value
        position_value = 1 if self.position == 'long' else 0
        
        observation = np.array([
            price / 1000.0,  # Normalize price
            volume / 1e6,    # Normalize volume
            position_value,
            self.returns,
            rsi / 100.0,     # Normalize RSI to 0-1
            ha_open / 1000.0,
            ha_close / 1000.0,
            ha_high / 1000.0,
            ha_low / 1000.0,
            ha_trend,
            rsi_trend,
            self.balance / self.initial_balance  # Normalized balance
        ], dtype=np.float32)
        
        # Replace any remaining NaN or inf values
        observation = np.nan_to_num(observation, nan=0.0, posinf=1.0, neginf=-1.0)
        
        return observation
    
    def step(self, action):
        sl_triggered = False
        tp_triggered = False

        if self.current_step >= len(self.df) - 1:
            return self._get_observation(), 0, True, False, {}
            
        current_price = float(self.df['Close'].iloc[self.current_step])
        previous_value = self.total_value
        
        # Check if we're in a new day and reset daily trade counter
        current_date = self.df.index[self.current_step]
        if isinstance(current_date, (pd.Timestamp, datetime)):
            current_date = current_date.date()
            
        if self.current_day != current_date:
            self.daily_trades = 0
            self.current_day = current_date
        
        # Check if enough time has passed since last trade
        can_trade = (self.current_step - self.last_trade_step) >= self.min_trade_interval
        
        # Check if we're in cooldown period after a loss
        if self.loss_cooldown > 0:
            self.loss_cooldown -= 1
            can_trade = False
        
        # Check stop loss and take profit if in position
        if self.position == 'long':
            price_change = (current_price - self.entry_price) / self.entry_price
            
            # Stop loss
            if price_change <= -self.stop_loss_pct:
                action = 2  # Force sell
                sl_triggered = True
                self.loss_cooldown = 12  # 3-hour cooldown after loss
            
            # Take profit
            elif price_change >= self.take_profit_pct:
                action = 2  # Force sell
                tp_triggered = True
        
        # Execute action
        trade_executed = False
        
        if action == 1 and can_trade:  # Buy
            if self.position is None:  # Only buy if no position
                # Calculate position size based on max_position_size
                max_position_value = self.total_value * self.max_position_size
                max_shares = max_position_value / current_price
                
                # Calculate how much we can afford including fees
                max_spend = self.balance / (1 + self.transaction_fee)
                shares_to_buy = min(max_shares, max_spend / current_price)
                
                if shares_to_buy > 0:
                    cost = shares_to_buy * current_price * (1 + self.transaction_fee)
                    
                    if cost <= self.balance:
                        self.position = 'long'
                        self.balance -= cost
                        self.last_trade_price = current_price
                        self.entry_price = current_price
                        self.trades.append(('BUY', current_price, shares_to_buy, self.current_step))
                        self.trade_count += 1
                        self.last_trade_step = self.current_step
                        self.daily_trades += 1
                        trade_executed = True
                        
        elif action == 2 and can_trade:  # Sell
            if self.position == 'long':  # Only sell if holding position
                # Check minimum holding period
                if (self.current_step - self.last_trade_step) < self.min_hold_period:
                    return self._get_observation(), -1.0, False, False, {}  # Penalty for early selling
                
                # Calculate position value
                position_value = self.balance * self.max_position_size  # Use max position size as reference
                shares_to_sell = position_value / current_price
                revenue = shares_to_sell * current_price * (1 - self.transaction_fee)
                
                # Calculate trade profit
                trade_profit = revenue - (shares_to_sell * self.entry_price * (1 + self.transaction_fee))
                self.last_trade_profit = trade_profit
                
                # Record closed trade
                self.closed_trades.append({
                    'entry_price': self.entry_price,
                    'exit_price': current_price,
                    'shares': shares_to_sell,
                    'profit': trade_profit
                })
                
                self.balance += revenue
                self.position = None
                self.last_trade_price = current_price
                self.entry_price = 0
                self.trades.append(('SELL', current_price, shares_to_sell, self.current_step))
                self.trade_count += 1
                self.last_trade_step = self.current_step
                self.daily_trades += 1
                trade_executed = True
                
                # Set cooldown if loss
                if trade_profit < 0:
                    self.loss_cooldown = 12  # 3-hour cooldown after loss
        
        # Calculate total value and returns
        position_value = self.balance * self.max_position_size if self.position == 'long' else 0
        self.total_value = self.balance + position_value
        self.returns = (self.total_value - self.initial_balance) / self.initial_balance
        
        # Calculate reward - improved version
        reward = 0.0
        
        if previous_value > 0:
            # Portfolio change reward
            portfolio_change = (self.total_value - previous_value) / previous_value
            reward += portfolio_change * self.portfolio_change_multiplier
            
            # Transaction cost penalty
            if trade_executed:
                transaction_cost = abs(position_value * self.transaction_fee)
                reward -= transaction_cost * 10  # Penalize transaction costs
            
            # Add exponential penalty for excessive trading
            if self.daily_trades > self.max_daily_trades:
                excess_trades = self.daily_trades - self.max_daily_trades
                penalty = self.excessive_trade_base_penalty ** excess_trades
                reward -= penalty
                print(f"Excessive trading penalty: -{penalty:.2f} (Daily trades: {self.daily_trades})")
            
            # Reward holding profitable positions
            if self.position == 'long':
                unrealized_profit = (current_price - self.entry_price) / self.entry_price
                if unrealized_profit > 0:
                    reward += unrealized_profit * self.holding_reward_multiplier
        
        if sl_triggered:
            reward -= self.stop_loss_penalty
        if tp_triggered:
            reward += self.take_profit_reward

        # Add win rate based rewards/penalties
        if len(self.closed_trades) >= self.win_rate_window:
            recent_trades = self.closed_trades[-self.win_rate_window:]
            win_rate = sum(1 for trade in recent_trades if trade['profit'] > 0) / self.win_rate_window
            if win_rate >= 0.6:  # High win rate bonus
                reward += self.high_win_rate_bonus
        else:
            # Small positive reward for early trades to encourage exploration
            reward += 0.5
            
        # Move to next step
        self.current_step += 1
        done = self.current_step >= len(self.df) - 1
        
        info = {
            'total_value': self.total_value,
            'balance': self.balance,
            'position': self.position,
            'trade_executed': trade_executed,
            'trade_count': self.trade_count,
            'daily_trades': self.daily_trades
        }
        
        return self._get_observation(), reward, done, False, info

def fetch_data(symbol='XRP-USD', start_date=None, end_date=None, interval='15m'):
    """Fetch data from yfinance with 15-minute candles"""
    ticker = yf.Ticker(symbol)
    
    if start_date is None:
        start_date = datetime.now() - timedelta(days=59)
    if end_date is None:
        end_date = datetime.now()
        
    df = ticker.history(start=start_date, end=end_date, interval=interval)
    
    # Ensure the index is datetime
    df.index = pd.to_datetime(df.index)
    
    # Remove any rows with NaN values
    df = df.dropna()
    
    print(f"Fetched {len(df)} candles from {df.index[0]} to {df.index[-1]}")
    return df

def train_model(df, total_timesteps=100000):
    """Train the trading model"""
    print(f"Training with {len(df)} data points")
    
    # Create environment
    env = TradingEnvironment(df)
    
    # Train model
    model = PPO(
        'MlpPolicy', 
        env, 
        verbose=1,
        learning_rate=0.0003,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        ent_coef=0.01,  # Encourage exploration
        device='cpu'  # Force CPU for MLP policies
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
    obs, _ = test_env.reset()  # Unpack observation and info
    
    for i in range(50):
        action, _ = model.predict(obs)
        obs, reward, done, truncated, info = test_env.step(action)
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