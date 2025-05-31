import numpy as np
import pandas as pd
from itertools import product
from stable_baselines3 import PPO
from trading_ai import TradingEnvironment, fetch_data
from datetime import datetime, timedelta
import json
from tqdm import tqdm
import multiprocessing
from functools import partial
import torch
import os

def get_device():
    """Get the best available device (CPU for MLP policies)"""
    # For MLP policies, CPU is typically faster than GPU
    print("Using CPU for MLP policy (faster than GPU for this type of network)")
    return torch.device("cpu")

def evaluate_single_split(params, train_data, validation_data, device):
    """Evaluate parameters on a single train/validation split"""
    # Create environment with parameters
    env = TradingEnvironment(train_data, **params)
    
    # Train model with GPU support
    model = PPO(
        'MlpPolicy',
        env,
        verbose=0,
        learning_rate=0.0003,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        ent_coef=0.01,
        device=device  # Specify device for the model
    )
    
    # Train for a shorter period during optimization
    model.learn(total_timesteps=50000)
    
    # Evaluate on validation data
    val_env = TradingEnvironment(validation_data, **params)
    obs, _ = val_env.reset()  # Unpack observation from reset
    done = False
    total_reward = 0
    
    while not done:
        action, _ = model.predict(obs)
        obs, reward, terminated, truncated, _ = val_env.step(action)  # Unpack all values from step
        done = terminated or truncated
        total_reward += reward
    
    # Calculate metrics
    final_value = val_env.total_value
    returns = (final_value - val_env.initial_balance) / val_env.initial_balance * 100
    win_rate = sum(1 for trade in val_env.closed_trades if trade['profit'] > 0) / len(val_env.closed_trades) if val_env.closed_trades else 0
    
    return {
        'params': params,
        'total_reward': total_reward,
        'final_value': final_value,
        'returns': returns,
        'win_rate': win_rate,
        'num_trades': len(val_env.closed_trades)
    }

class ParameterOptimizer:
    def __init__(self, param_grid):
        self.param_grid = param_grid
        self.results = []
        self.device = get_device()
        
    def evaluate_parameters(self, params, splits):
        """Evaluate parameters using cross-validation in parallel"""
        # Create train/validation splits
        split_results = []
        for i in range(len(splits)):
            validation_data = splits[i]
            train_data = pd.concat(splits[:i] + splits[i+1:])
            split_results.append(evaluate_single_split(params, train_data, validation_data, self.device))
        
        # Average results across splits
        avg_result = {
            'params': params,
            'avg_total_reward': np.mean([r['total_reward'] for r in split_results]),
            'avg_returns': np.mean([r['returns'] for r in split_results]),
            'avg_win_rate': np.mean([r['win_rate'] for r in split_results]),
            'avg_num_trades': np.mean([r['num_trades'] for r in split_results]),
            'std_returns': np.std([r['returns'] for r in split_results])
        }
        
        return avg_result
    
    def optimize(self, data, n_splits=3):
        """Run grid search optimization in parallel"""
        # Split data into n_splits for cross-validation
        split_size = len(data) // n_splits
        splits = [data[i:i + split_size] for i in range(0, len(data), split_size)]
        
        # Generate all parameter combinations
        param_combinations = [dict(zip(self.param_grid.keys(), v)) 
                            for v in product(*self.param_grid.values())]
        
        print(f"Testing {len(param_combinations)} parameter combinations...")
        
        # Create a pool of workers
        num_cores = multiprocessing.cpu_count()
        print(f"Using {num_cores} CPU cores for parallel processing")
        pool = multiprocessing.Pool(processes=num_cores)
        
        # Create partial function with fixed splits
        evaluate_func = partial(self.evaluate_parameters, splits=splits)
        
        # Run optimization in parallel with progress bar
        self.results = list(tqdm(
            pool.imap(evaluate_func, param_combinations),
            total=len(param_combinations)
        ))
        
        # Close the pool
        pool.close()
        pool.join()
        
        # Sort results by average returns
        self.results.sort(key=lambda x: x['avg_returns'], reverse=True)
        
        return self.results
    
    def save_results(self, filename='parameter_optimization_results.json'):
        """Save optimization results to file"""
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=4)
    
    def print_top_results(self, n=5):
        """Print top n results"""
        print("\nTop parameter combinations:")
        for i, result in enumerate(self.results[:n], 1):
            print(f"\n{i}. Parameters:")
            for param, value in result['params'].items():
                print(f"   {param}: {value}")
            print(f"   Average Returns: {result['avg_returns']:.2f}%")
            print(f"   Average Win Rate: {result['avg_win_rate']:.2f}")
            print(f"   Average Number of Trades: {result['avg_num_trades']:.1f}")
            print(f"   Returns Std Dev: {result['std_returns']:.2f}%")

def main():
    # Define parameter grid
    param_grid = {
        'portfolio_change_multiplier': [10, 50, 100],
        'stop_loss_penalty': [5, 10, 20],
        'take_profit_reward': [5, 10, 20],
        'excessive_trade_base_penalty': [25, 50, 100],
        'high_win_rate_bonus': [5, 10, 20],
        'win_rate_window': [5, 10, 15]
    }
    
    # Fetch data
    print("Fetching data...")
    df = fetch_data()
    
    # Create optimizer
    optimizer = ParameterOptimizer(param_grid)
    
    # Run optimization
    print("Starting parameter optimization...")
    results = optimizer.optimize(df)
    
    # Save and print results
    optimizer.save_results()
    optimizer.print_top_results(n=5)

if __name__ == "__main__":
    # Set start method for multiprocessing
    multiprocessing.set_start_method('spawn')
    main() 