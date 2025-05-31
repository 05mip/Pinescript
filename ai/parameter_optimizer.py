import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from trading_ai import TradingEnvironment, fetch_data
from datetime import datetime, timedelta
import json
from tqdm import tqdm
import multiprocessing
from functools import partial
import torch
import os
import random
from typing import List, Dict, Tuple

def get_device():
    """Get the best available device (CPU for MLP policies)"""
    # For MLP policies, CPU is typically faster than GPU
    print("Using CPU for MLP policy (faster than GPU for this type of network)")
    return torch.device("cpu")

def generate_random_params() -> Dict:
    """Generate random parameters within reasonable ranges"""
    return {
        'portfolio_change_multiplier': random.uniform(30, 300),
        'stop_loss_penalty': random.uniform(10, 50),
        'take_profit_reward': random.uniform(10, 50),
        'excessive_trade_base_penalty': random.uniform(50, 300),
        'high_win_rate_bonus': random.uniform(10, 40),
        'win_rate_window': random.randint(5, 30),
        'holding_reward_multiplier': random.uniform(2, 15)
    }

def mutate_params(params: Dict, mutation_rate: float = 0.2) -> Dict:
    """Mutate parameters with some probability"""
    new_params = params.copy()
    mutated = False
    
    for key in new_params:
        if random.random() < mutation_rate:
            old_value = new_params[key]
            if key == 'win_rate_window':
                # For integer parameters
                new_params[key] = max(1, int(new_params[key] * random.uniform(0.5, 1.5)))
            else:
                # For float parameters
                new_params[key] = new_params[key] * random.uniform(0.5, 1.5)
            print(f"  Mutated {key}: {old_value:.2f} -> {new_params[key]:.2f}")
            mutated = True
    
    if not mutated:
        print("  No mutations occurred")
    return new_params

def crossover(parent1: Dict, parent2: Dict) -> Dict:
    """Create a child by combining parameters from two parents"""
    child = {}
    print("\nCrossover:")
    for key in parent1:
        if random.random() < 0.5:
            child[key] = parent1[key]
            print(f"  {key}: {child[key]:.2f} (from parent 1)")
        else:
            child[key] = parent2[key]
            print(f"  {key}: {child[key]:.2f} (from parent 2)")
    return child

def train_and_evaluate(params: Dict, train_data: pd.DataFrame, validation_data: pd.DataFrame) -> Dict:
    """Train and evaluate a single model with given parameters"""
    print(f"\nTraining with parameters:")
    for param, value in params.items():
        print(f"  {param}: {value:.2f}")
    
    # Create environment with parameters
    env = TradingEnvironment(train_data, **params)
    
    # Train model
    model = PPO(
        'MlpPolicy',
        env,
        verbose=0,  # Reduce verbosity for parallel training
        learning_rate=0.0003,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        ent_coef=0.01,
        device='cpu'  # Force CPU for parallel processing
    )
    
    print("\nStarting training...")
    model.learn(total_timesteps=50000)
    print("Training completed. Starting evaluation...")
    
    # Evaluate on validation data
    val_env = TradingEnvironment(validation_data, **params)
    obs, _ = val_env.reset()
    done = False
    total_reward = 0
    
    while not done:
        action, _ = model.predict(obs)
        obs, reward, terminated, truncated, _ = val_env.step(action)
        done = terminated or truncated
        total_reward += reward
    
    # Calculate metrics
    final_value = val_env.total_value
    returns = (final_value - val_env.initial_balance) / val_env.initial_balance * 100
    win_rate = sum(1 for trade in val_env.closed_trades if trade['profit'] > 0) / len(val_env.closed_trades) if val_env.closed_trades else 0
    
    print(f"\nEvaluation Results:")
    print(f"  Returns: {returns:.2f}%")
    print(f"  Win Rate: {win_rate:.2f}")
    print(f"  Number of Trades: {len(val_env.closed_trades)}")
    print(f"  Final Portfolio Value: ${final_value:.2f}")
    
    return {
        'params': params,
        'total_reward': total_reward,
        'final_value': final_value,
        'returns': returns,
        'win_rate': win_rate,
        'num_trades': len(val_env.closed_trades)
    }

def evaluate_population_parallel(population: List[Dict], data: pd.DataFrame, generation: int, n_splits: int = 3) -> List[Dict]:
    """Evaluate all individuals in the population in parallel"""
    print(f"\n{'='*50}")
    print(f"Evaluating Generation {generation}")
    print(f"{'='*50}")
    
    # Split data for cross-validation
    split_size = len(data) // n_splits
    splits = [data[i:i + split_size] for i in range(0, len(data), split_size)]
    
    # Create a pool of workers
    num_cores = multiprocessing.cpu_count()
    print(f"Using {num_cores} CPU cores for parallel processing")
    pool = multiprocessing.Pool(processes=num_cores)
    
    all_results = []
    
    # Evaluate each individual in parallel
    for i, params in enumerate(population):
        print(f"\nEvaluating individual {i+1}/{len(population)}")
        split_results = []
        
        # Create tasks for each cross-validation fold
        tasks = []
        for j in range(n_splits):
            validation_data = splits[j]
            train_data = pd.concat(splits[:j] + splits[j+1:])
            tasks.append((params, train_data, validation_data))
        
        # Run tasks in parallel
        fold_results = pool.starmap(train_and_evaluate, tasks)
        split_results.extend(fold_results)
        
        # Average results across splits
        avg_result = {
            'params': params,
            'avg_returns': np.mean([r['returns'] for r in split_results]),
            'avg_win_rate': np.mean([r['win_rate'] for r in split_results]),
            'avg_num_trades': np.mean([r['num_trades'] for r in split_results]),
            'std_returns': np.std([r['returns'] for r in split_results])
        }
        all_results.append(avg_result)
        
        print(f"\nIndividual {i+1} Results:")
        print(f"  Average Returns: {avg_result['avg_returns']:.2f}%")
        print(f"  Average Win Rate: {avg_result['avg_win_rate']:.2f}")
        print(f"  Average Number of Trades: {avg_result['avg_num_trades']:.1f}")
        print(f"  Returns Std Dev: {avg_result['std_returns']:.2f}%")
    
    # Close the pool
    pool.close()
    pool.join()
    
    return sorted(all_results, key=lambda x: x['avg_returns'], reverse=True)

class GeneticOptimizer:
    def __init__(self, population_size: int = 10, elite_size: int = 2, mutation_rate: float = 0.2):
        self.population_size = population_size
        self.elite_size = elite_size
        self.mutation_rate = mutation_rate
        self.device = get_device()
        self.generation = 0
        self.best_params = None
        self.best_score = float('-inf')
        
    def create_initial_population(self) -> List[Dict]:
        """Create initial random population"""
        print("\nCreating initial population...")
        population = [generate_random_params() for _ in range(self.population_size)]
        print(f"Generated {len(population)} random parameter sets")
        return population
    
    def evaluate_population(self, population: List[Dict], data: pd.DataFrame) -> List[Dict]:
        """Evaluate all individuals in the population using parallel processing"""
        return evaluate_population_parallel(population, data, self.generation)
    
    def create_next_generation(self, evaluated_population: List[Dict]) -> List[Dict]:
        """Create next generation using genetic operators"""
        print(f"\n{'='*50}")
        print(f"Creating Generation {self.generation + 1}")
        print(f"{'='*50}")
        
        next_generation = []
        
        # Keep elite individuals
        print("\nKeeping elite individuals:")
        for i in range(self.elite_size):
            elite = evaluated_population[i]
            print(f"\nElite {i+1}:")
            print(f"  Returns: {elite['avg_returns']:.2f}%")
            print(f"  Win Rate: {elite['avg_win_rate']:.2f}")
            for param, value in elite['params'].items():
                print(f"  {param}: {value:.2f}")
            next_generation.append(elite['params'])
        
        # Create rest of population through crossover and mutation
        print("\nCreating new individuals through crossover and mutation:")
        while len(next_generation) < self.population_size:
            print(f"\nCreating individual {len(next_generation) + 1}/{self.population_size}")
            
            # Tournament selection
            parent1 = random.choice(evaluated_population[:self.population_size//2])['params']
            parent2 = random.choice(evaluated_population[:self.population_size//2])['params']
            
            print("\nSelected parents:")
            print("Parent 1:", {k: f"{v:.2f}" for k, v in parent1.items()})
            print("Parent 2:", {k: f"{v:.2f}" for k, v in parent2.items()})
            
            # Crossover
            child = crossover(parent1, parent2)
            
            # Mutation
            print("\nApplying mutations:")
            child = mutate_params(child, self.mutation_rate)
            
            next_generation.append(child)
        
        return next_generation
    
    def optimize(self, data: pd.DataFrame, generations: int = 5) -> Tuple[Dict, float]:
        """Run genetic optimization"""
        print(f"\n{'='*50}")
        print("Starting Genetic Optimization")
        print(f"{'='*50}")
        print(f"Population Size: {self.population_size}")
        print(f"Elite Size: {self.elite_size}")
        print(f"Mutation Rate: {self.mutation_rate}")
        print(f"Number of Generations: {generations}")
        print(f"{'='*50}")
        
        # Create initial population
        population = self.create_initial_population()
        
        for gen in range(generations):
            self.generation = gen + 1
            
            # Evaluate current population
            evaluated_population = self.evaluate_population(population, data)
            
            # Update best parameters if found
            if evaluated_population[0]['avg_returns'] > self.best_score:
                self.best_score = evaluated_population[0]['avg_returns']
                self.best_params = evaluated_population[0]['params']
                print(f"\n{'='*50}")
                print("NEW BEST PARAMETERS FOUND!")
                print(f"{'='*50}")
                print(f"Returns: {self.best_score:.2f}%")
                print("Parameters:")
                for param, value in self.best_params.items():
                    print(f"  {param}: {value:.2f}")
            
            # Create next generation
            population = self.create_next_generation(evaluated_population)
            
            # Save progress
            self.save_generation_results(evaluated_population)
            
            print(f"\n{'='*50}")
            print(f"Generation {self.generation} completed")
            print(f"{'='*50}")
        
        return self.best_params, self.best_score
    
    def save_generation_results(self, results: List[Dict], filename: str = 'genetic_optimization_results.json'):
        """Save results for current generation"""
        generation_data = {
            'generation': self.generation,
            'best_score': self.best_score,
            'best_params': self.best_params,
            'all_results': results
        }
        
        # Load existing results if file exists
        if os.path.exists(filename):
            with open(filename, 'r') as f:
                all_results = json.load(f)
        else:
            all_results = []
        
        all_results.append(generation_data)
        
        # Save updated results
        with open(filename, 'w') as f:
            json.dump(all_results, f, indent=4)

def main():
    # Fetch data
    print("Fetching data...")
    df = fetch_data()
    print(f"Data fetched: {len(df)} candles")
    
    # Create and run genetic optimizer
    optimizer = GeneticOptimizer(
        population_size=10,  # Number of parameter sets to test per generation
        elite_size=2,       # Number of best performers to keep
        mutation_rate=0.2   # Probability of parameter mutation
    )
    
    # Run optimization
    best_params, best_score = optimizer.optimize(df, generations=5)
    
    print("\nOptimization completed!")
    print("\nBest parameters found:")
    for param, value in best_params.items():
        print(f"{param}: {value}")
    print(f"\nBest score: {best_score:.2f}%")

if __name__ == "__main__":
    # Set start method for multiprocessing
    multiprocessing.set_start_method('spawn')
    main() 