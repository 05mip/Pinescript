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
    """Generate random parameters within wider ranges"""
    return {
        'portfolio_change_multiplier': random.uniform(5, 500),  # Wider range
        'stop_loss_penalty': random.uniform(5, 200),  # Wider range
        'take_profit_reward': random.uniform(5, 200),  # Wider range
        'excessive_trade_base_penalty': random.uniform(20, 500),  # Wider range
        'high_win_rate_bonus': random.uniform(5, 200),  # Wider range
        'win_rate_window': random.randint(3, 30),  # Keep as integer
        'holding_reward_multiplier': random.uniform(2, 100),  # Wider range
        'buy_pyramiding_penalty': random.uniform(5, 100),  # Wider range
        'sell_pyramiding_penalty': random.uniform(5, 100)  # Wider range
    }

def mutate_params(params: Dict, mutation_rate: float = 0.2) -> Dict:
    """Mutate parameters with higher variation"""
    new_params = params.copy()
    mutated = False
    
    for key in new_params:
        if random.random() < mutation_rate:
            old_value = new_params[key]
            if key == 'win_rate_window':
                # For integer parameters - ensure it stays an integer
                new_value = max(1, int(new_params[key] * random.uniform(0.3, 2.0)))
                new_params[key] = min(50, new_value)  # Cap at reasonable maximum
            else:
                # For float parameters - allow larger changes
                new_params[key] = new_params[key] * random.uniform(0.3, 2.0)
            print(f"  Mutated {key}: {old_value:.2f} -> {new_params[key]:.2f}")
            mutated = True
    
    if not mutated:
        print("  No mutations occurred")
    return new_params

def crossover(parent1: Dict, parent2: Dict) -> Dict:
    """Create a child by combining parameters from two parents with more variation"""
    child = {}
    print("\nCrossover:")
    
    # Add some random variation to the crossover
    for key in parent1:
        if random.random() < 0.5:
            # Take from parent 1 with some random variation
            if key == 'win_rate_window':
                # Keep win_rate_window as integer
                child[key] = max(1, min(50, int(parent1[key] * random.uniform(0.9, 1.1))))
            else:
                child[key] = parent1[key] * random.uniform(0.9, 1.1)
            print(f"  {key}: {child[key]:.2f} (from parent 1 with variation)")
        else:
            # Take from parent 2 with some random variation
            if key == 'win_rate_window':
                # Keep win_rate_window as integer
                child[key] = max(1, min(50, int(parent2[key] * random.uniform(0.9, 1.1))))
            else:
                child[key] = parent2[key] * random.uniform(0.9, 1.1)
            print(f"  {key}: {child[key]:.2f} (from parent 2 with variation)")
    
    return child

def validate_params(params: Dict) -> Dict:
    """Ensure all parameters are of correct type and within valid ranges"""
    validated = params.copy()
    
    # Ensure win_rate_window is an integer
    if 'win_rate_window' in validated:
        validated['win_rate_window'] = max(1, min(50, int(validated['win_rate_window'])))
    
    # Ensure all other parameters are positive
    for key, value in validated.items():
        if key != 'win_rate_window':
            validated[key] = max(0.1, float(value))  # Minimum positive value
    
    return validated

def train_and_evaluate_single(params: Dict, data: pd.DataFrame, individual_id: int, run_id: int) -> Dict:
    """Train and evaluate a single model with given parameters for one run"""
    # Validate parameters before use
    params = validate_params(params)
    
    print(f"Individual {individual_id}, Run {run_id}: Starting training...")
    
    try:
        # Create environment with parameters
        env = TradingEnvironment(data, **params)
        
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
        
        model.learn(total_timesteps=50000)
        
        # Evaluate on the same data (you can split this differently if needed)
        obs, _ = env.reset()
        done = False
        total_reward = 0
        
        while not done:
            action, _ = model.predict(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
        
        # Calculate metrics
        final_value = env.total_value
        returns = (final_value - env.initial_balance) / env.initial_balance * 100
        win_rate = sum(1 for trade in env.closed_trades if trade['profit'] > 0) / len(env.closed_trades) if env.closed_trades else 0
        
        print(f"Individual {individual_id}, Run {run_id}: Returns: {returns:.2f}%, Win Rate: {win_rate:.2f}, Trades: {len(env.closed_trades)}")
        
        return {
            'individual_id': individual_id,
            'run_id': run_id,
            'params': params,
            'total_reward': total_reward,
            'final_value': final_value,
            'returns': returns,
            'win_rate': win_rate,
            'num_trades': len(env.closed_trades),
            'success': True
        }
    
    except Exception as e:
        print(f"Individual {individual_id}, Run {run_id}: Error - {e}")
        # Return poor performance for failed runs
        return {
            'individual_id': individual_id,
            'run_id': run_id,
            'params': params,
            'total_reward': -1000,
            'final_value': 5000,  # Half the initial balance
            'returns': -50.0,
            'win_rate': 0.0,
            'num_trades': 0,
            'success': False
        }

def evaluate_population_parallel(population: List[Dict], data: pd.DataFrame, generation: int, n_runs: int = 3) -> List[Dict]:
    """Evaluate all individuals in the population in parallel across multiple runs"""
    print(f"\n{'='*50}")
    print(f"Evaluating Generation {generation}")
    print(f"Population size: {len(population)}, Runs per individual: {n_runs}")
    print(f"Total parallel tasks: {len(population) * n_runs}")
    print(f"{'='*50}")
    
    # Create all tasks for parallel execution
    all_tasks = []
    for i, params in enumerate(population):
        params = validate_params(params)  # Validate parameters
        for run in range(n_runs):
            all_tasks.append((params, data, i + 1, run + 1))
    
    print(f"Created {len(all_tasks)} tasks for parallel execution")
    
    # Use all available cores for maximum parallelization
    num_cores = min(len(all_tasks), multiprocessing.cpu_count())
    print(f"Using {num_cores} CPU cores for parallel processing")
    
    # Run all tasks in parallel
    with multiprocessing.Pool(processes=num_cores) as pool:
        all_results = pool.starmap(train_and_evaluate_single, all_tasks)
    
    print("All parallel tasks completed. Aggregating results...")
    
    # Group results by individual and calculate averages
    individual_results = {}
    for result in all_results:
        individual_id = result['individual_id']
        if individual_id not in individual_results:
            individual_results[individual_id] = []
        individual_results[individual_id].append(result)
    
    # Calculate averaged results for each individual
    final_results = []
    for individual_id in sorted(individual_results.keys()):
        results = individual_results[individual_id]
        successful_results = [r for r in results if r['success']]
        
        if successful_results:
            avg_result = {
                'params': results[0]['params'],  # All runs have same params
                'avg_returns': np.mean([r['returns'] for r in successful_results]),
                'avg_win_rate': np.mean([r['win_rate'] for r in successful_results]),
                'avg_num_trades': np.mean([r['num_trades'] for r in successful_results]),
                'std_returns': np.std([r['returns'] for r in successful_results]),
                'successful_runs': len(successful_results),
                'total_runs': len(results)
            }
        else:
            # All runs failed
            avg_result = {
                'params': results[0]['params'],
                'avg_returns': -50.0,
                'avg_win_rate': 0.0,
                'avg_num_trades': 0.0,
                'std_returns': 0.0,
                'successful_runs': 0,
                'total_runs': len(results)
            }
        
        final_results.append(avg_result)
        
        print(f"\nIndividual {individual_id} Final Results:")
        print(f"  Average Returns: {avg_result['avg_returns']:.2f}%")
        print(f"  Average Win Rate: {avg_result['avg_win_rate']:.2f}")
        print(f"  Average Number of Trades: {avg_result['avg_num_trades']:.1f}")
        print(f"  Returns Std Dev: {avg_result['std_returns']:.2f}%")
        print(f"  Successful Runs: {avg_result['successful_runs']}/{avg_result['total_runs']}")
    
    # Sort by average returns (best first)
    final_results.sort(key=lambda x: x['avg_returns'], reverse=True)
    
    print(f"\n{'='*50}")
    print("Generation evaluation completed!")
    print(f"Best individual: {final_results[0]['avg_returns']:.2f}% returns")
    print(f"{'='*50}")
    
    return final_results

class GeneticOptimizer:
    def __init__(self, population_size: int = 10, elite_size: int = 2, mutation_rate: float = 0.3):
        self.population_size = population_size
        self.elite_size = elite_size
        self.mutation_rate = mutation_rate
        self.device = get_device()
        self.generation = 0
        self.best_params = None
        self.best_score = float('-inf')
        
    def create_initial_population(self) -> List[Dict]:
        """Create initial random population with more variation"""
        print("\nCreating initial population...")
        population = []
        
        # Add some completely random individuals
        for _ in range(self.population_size // 2):
            population.append(generate_random_params())
        
        # Add some individuals with extreme values
        for _ in range(self.population_size // 2):
            params = generate_random_params()
            # Randomly select some parameters to set to extreme values
            for key in params:
                if random.random() < 0.3:  # 30% chance to set extreme value
                    if key == 'win_rate_window':
                        params[key] = random.choice([3, 30])  # Min or max value, keep as int
                    else:
                        params[key] = random.choice([params[key] * 0.1, params[key] * 10])  # Extreme low or high
            population.append(params)
        
        # Validate all parameters in initial population
        population = [validate_params(params) for params in population]
        
        print(f"Generated {len(population)} parameter sets with high variation")
        return population
    
    def evaluate_population(self, population: List[Dict], data: pd.DataFrame) -> List[Dict]:
        """Evaluate all individuals in the population using maximum parallelization"""
        return evaluate_population_parallel(population, data, self.generation, n_runs=3)
    
    def create_next_generation(self, evaluated_population: List[Dict]) -> List[Dict]:
        """Create next generation using genetic operators with more variation"""
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
            print(f"  Successful Runs: {elite['successful_runs']}/{elite['total_runs']}")
            for param, value in elite['params'].items():
                if param == 'win_rate_window':
                    print(f"  {param}: {value}")
                else:
                    print(f"  {param}: {value:.2f}")
            next_generation.append(elite['params'])
        
        # Create rest of population through crossover and mutation
        print("\nCreating new individuals through crossover and mutation:")
        while len(next_generation) < self.population_size:
            print(f"\nCreating individual {len(next_generation) + 1}/{self.population_size}")
            
            # Tournament selection with more randomness
            if random.random() < 0.3:  # 30% chance to select from entire population
                parent1 = random.choice(evaluated_population)['params']
                parent2 = random.choice(evaluated_population)['params']
            else:
                # Select from top half with some randomness
                parent1 = random.choice(evaluated_population[:self.population_size//2])['params']
                parent2 = random.choice(evaluated_population[:self.population_size//2])['params']
            
            print("\nSelected parents:")
            print("Parent 1:", {k: f"{v}" if k == 'win_rate_window' else f"{v:.2f}" for k, v in parent1.items()})
            print("Parent 2:", {k: f"{v}" if k == 'win_rate_window' else f"{v:.2f}" for k, v in parent2.items()})
            
            # Crossover
            child = crossover(parent1, parent2)
            
            # Mutation with increased rate for some parameters
            print("\nApplying mutations:")
            child = mutate_params(child, self.mutation_rate)
            
            # Occasionally add a completely random individual
            if random.random() < 0.2:  # 20% chance
                print("\nAdding random variation to individual")
                for key in child:
                    if random.random() < 0.3:  # 30% chance to randomize each parameter
                        child[key] = generate_random_params()[key]
            
            # Validate the child parameters
            child = validate_params(child)
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
        print(f"Available CPU cores: {multiprocessing.cpu_count()}")
        print(f"Max parallel tasks per generation: {self.population_size * 3}")
        print(f"{'='*50}")
        
        # Create initial population
        population = self.create_initial_population()
        
        for gen in range(generations):
            self.generation = gen + 1
            print(f"\n{'='*70}")
            print(f"STARTING GENERATION {self.generation}")
            print(f"{'='*70}")
            
            # Evaluate current population (all individuals in parallel)
            evaluated_population = self.evaluate_population(population, data)
            
            # Update best parameters if found
            if evaluated_population[0]['avg_returns'] > self.best_score:
                self.best_score = evaluated_population[0]['avg_returns']
                self.best_params = evaluated_population[0]['params']
                print(f"\n{'='*50}")
                print("NEW BEST PARAMETERS FOUND!")
                print(f"{'='*50}")
                print(f"Returns: {self.best_score:.2f}%")
                print(f"Successful runs: {evaluated_population[0]['successful_runs']}/{evaluated_population[0]['total_runs']}")
                print("Parameters:")
                for param, value in self.best_params.items():
                    if param == 'win_rate_window':
                        print(f"  {param}: {value}")
                    else:
                        print(f"  {param}: {value:.2f}")
            
            # Create next generation
            if gen < generations - 1:  # Don't create next generation on last iteration
                population = self.create_next_generation(evaluated_population)
            
            # Save progress
            self.save_generation_results(evaluated_population)
            
            print(f"\n{'='*70}")
            print(f"GENERATION {self.generation} COMPLETED")
            print(f"Best so far: {self.best_score:.2f}%")
            print(f"{'='*70}")
        
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
    
    # Display system info
    print(f"\nSystem Info:")
    print(f"CPU cores: {multiprocessing.cpu_count()}")
    print(f"Recommended for max parallelization")
    
    # Create and run genetic optimizer
    optimizer = GeneticOptimizer(
        population_size=10,  # 10 individuals
        elite_size=2,       # Keep top 2
        mutation_rate=0.3   # High mutation rate for exploration
    )
    
    # Run optimization
    best_params, best_score = optimizer.optimize(df, generations=5)
    
    print("\n" + "="*70)
    print("OPTIMIZATION COMPLETED!")
    print("="*70)
    print("\nBest parameters found:")
    for param, value in best_params.items():
        if param == 'win_rate_window':
            print(f"{param}: {value}")
        else:
            print(f"{param}: {value:.2f}")
    print(f"\nBest score: {best_score:.2f}%")
    
    # Save final results to a separate file
    final_results = {
        'best_params': best_params,
        'best_score': best_score,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    with open('final_optimization_results.json', 'w') as f:
        json.dump(final_results, f, indent=4)
    print("\nFinal results saved to 'final_optimization_results.json'")

if __name__ == "__main__":
    # Set start method for multiprocessing
    multiprocessing.set_start_method('spawn')
    main()