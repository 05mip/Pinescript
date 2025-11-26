import os
import glob
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from backtesting import Backtest
import random
from tqdm import tqdm
from typing import Dict, List, Tuple, Callable, Any
import json

# 1. Load all CSVs in data folder
def load_data(file):
    df = pd.read_csv(file, skiprows=2)
    df = df.rename(columns={
        df.columns[0]: 'timestamp',
        df.columns[1]: 'Close',
        df.columns[2]: 'High',
        df.columns[3]: 'Low',
        df.columns[4]: 'Open',
        df.columns[5]: 'Volume'
    })
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df.set_index('timestamp', inplace=True)
    for col in ['Open', 'High', 'Low', 'Close']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=['Open', 'High', 'Low', 'Close'])
    return df

METRICS = ['Equity Final [$]', 'Win Rate [%]', 'Sharpe Ratio', 'Avg. Trade']

def get_avg_trade(stats):
    for key in ['Avg. Trade', 'Avg. Trade [%]', 'Avg. Trade [$]', 'Avg. Trade Net [%]', 'Avg. Trade Net [$]']:
        if key in stats:
            return stats[key]
    return float('nan')

def run_backtest(data, strategy, params, verbose=False, label=None):
    try:
        if verbose and label:
            print(f"  Running {label} backtest with params: {params}")
        bt = Backtest(data, strategy, cash=1000000, commission=.001, trade_on_close=False)
        stats = bt.run(**params)
        metrics = {m: stats.get(m, np.nan) for m in METRICS}
        metrics['Avg. Trade'] = get_avg_trade(stats)
        if verbose and label:
            print(f"    {label} metrics: " + ", ".join([f'{m}: {metrics[m]:.4f}' for m in METRICS]))
        return metrics, params
    except Exception as e:
        if verbose:
            print(f"    Error in {label} backtest: {e}")
            print(f"    Params: {params}")
        return {m: np.nan for m in METRICS}, params

def permute_data(df, start_index=0, seed=None):
    np.random.seed(seed)
    df = df.copy()
    n_bars = len(df)
    perm_index = start_index + 1
    perm_n = n_bars - perm_index
    if perm_n <= 0:
        return df
    log_bars = np.log(df[['Open', 'High', 'Low', 'Close']])
    start_bar = log_bars.iloc[start_index].to_numpy()
    r_o = (log_bars['Open'] - log_bars['Close'].shift()).to_numpy()
    r_h = (log_bars['High'] - log_bars['Open']).to_numpy()
    r_l = (log_bars['Low'] - log_bars['Open']).to_numpy()
    r_c = (log_bars['Close'] - log_bars['Open']).to_numpy()
    relative_open = r_o[perm_index:]
    relative_high = r_h[perm_index:]
    relative_low = r_l[perm_index:]
    relative_close = r_c[perm_index:]
    idx = np.arange(perm_n)
    perm1 = np.random.permutation(idx)
    perm2 = np.random.permutation(idx)
    relative_high = relative_high[perm1]
    relative_low = relative_low[perm1]
    relative_close = relative_close[perm1]
    relative_open = relative_open[perm2]
    perm_bars = np.zeros((n_bars, 4))
    log_bars_np = log_bars.to_numpy().copy()
    perm_bars[:start_index] = log_bars_np[:start_index]
    perm_bars[start_index] = start_bar
    for i in range(perm_index, n_bars):
        k = i - perm_index
        perm_bars[i, 0] = perm_bars[i - 1, 3] + relative_open[k]
        perm_bars[i, 1] = perm_bars[i, 0] + relative_high[k]
        perm_bars[i, 2] = perm_bars[i, 0] + relative_low[k]
        perm_bars[i, 3] = perm_bars[i, 0] + relative_close[k]
    perm_bars = np.exp(perm_bars)
    perm_bars = pd.DataFrame(perm_bars, index=df.index, columns=['Open', 'High', 'Low', 'Close'])
    if 'Volume' in df.columns:
        perm_bars['Volume'] = df['Volume'].values
    return perm_bars

def evaluate_params_across_datasets(strategy, params, datasets, verbose=False):
    """Evaluate a parameter set across all datasets and return aggregated metrics."""
    results = []
    
    for i, data in enumerate(datasets):
        metrics, _ = run_backtest(data, strategy, params, verbose=verbose, 
                                 label=f"Dataset_{i+1}")
        results.append(metrics)
    
    # Aggregate metrics across datasets
    aggregated = {}
    for metric in METRICS:
        values = [r[metric] for r in results if not np.isnan(r[metric])]
        if values:
            aggregated[f'{metric}_mean'] = np.mean(values)
            aggregated[f'{metric}_std'] = np.std(values)
            aggregated[f'{metric}_min'] = np.min(values)
            aggregated[f'{metric}_max'] = np.max(values)
        else:
            aggregated[f'{metric}_mean'] = np.nan
            aggregated[f'{metric}_std'] = np.nan
            aggregated[f'{metric}_min'] = np.nan
            aggregated[f'{metric}_max'] = np.nan
    
    aggregated['individual_results'] = results
    return aggregated

def mc_optimize(strategy, param_space, data_dir=None, n_iterations=1000, 
                optimization_metric='Equity Final [$]_mean', minimize=False,
                validation_permutations=100, n_threads=4, verbose=False):
    """
    Proper Monte Carlo optimization for backtesting strategies.
    
    Args:
        strategy: The strategy class (from backtesting lib)
        param_space: Dict of parameter sampling lambdas
        data_dir: Directory containing CSVs (default: '../data')
        n_iterations: Number of parameter combinations to try
        optimization_metric: Metric to optimize (with _mean, _std, _min, _max suffix)
        minimize: Whether to minimize the metric (default: maximize)
        validation_permutations: Number of permutations for statistical validation
        n_threads: Number of threads to use
        verbose: Print progress
        
    Returns:
        dict with optimization results and statistical validation
    """
    
    # Load datasets
    if data_dir is None:
        data_dir = os.path.join(os.path.dirname(__file__), '../data_30m')
    
    csv_files = glob.glob(os.path.join(data_dir, '*.csv'))
    if not csv_files:
        raise ValueError(f"No CSV files found in {data_dir}")
    
    if verbose:
        print(f"Found {len(csv_files)} CSV files in {data_dir}:")
        for f in csv_files:
            print(f"  - {os.path.basename(f)}")
    
    datasets = []
    for f in csv_files:
        df = load_data(f)
        if df is not None and len(df) > 0:
            datasets.append(df)
    
    if not datasets:
        raise ValueError("No valid datasets loaded")
    
    if verbose:
        print(f"Loaded {len(datasets)} datasets.")
        print(f"Optimizing for: {optimization_metric}")
    
    # Phase 1: Parameter Optimization
    if verbose:
        print(f"\nPhase 1: Parameter Optimization ({n_iterations} iterations)")
    
    def evaluate_single_params(i):
        params = {k: v() for k, v in param_space.items()}
        aggregated_metrics = evaluate_params_across_datasets(
            strategy, params, datasets, verbose=verbose and i < 3
        )
        return params, aggregated_metrics
    
    optimization_results = []
    
    with ThreadPoolExecutor(max_workers=n_threads) as executor:
        futures = [executor.submit(evaluate_single_params, i) for i in range(n_iterations)]
        
        for f in tqdm(as_completed(futures), total=n_iterations, desc="Optimizing parameters"):
            params, metrics = f.result()
            optimization_results.append({
                'params': params,
                'metrics': metrics,
                'objective_value': metrics.get(optimization_metric, np.nan)
            })
    
    # Find best parameters
    valid_results = [r for r in optimization_results if not np.isnan(r['objective_value'])]
    
    if not valid_results:
        raise ValueError("No valid parameter combinations found")
    
    if minimize:
        best_result = min(valid_results, key=lambda x: x['objective_value'])
    else:
        best_result = max(valid_results, key=lambda x: x['objective_value'])
    
    best_params = best_result['params']
    best_metrics = best_result['metrics']
    
    if verbose:
        print(f"\nBest parameters found: {best_params}")
        print(f"Best {optimization_metric}: {best_result['objective_value']:.4f}")
    
    # Phase 2: Statistical Validation via Permutation Testing
    if verbose:
        print(f"\nPhase 2: Statistical Validation ({validation_permutations} permutations)")
    
    def run_permutation_test(i):
        # Test best params on each dataset vs its permutation
        real_results = []
        perm_results = []
        
        for j, data in enumerate(datasets):
            # Real data
            real_metrics, _ = run_backtest(data, strategy, best_params, 
                                         verbose=verbose and i < 3, 
                                         label=f"Real_Dataset_{j+1}")
            real_results.append(real_metrics)
            
            # Permuted data
            perm_data = permute_data(data, seed=i*len(datasets)+j)
            perm_metrics, _ = run_backtest(perm_data, strategy, best_params, 
                                         verbose=verbose and i < 3, 
                                         label=f"Perm_Dataset_{j+1}")
            perm_results.append(perm_metrics)
        
        # Aggregate results
        real_agg = np.mean([r[optimization_metric.replace('_mean', '')] 
                           for r in real_results if not np.isnan(r[optimization_metric.replace('_mean', '')])])
        perm_agg = np.mean([r[optimization_metric.replace('_mean', '')] 
                           for r in perm_results if not np.isnan(r[optimization_metric.replace('_mean', '')])])
        
        return real_agg, perm_agg
    
    validation_results = []
    
    with ThreadPoolExecutor(max_workers=n_threads) as executor:
        futures = [executor.submit(run_permutation_test, i) for i in range(validation_permutations)]
        
        for f in tqdm(as_completed(futures), total=validation_permutations, desc="Validating significance"):
            real_agg, perm_agg = f.result()
            validation_results.append({'real': real_agg, 'permuted': perm_agg})
    
    # Calculate p-value
    real_values = [r['real'] for r in validation_results if not np.isnan(r['real'])]
    perm_values = [r['permuted'] for r in validation_results if not np.isnan(r['permuted'])]
    
    if real_values and perm_values:
        best_real_performance = np.mean(real_values)
        if minimize:
            p_value = np.mean([p <= best_real_performance for p in perm_values])
        else:
            p_value = np.mean([p >= best_real_performance for p in perm_values])
    else:
        p_value = np.nan
    
    # Final evaluation on original datasets
    final_evaluation = evaluate_params_across_datasets(
        strategy, best_params, datasets, verbose=verbose
    )
    
    return {
        'best_params': best_params,
        'best_metrics': best_metrics,
        'optimization_metric': optimization_metric,
        'optimization_value': best_result['objective_value'],
        'p_value': p_value,
        'final_evaluation': final_evaluation,
        'all_optimization_results': optimization_results,
        'validation_results': validation_results,
        'n_datasets': len(datasets),
        'n_iterations': n_iterations,
        'validation_permutations': validation_permutations
    }

def print_optimization_summary(results):
    """Print a summary of optimization results."""
    print("\n" + "="*60)
    print("MONTE CARLO OPTIMIZATION SUMMARY")
    print("="*60)
    
    print(f"Datasets used: {results['n_datasets']}")
    print(f"Parameter combinations tested: {results['n_iterations']}")
    print(f"Validation permutations: {results['validation_permutations']}")
    print(f"Optimization metric: {results['optimization_metric']}")
    
    print(f"\nBest Parameters:")
    for k, v in results['best_params'].items():
        print(f"  {k}: {v}")
    
    print(f"\nOptimization Performance:")
    print(f"  {results['optimization_metric']}: {results['optimization_value']:.4f}")
    
    print(f"\nStatistical Validation:")
    print(f"  P-value: {results['p_value']:.4f}")
    significance = "significant" if results['p_value'] < 0.05 else "not significant"
    print(f"  Result: {significance} at α=0.05")
    
    print(f"\nFinal Evaluation Across All Datasets:")
    final_eval = results['final_evaluation']
    for metric in METRICS:
        mean_val = final_eval.get(f'{metric}_mean', np.nan)
        std_val = final_eval.get(f'{metric}_std', np.nan)
        print(f"  {metric}: {mean_val:.4f} ± {std_val:.4f}")