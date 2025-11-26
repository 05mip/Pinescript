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
from sklearn.preprocessing import StandardScaler
from dataclasses import dataclass

# Original data loading and permutation functions remain the same
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

@dataclass
class ScoringConfig:
    """Configuration for multi-metric scoring system"""
    
    # Profitability metrics (higher is better)
    profitability_metrics: List[str] = None
    
    # Risk-adjusted metrics (higher is better)  
    risk_adjusted_metrics: List[str] = None
    
    # Consistency metrics (higher is better)
    consistency_metrics: List[str] = None
    
    # Penalty metrics (lower is better)
    penalty_metrics: List[str] = None
    
    # Gaming prevention
    min_trades_threshold: int = 5
    min_trades_penalty: float = -10.0  # Heavy penalty for insufficient trades
    max_volatility_threshold: float = 2000.0  # Annualized volatility %
    high_volatility_penalty: float = -5.0
    max_drawdown_threshold: float = -50.0  # Max drawdown %
    excessive_drawdown_penalty: float = -5.0
    
    # Scoring weights for different categories
    profitability_weight: float = 1.0
    risk_adjusted_weight: float = 1.5  # Emphasize risk-adjusted returns
    consistency_weight: float = 1.2
    penalty_weight: float = 2.0  # Heavy penalty weighting
    
    def __post_init__(self):
        if self.profitability_metrics is None:
            self.profitability_metrics = [
                'Return [%]', 'CAGR [%]', 'Expectancy [%]', 'Profit Factor'
            ]
        
        if self.risk_adjusted_metrics is None:
            self.risk_adjusted_metrics = [
                'Sharpe Ratio', 'Sortino Ratio', 'Calmar Ratio', 'SQN'
            ]
            
        if self.consistency_metrics is None:
            self.consistency_metrics = [
                'Win Rate [%]', '# Trades', 'Kelly Criterion'
            ]
            
        if self.penalty_metrics is None:
            self.penalty_metrics = [
                'Max. Drawdown [%]', 'Avg. Drawdown [%]', 'Volatility (Ann.) [%]'
            ]

def get_avg_trade(stats):
    for key in ['Avg. Trade', 'Avg. Trade [%]', 'Avg. Trade [$]', 'Avg. Trade Net [%]', 'Avg. Trade Net [$]']:
        if key in stats:
            return stats[key]
    return float('nan')

def extract_comprehensive_metrics(stats):
    """Extract all available metrics from backtest results"""
    metrics = {}
    
    # Core metrics we always want
    core_metrics = [
        'Return [%]', 'CAGR [%]', 'Sharpe Ratio', 'Sortino Ratio', 'Calmar Ratio',
        'Max. Drawdown [%]', 'Avg. Drawdown [%]', 'Win Rate [%]', '# Trades',
        'Profit Factor', 'Expectancy [%]', 'SQN', 'Kelly Criterion',
        'Volatility (Ann.) [%]', 'Alpha [%]', 'Beta'
    ]
    
    for metric in core_metrics:
        metrics[metric] = stats.get(metric, np.nan)
    
    # Handle avg trade specially
    metrics['Avg. Trade [%]'] = get_avg_trade(stats)
    
    return metrics

def run_backtest(data, strategy, params, verbose=False, label=None):
    try:
        if verbose and label:
            print(f"  Running {label} backtest with params: {params}")
        bt = Backtest(data, strategy, cash=1000000, commission=.001, trade_on_close=False)
        stats = bt.run(**params)
        
        trades_count = stats.get('# Trades', 0)
        
        if verbose and label:
            print(f"    Number of trades: {trades_count}")
        
        # Extract comprehensive metrics
        metrics = extract_comprehensive_metrics(stats)
        
        # Handle no trades case
        if trades_count == 0:
            if verbose:
                print(f"    Warning: No trades executed for {label} with these params.")
            # Set appropriate defaults for no-trade scenarios
            metrics['Win Rate [%]'] = 0.0
            metrics['Avg. Trade [%]'] = 0.0
            metrics['Profit Factor'] = 0.0
            metrics['Expectancy [%]'] = 0.0
            metrics['SQN'] = 0.0
        
        if verbose and label:
            key_metrics = ['Return [%]', 'Sharpe Ratio', '# Trades', 'Max. Drawdown [%]']
            print(f"    {label} key metrics: " + ", ".join([
                f'{m}: {metrics[m]:.4f}' if not pd.isna(metrics[m]) else f'{m}: nan' 
                for m in key_metrics if m in metrics
            ]))
        
        return metrics, params
    except Exception as e:
        if verbose:
            print(f"    Error in {label} backtest: {e}")
            print(f"    Params: {params}")
        # Return NaN metrics for failed backtests
        return {k: np.nan for k in extract_comprehensive_metrics({}).keys()}, params

def calculate_composite_score(metrics_list: List[Dict], config: ScoringConfig = None, verbose: bool = False):
    """
    Calculate composite scores using rank-based normalization
    
    Args:
        metrics_list: List of metric dictionaries from different parameter combinations
        config: Scoring configuration
        verbose: Print scoring details
    
    Returns:
        List of composite scores corresponding to input metrics
    """
    if config is None:
        config = ScoringConfig()
    
    if len(metrics_list) == 0:
        return []
    
    # Convert to DataFrame for easier processing
    df = pd.DataFrame(metrics_list)
    
    # Fill NaN values with worst possible scores for ranking
    df_filled = df.copy()
    for col in df.columns:
        if col in config.penalty_metrics:
            # For penalty metrics, NaN should be treated as worst (most negative)
            df_filled[col] = df_filled[col].fillna(df_filled[col].min() - 1)
        else:
            # For benefit metrics, NaN should be treated as worst (lowest)
            df_filled[col] = df_filled[col].fillna(df_filled[col].min() - 1)
    
    # Initialize score components
    n_strategies = len(df_filled)
    composite_scores = np.zeros(n_strategies)
    score_components = {
        'profitability': np.zeros(n_strategies),
        'risk_adjusted': np.zeros(n_strategies),
        'consistency': np.zeros(n_strategies),
        'penalties': np.zeros(n_strategies),
        'gaming_penalties': np.zeros(n_strategies)
    }
    
    # Helper function to normalize metrics to z-scores
    def rank_normalize(series, reverse=False):
        """Convert series to rank-based z-scores"""
        if reverse:
            # For penalty metrics, lower values should get higher scores
            ranks = (-series).rank(method='min')
        else:
            # For benefit metrics, higher values should get higher scores  
            ranks = series.rank(method='min')
        
        # Convert ranks to z-scores (mean=0, std=1)
        if ranks.std() > 0:
            return (ranks - ranks.mean()) / ranks.std()
        else:
            return np.zeros(len(ranks))
    
    # Process profitability metrics
    for metric in config.profitability_metrics:
        if metric in df_filled.columns:
            normalized = rank_normalize(df_filled[metric])
            score_components['profitability'] += normalized
            if verbose:
                print(f"Profitability - {metric}: mean={normalized.mean():.3f}, std={normalized.std():.3f}")
    
    # Process risk-adjusted metrics
    for metric in config.risk_adjusted_metrics:
        if metric in df_filled.columns:
            normalized = rank_normalize(df_filled[metric])
            score_components['risk_adjusted'] += normalized
            if verbose:
                print(f"Risk-Adjusted - {metric}: mean={normalized.mean():.3f}, std={normalized.std():.3f}")
    
    # Process consistency metrics
    for metric in config.consistency_metrics:
        if metric in df_filled.columns:
            normalized = rank_normalize(df_filled[metric])
            score_components['consistency'] += normalized
            if verbose:
                print(f"Consistency - {metric}: mean={normalized.mean():.3f}, std={normalized.std():.3f}")
    
    # Process penalty metrics (reverse scoring)
    for metric in config.penalty_metrics:
        if metric in df_filled.columns:
            normalized = rank_normalize(df_filled[metric], reverse=True)
            score_components['penalties'] += normalized
            if verbose:
                print(f"Penalty - {metric}: mean={normalized.mean():.3f}, std={normalized.std():.3f}")
    
    # Apply gaming prevention penalties
    if '# Trades' in df_filled.columns:
        insufficient_trades = df_filled['# Trades'] < config.min_trades_threshold
        score_components['gaming_penalties'][insufficient_trades] += config.min_trades_penalty
        if verbose and insufficient_trades.sum() > 0:
            print(f"Gaming Penalty - Insufficient trades: {insufficient_trades.sum()} strategies penalized")
    
    if 'Volatility (Ann.) [%]' in df_filled.columns:
        high_volatility = df_filled['Volatility (Ann.) [%]'] > config.max_volatility_threshold
        score_components['gaming_penalties'][high_volatility] += config.high_volatility_penalty
        if verbose and high_volatility.sum() > 0:
            print(f"Gaming Penalty - High volatility: {high_volatility.sum()} strategies penalized")
    
    if 'Max. Drawdown [%]' in df_filled.columns:
        excessive_drawdown = df_filled['Max. Drawdown [%]'] < config.max_drawdown_threshold
        score_components['gaming_penalties'][excessive_drawdown] += config.excessive_drawdown_penalty
        if verbose and excessive_drawdown.sum() > 0:
            print(f"Gaming Penalty - Excessive drawdown: {excessive_drawdown.sum()} strategies penalized")
    
    # Calculate weighted composite score
    composite_scores = (
        score_components['profitability'] * config.profitability_weight +
        score_components['risk_adjusted'] * config.risk_adjusted_weight +
        score_components['consistency'] * config.consistency_weight +
        score_components['penalties'] * config.penalty_weight +
        score_components['gaming_penalties']  # Gaming penalties are already weighted
    )
    
    if verbose:
        print(f"\nComposite Score Statistics:")
        print(f"  Mean: {composite_scores.mean():.3f}")
        print(f"  Std: {composite_scores.std():.3f}")
        print(f"  Min: {composite_scores.min():.3f}")
        print(f"  Max: {composite_scores.max():.3f}")
    
    return composite_scores.tolist(), score_components

def evaluate_params_across_datasets(strategy, params, datasets, verbose=False):
    """Evaluate a parameter set across all datasets and return comprehensive metrics."""
    results = []
    
    for i, data in enumerate(datasets):
        metrics, _ = run_backtest(data, strategy, params, verbose=verbose, 
                                 label=f"Dataset_{i+1}")
        results.append(metrics)
    
    # Get all possible metric keys
    all_metrics = set()
    for result in results:
        all_metrics.update(result.keys())
    
    # Aggregate metrics across datasets
    aggregated = {}
    for metric in all_metrics:
        values = [r.get(metric, np.nan) for r in results]
        valid_values = [v for v in values if not np.isnan(v)]
        
        if valid_values:
            aggregated[f'{metric}_mean'] = np.mean(valid_values)
            aggregated[f'{metric}_std'] = np.std(valid_values)
            aggregated[f'{metric}_min'] = np.min(valid_values)
            aggregated[f'{metric}_max'] = np.max(valid_values)
        else:
            aggregated[f'{metric}_mean'] = np.nan
            aggregated[f'{metric}_std'] = np.nan
            aggregated[f'{metric}_min'] = np.nan
            aggregated[f'{metric}_max'] = np.nan
    
    aggregated['individual_results'] = results
    return aggregated

def validate_top_params(strategy, top_params_list, datasets, validation_permutations=100, 
                       scoring_config=None, n_threads=4, verbose=False):
    """Validate the top parameter sets with permutation testing using composite scores."""
    
    if scoring_config is None:
        scoring_config = ScoringConfig()
    
    def run_permutation_test_for_params(params_idx_and_params):
        params_idx, params = params_idx_and_params
        
        # Test params on each dataset vs its permutation
        real_results = []
        perm_results = []
        
        for j, data in enumerate(datasets):
            # Real data
            real_metrics, _ = run_backtest(data, strategy, params, 
                                         verbose=False, 
                                         label=f"Real_Dataset_{j+1}")
            real_results.append(real_metrics)
            
            # Permuted data
            perm_data = permute_data(data, seed=params_idx*len(datasets)+j)
            perm_metrics, _ = run_backtest(perm_data, strategy, params, 
                                         verbose=False, 
                                         label=f"Perm_Dataset_{j+1}")
            perm_results.append(perm_metrics)
        
        # Calculate composite scores for real vs permuted
        real_composite, _ = calculate_composite_score(real_results, scoring_config)
        perm_composite, _ = calculate_composite_score(perm_results, scoring_config)
        
        real_agg = np.mean(real_composite) if real_composite else np.nan
        perm_agg = np.mean(perm_composite) if perm_composite else np.nan
        
        return params_idx, real_agg, perm_agg
    
    validation_results = {}
    
    if verbose:
        print(f"Running validation for top {len(top_params_list)} parameter sets...")
    
    # Run validation for each parameter set
    for params_idx, result in enumerate(top_params_list):
        params = result['params']
        validation_data = []
        
        with ThreadPoolExecutor(max_workers=n_threads) as executor:
            futures = [executor.submit(run_permutation_test_for_params, (params_idx, params)) 
                      for _ in range(validation_permutations)]
            
            for f in tqdm(as_completed(futures), total=validation_permutations, 
                         desc=f"Validating params {params_idx+1}/{len(top_params_list)}"):
                _, real_agg, perm_agg = f.result()
                validation_data.append({'real': real_agg, 'permuted': perm_agg})
        
        # Calculate p-value (composite score is always maximize)
        real_values = [r['real'] for r in validation_data if not np.isnan(r['real'])]
        perm_values = [r['permuted'] for r in validation_data if not np.isnan(r['permuted'])]
        
        if real_values and perm_values:
            best_real_performance = np.mean(real_values)
            p_value = np.mean([p >= best_real_performance for p in perm_values])
        else:
            p_value = np.nan
        
        validation_results[params_idx] = {
            'p_value': p_value,
            'validation_data': validation_data
        }
    
    return validation_results

def mc_optimize_composite(strategy, param_space, data_dir=None, n_iterations=1000, 
                         scoring_config=None, validation_permutations=100, n_threads=4, 
                         verbose=False, output_file='top10_composite_results.json', top_n=10):
    """
    Monte Carlo optimization using composite scoring system.
    
    Args:
        strategy: The strategy class (from backtesting lib)
        param_space: Dict of parameter sampling lambdas
        data_dir: Directory containing CSVs (default: '../data_30m')
        n_iterations: Number of parameter combinations to try
        scoring_config: ScoringConfig object for composite scoring
        validation_permutations: Number of permutations for statistical validation
        n_threads: Number of threads to use
        verbose: Print progress
        output_file: JSON file to save results
        top_n: Number of top results to return and validate
        
    Returns:
        dict with top N optimization results and statistical validation
    """
    
    if scoring_config is None:
        scoring_config = ScoringConfig()
    
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
        print(f"Using composite scoring with {len(scoring_config.profitability_metrics)} profitability, "
              f"{len(scoring_config.risk_adjusted_metrics)} risk-adjusted, "
              f"{len(scoring_config.consistency_metrics)} consistency, and "
              f"{len(scoring_config.penalty_metrics)} penalty metrics.")
    
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
                'metrics': metrics
            })
    
    # Calculate composite scores for all results
    if verbose:
        print("\nCalculating composite scores...")
    
    # Extract mean metrics for scoring
    mean_metrics_list = []
    for result in optimization_results:
        mean_metrics = {}
        for key, value in result['metrics'].items():
            if key.endswith('_mean') and not key.startswith('individual_results'):
                metric_name = key.replace('_mean', '')
                mean_metrics[metric_name] = value
        mean_metrics_list.append(mean_metrics)
    
    composite_scores, score_components = calculate_composite_score(
        mean_metrics_list, scoring_config, verbose=verbose
    )
    
    # Add composite scores to results
    for i, score in enumerate(composite_scores):
        optimization_results[i]['composite_score'] = score
        optimization_results[i]['score_components'] = {
            k: v[i] for k, v in score_components.items()
        }
    
    # Find top N parameters based on composite score
    valid_results = [r for r in optimization_results if not np.isnan(r['composite_score'])]
    
    if not valid_results:
        raise ValueError("No valid parameter combinations found")
    
    # Sort by composite score (higher is better)
    top_results = sorted(valid_results, key=lambda x: x['composite_score'], reverse=True)[:top_n]
    
    if verbose:
        print(f"\nTop {len(top_results)} parameter combinations found:")
        for i, result in enumerate(top_results):
            print(f"  {i+1}. {result['params']} -> Composite Score: {result['composite_score']:.4f}")
    
    # Phase 2: Statistical Validation for top N
    if verbose:
        print(f"\nPhase 2: Statistical Validation for top {len(top_results)} parameters ({validation_permutations} permutations each)")
    
    validation_results = validate_top_params(
        strategy, top_results, datasets, validation_permutations,
        scoring_config, n_threads, verbose
    )
    
    # Prepare final results
    final_results = {
        'optimization_settings': {
            'optimization_method': 'composite_scoring',
            'scoring_config': {
                'profitability_metrics': scoring_config.profitability_metrics,
                'risk_adjusted_metrics': scoring_config.risk_adjusted_metrics,
                'consistency_metrics': scoring_config.consistency_metrics,
                'penalty_metrics': scoring_config.penalty_metrics,
                'min_trades_threshold': scoring_config.min_trades_threshold,
                'weights': {
                    'profitability': scoring_config.profitability_weight,
                    'risk_adjusted': scoring_config.risk_adjusted_weight,
                    'consistency': scoring_config.consistency_weight,
                    'penalty': scoring_config.penalty_weight
                }
            },
            'n_iterations': n_iterations,
            'validation_permutations': validation_permutations,
            'n_datasets': len(datasets),
            'dataset_files': [os.path.basename(f) for f in csv_files],
            'top_n': top_n
        },
        'top_results': []
    }
    
    for i, result in enumerate(top_results):
        # Convert numpy types to native Python types for JSON serialization
        params_json = {}
        for k, v in result['params'].items():
            if isinstance(v, (np.integer, np.floating)):
                params_json[k] = v.item()
            else:
                params_json[k] = v
        
        metrics_json = {}
        for k, v in result['metrics'].items():
            if k == 'individual_results':
                # Convert individual results
                individual_json = []
                for ind_result in v:
                    ind_json = {}
                    for metric_k, metric_v in ind_result.items():
                        if isinstance(metric_v, (np.integer, np.floating)):
                            ind_json[metric_k] = metric_v.item() if not np.isnan(metric_v) else None
                        else:
                            ind_json[metric_k] = metric_v
                    individual_json.append(ind_json)
                metrics_json[k] = individual_json
            elif isinstance(v, (np.integer, np.floating)):
                metrics_json[k] = v.item() if not np.isnan(v) else None
            else:
                metrics_json[k] = v
        
        score_components_json = {}
        for k, v in result['score_components'].items():
            if isinstance(v, (np.integer, np.floating)):
                score_components_json[k] = v.item()
            else:
                score_components_json[k] = v
        
        validation_data = validation_results.get(i, {})
        p_value = validation_data.get('p_value', np.nan)
        
        final_results['top_results'].append({
            'rank': i + 1,
            'params': params_json,
            'metrics': metrics_json,
            'composite_score': result['composite_score'].item() if isinstance(result['composite_score'], (np.integer, np.floating)) else result['composite_score'],
            'score_components': score_components_json,
            'p_value': p_value.item() if isinstance(p_value, (np.integer, np.floating)) and not np.isnan(p_value) else None,
            'significant': p_value < 0.05 if not np.isnan(p_value) else None
        })
    
    # Save to JSON file
    def convert_numpy(obj):
        if isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(v) for v in obj]
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        else:
            return obj

    clean_results = convert_numpy(final_results)

    with open(output_file, 'w') as f:
        json.dump(clean_results, f, indent=2)
    
    if verbose:
        print(f"\nResults saved to {output_file}")
    
    return final_results

def print_composite_summary(results):
    """Print a summary of composite scoring optimization results."""
    print("\n" + "="*80)
    print("MONTE CARLO OPTIMIZATION - COMPOSITE SCORING RESULTS SUMMARY")
    print("="*80)
    
    settings = results['optimization_settings']
    print(f"Datasets used: {settings['n_datasets']}")
    print(f"Parameter combinations tested: {settings['n_iterations']}")
    print(f"Validation permutations: {settings['validation_permutations']}")
    print(f"Optimization method: {settings['optimization_method']}")
    print(f"Top N results: {settings['top_n']}")
    
    scoring_config = settings['scoring_config']
    print(f"\nScoring Configuration:")
    print(f"  Profitability metrics: {len(scoring_config['profitability_metrics'])}")
    print(f"  Risk-adjusted metrics: {len(scoring_config['risk_adjusted_metrics'])}")
    print(f"  Consistency metrics: {len(scoring_config['consistency_metrics'])}")
    print(f"  Penalty metrics: {len(scoring_config['penalty_metrics'])}")
    print(f"  Min trades threshold: {scoring_config['min_trades_threshold']}")
    
    print(f"\nTop {len(results['top_results'])} Results:")
    print("-" * 80)
    
    for result in results['top_results']:
        print(f"\nRank {result['rank']}:")
        print(f"  Parameters: {result['params']}")
        print(f"  Composite Score: {result['composite_score']:.4f}")
        
        # Show score components
        components = result['score_components']
        print(f"  Score Components:")
        print(f"    Profitability: {components['profitability']:.3f}")
        print(f"    Risk-Adjusted: {components['risk_adjusted']:.3f}")
        print(f"    Consistency: {components['consistency']:.3f}")
        print(f"    Penalties: {components['penalties']:.3f}")
        print(f"    Gaming Penalties: {components['gaming_penalties']:.3f}")
        
        if result['p_value'] is not None:
            significance = "significant" if result['significant'] else "not significant"
            print(f"  P-value: {result['p_value']:.4f} ({significance})")
        else:
            print(f"  P-value: N/A")
        
        # Show key metrics
        metrics = result['metrics']
        key_display_metrics = [
            'Return [%]_mean', 'Sharpe Ratio_mean', '# Trades_mean', 
            'Max. Drawdown [%]_mean', 'Win Rate [%]_mean'
        ]
        
        print(f"  Key Metrics (mean ± std):")
        for metric_mean in key_display_metrics:
            metric_std = metric_mean.replace('_mean', '_std')
            if metric_mean in metrics and metric_std in metrics:
                mean_val = metrics[metric_mean]
                std_val = metrics[metric_std]
                if mean_val is not None and std_val is not None:
                    metric_name = metric_mean.replace('_mean', '')
                    print(f"    {metric_name}: {mean_val:.4f} ± {std_val:.4f}")

# Convenience function that maintains backward compatibility
def mc_optimize_top10(strategy, param_space, data_dir=None, n_iterations=1000, 
                      optimization_metric='composite_score', minimize=False,
                      validation_permutations=100, n_threads=4, verbose=False,
                      output_file='top10_results.json', top_n=10, scoring_config=None):
    """
    Legacy wrapper for backward compatibility. Now uses composite scoring by default.
    
    Note: optimization_metric and minimize parameters are ignored when using composite scoring.
    Use mc_optimize_composite() directly for full control over the composite scoring system.
    """
    if verbose:
        print("Using composite scoring system (optimization_metric and minimize parameters ignored)")
    
    return mc_optimize_composite(
        strategy=strategy,
        param_space=param_space,
        data_dir=data_dir,
        n_iterations=n_iterations,
        scoring_config=scoring_config,
        validation_permutations=validation_permutations,
        n_threads=n_threads,
        verbose=verbose,
        output_file=output_file,
        top_n=top_n
    )

# Example usage and preset configurations
def get_conservative_scoring_config():
    """Conservative scoring configuration emphasizing risk management"""
    return ScoringConfig(
        profitability_weight=0.8,
        risk_adjusted_weight=2.0,  # Heavy emphasis on risk-adjusted returns
        consistency_weight=1.5,
        penalty_weight=3.0,  # Heavy penalties for risky behavior
        min_trades_threshold=10,  # Require more trades
        min_trades_penalty=-15.0,
        max_volatility_threshold=1500.0,  # Lower volatility threshold
        high_volatility_penalty=-8.0,
        max_drawdown_threshold=-30.0,  # Stricter drawdown limit
        excessive_drawdown_penalty=-10.0
    )

def get_aggressive_scoring_config():
    """Aggressive scoring configuration emphasizing returns"""
    return ScoringConfig(
        profitability_weight=2.0,  # Heavy emphasis on profits
        risk_adjusted_weight=1.0,
        consistency_weight=0.8,
        penalty_weight=1.0,  # Lighter penalties
        min_trades_threshold=3,  # Allow fewer trades
        min_trades_penalty=-5.0,
        max_volatility_threshold=3000.0,  # Higher volatility tolerance
        high_volatility_penalty=-2.0,
        max_drawdown_threshold=-60.0,  # More drawdown tolerance
        excessive_drawdown_penalty=-3.0
    )

def get_balanced_scoring_config():
    """Balanced scoring configuration (default)"""
    return ScoringConfig()  # Uses default values

# Example usage:
"""
# Basic usage with default composite scoring
results = mc_optimize_composite(
    strategy=MyStrategy,
    param_space={
        'rsi_period': lambda: random.randint(10, 30),
        'macd_fast': lambda: random.randint(8, 15),
        'macd_slow': lambda: random.randint(20, 30),
    },
    n_iterations=1000,
    validation_permutations=100,
    n_threads=20,
    output_file='composite_results.json',
    verbose=True
)

# Using conservative configuration
conservative_config = get_conservative_scoring_config()
results = mc_optimize_composite(
    strategy=MyStrategy,
    param_space=param_space,
    scoring_config=conservative_config,
    n_iterations=1000,
    validation_permutations=100,
    n_threads=20,
    output_file='conservative_results.json',
    verbose=True
)

# Custom scoring configuration
custom_config = ScoringConfig(
    profitability_metrics=['Return [%]', 'CAGR [%]'],
    risk_adjusted_metrics=['Sharpe Ratio', 'Sortino Ratio'],
    consistency_metrics=['Win Rate [%]', '# Trades'],
    penalty_metrics=['Max. Drawdown [%]', 'Volatility (Ann.) [%]'],
    min_trades_threshold=8,
    profitability_weight=1.5,
    risk_adjusted_weight=2.0,
    consistency_weight=1.0,
    penalty_weight=2.5
)

results = mc_optimize_composite(
    strategy=MyStrategy,
    param_space=param_space,
    scoring_config=custom_config,
    n_iterations=1000,
    validation_permutations=100,
    n_threads=20,
    output_file='custom_results.json',
    verbose=True
)

# Print results summary
print_composite_summary(results)
"""