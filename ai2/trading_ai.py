import yfinance as yf
import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import pytz

def fetch_data(symbol, start_date, end_date, interval='15m', max_retries=3, delay=2):
    if isinstance(start_date, datetime):
        start_date = start_date.astimezone(pytz.UTC)
    if isinstance(end_date, datetime):
        end_date = end_date.astimezone(pytz.UTC)
    
    for attempt in range(max_retries):
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(start=start_date, end=end_date, interval=interval)
            if not data.empty:
                # Convert index from UTC to PST
                data.index = data.index.tz_convert('America/Los_Angeles')
                return data
            print(f"Attempt {attempt + 1}: No data received, retrying...")
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {str(e)}")
            if attempt < max_retries - 1:
                print(f"Waiting {delay} seconds before retrying...")
                time.sleep(delay)
            else:
                raise Exception(f"Failed to fetch data after {max_retries} attempts")
    return None

# Example usage:
symbols = ['XRP-USD', 'BTC-USD', 'SOL-USD', 'ETH-USD']
end = datetime.now()
start = end - timedelta(days=59)
data = {sym: fetch_data(sym, start, end) for sym in symbols}

# --- Genome definition ---
GENOME_PARAMS = {
    'length_rsi': (7, 21),         # int
    'length_stoch': (7, 21),       # int
    'smooth_k': (1, 5),            # int
    'smooth_d': (1, 5),            # int
    'ha_smooth_period': (1, 5),    # int
    'k_buy_threshold': (-40, 40),  # float
    'k_sell_threshold': (-40, 40), # float
}

def random_genome():
    return {
        k: random.randint(*v) if isinstance(v[0], int) else random.uniform(*v)
        for k, v in GENOME_PARAMS.items()
    }

def mutate_genome(genome, mutation_rate=0.2):
    new_genome = genome.copy()
    for k, v in GENOME_PARAMS.items():
        if random.random() < mutation_rate:
            if isinstance(v[0], int):
                new_genome[k] = random.randint(*v)
            else:
                new_genome[k] = random.uniform(*v)
    return new_genome

def crossover_genome(g1, g2):
    return {k: random.choice([g1[k], g2[k]]) for k in GENOME_PARAMS}

# --- Trading simulation ---
def calc_rsi(close, period):
    close = np.array(close, dtype=float)
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    gain_ema = pd.Series(gain).ewm(span=period, adjust=False).mean()
    loss_ema = pd.Series(loss).ewm(span=period, adjust=False).mean()
    rs = gain_ema / loss_ema
    rsi_values = 100 - (100 / (1 + rs))
    rsi = np.full(len(close), np.nan)
    rsi[1:] = rsi_values
    rsi[:period] = np.nan
    return rsi

def calc_stoch_rsi(rsi, length_stoch):
    stoch_rsi = np.zeros_like(rsi)
    for i in range(len(rsi)):
        if i < length_stoch or np.isnan(rsi[i]):
            stoch_rsi[i] = 0
        else:
            rsi_min = np.nanmin(rsi[i-length_stoch+1:i+1])
            rsi_max = np.nanmax(rsi[i-length_stoch+1:i+1])
            stoch_rsi[i] = (rsi[i] - rsi_min) / (rsi_max - rsi_min) if rsi_max != rsi_min else 0
    return stoch_rsi

def smooth(values, period):
    smoothed = np.zeros_like(values)
    for i in range(len(values)):
        start_idx = max(0, i - period + 1)
        smoothed[i] = np.mean(values[start_idx:i+1])
    return smoothed

def simulate_trading(df, genome):
    # Calculate Heikin Ashi
    ha_close = (df['Open'] + df['High'] + df['Low'] + df['Close']) / 4
    ha_open = np.zeros_like(ha_close)
    ha_open[0] = df['Open'].iloc[0]
    for i in range(1, len(ha_open)):
        ha_open[i] = (ha_open[i-1] + ha_close[i-1]) / 2
    ha_high = np.maximum.reduce([df['High'], ha_open, ha_close])
    ha_low = np.minimum.reduce([df['Low'], ha_open, ha_close])
    # Smooth Heikin Ashi
    ha_open = smooth(ha_open, genome['ha_smooth_period'])
    ha_close = smooth(ha_close, genome['ha_smooth_period'])
    # Stoch RSI
    rsi = calc_rsi(df['Close'], genome['length_rsi'])
    stoch_rsi = calc_stoch_rsi(rsi, genome['length_stoch'])
    k = smooth(stoch_rsi, genome['smooth_k'])
    d = smooth(k, genome['smooth_d'])
    k_scaled = k * 80 - 40
    d_scaled = d * 80 - 40
    # Trading logic
    position = 0  # 0 = out, 1 = in
    entry_price = 0
    profit = 0
    for i in range(2, len(df)):
        ha_green = ha_close[i] > ha_open[i]
        ha_red = ha_close[i] < ha_open[i]
        k_rising = k_scaled[i] > k_scaled[i-1]
        k_falling = k_scaled[i] < k_scaled[i-1]
        # Buy
        if not position and k_scaled[i] > genome['k_buy_threshold'] and ha_green and k_rising:
            position = 1
            entry_price = df['Close'].iloc[i]
        # Sell
        elif position and (k_scaled[i] < genome['k_sell_threshold'] and ha_red and k_falling):
            profit += df['Close'].iloc[i] - entry_price
            position = 0
    # If still in position, close at last price
    if position:
        profit += df['Close'].iloc[-1] - entry_price
    return profit

# --- Fitness function ---
def fitness(genome, data):
    # Average profit across all coins
    profits = []
    for sym, df in data.items():
        p = simulate_trading(df, genome)
        profits.append(p)
    return np.mean(profits)

# --- Genetic Algorithm ---
def genetic_optimize(data, population_size=30, generations=20, elite_frac=0.2, mutation_rate=0.2):
    population = [random_genome() for _ in range(population_size)]
    n_elite = int(population_size * elite_frac)
    for gen in range(generations):
        scored = [(fitness(g, data), g) for g in population]
        scored.sort(reverse=True, key=lambda x: x[0])
        print(f"Gen {gen}: Best profit = {scored[0][0]:.2f}")
        # Elitism
        next_gen = [g for _, g in scored[:n_elite]]
        # Crossover + mutation
        while len(next_gen) < population_size:
            parents = random.sample(scored[:n_elite], 2)
            child = crossover_genome(parents[0][1], parents[1][1])
            child = mutate_genome(child, mutation_rate)
            next_gen.append(child)
        population = next_gen
    # Return best genome
    best = max(population, key=lambda g: fitness(g, data))
    return best

# --- Usage ---
# data = {sym: DataFrame, ...}  # Already loaded
best_genome = genetic_optimize(data)
print("Best parameters:", best_genome)
