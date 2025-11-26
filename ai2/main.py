import pandas as pd
import numpy as np
from deap import base, creator, tools, algorithms
import random
import glob
import os
from scipy.signal import butter, filtfilt
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


# 1. Data Loading and Preprocessing
# ---------------------------------
# Load all *_30m.csv files from the data folder, skip first 3 rows, use correct columns
DATA_DIR = './data'
all_files = glob.glob(os.path.join(DATA_DIR, '*_30m.csv'))

COLS = ['Datetime', 'Close', 'High', 'Low', 'Open', 'Volume']

def load_and_concat(files):
    dfs = []
    for f in files:
        df = pd.read_csv(f, skiprows=3, names=COLS)
        df['symbol'] = os.path.basename(f).split('-')[0]
        df = df.dropna(subset=['Close', 'High', 'Low', 'Open', 'Volume'])
        if len(df) == 0:
            print(f"  WARNING: No data loaded from {f} after dropna!")
        else:
            dfs.append(df)
    if not dfs:
        raise ValueError('No dataframes loaded. Check file paths and structure.')
    return pd.concat(dfs, ignore_index=True)

print("Loading and preprocessing data...")
df = load_and_concat(all_files)
print(f"Loaded {len(df)} rows from {len(all_files)} files.")

# Feature Engineering: HARSIStrategy indicators
# --------------------------------------------
def add_harsi_indicators(df, window=100, rsi_period=14, stoch_period=14, smooth_k=3, smooth_d=3, ha_smooth_period=1, butterworth=True, filter_order=1, filter_cutoff=0.1):
    df = df.copy()
    # Heikin Ashi calculation
    ha_close = (df['Open'] + df['High'] + df['Low'] + df['Close']) / 4
    ha_open = df['Open'].copy()
    for i in range(1, len(df)):
        ha_open.iloc[i] = (ha_open.iloc[i-1] + ha_close.iloc[i-1]) / 2
    ha_high = pd.concat([
        df['High'], ha_open, ha_close
    ], axis=1).max(axis=1)
    ha_low = pd.concat([
        df['Low'], ha_open, ha_close
    ], axis=1).min(axis=1)
    # Smoothing
    ha_open_s = ha_open.rolling(ha_smooth_period, min_periods=1).mean()
    ha_close_s = ha_close.rolling(ha_smooth_period, min_periods=1).mean()
    ha_high_s = ha_high.rolling(ha_smooth_period, min_periods=1).mean()
    ha_low_s = ha_low.rolling(ha_smooth_period, min_periods=1).mean()
    # Scaling (min-max over window)
    min_val = ha_low_s.rolling(window, min_periods=1).min()
    max_val = ha_high_s.rolling(window, min_periods=1).max()
    scale = 1.0 / (max_val - min_val).replace(0, np.nan)
    ha_open_scaled = (ha_open_s - min_val) * scale
    ha_close_scaled = (ha_close_s - min_val) * scale
    ha_high_scaled = (ha_high_s - min_val) * scale
    ha_low_scaled = (ha_low_s - min_val) * scale
    # RSI
    close = df['Close']
    delta = close.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(span=rsi_period, adjust=False).mean()
    avg_loss = loss.ewm(span=rsi_period, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    # Stochastic RSI
    stoch_rsi = pd.Series(np.zeros(len(rsi)), index=rsi.index)
    for i in range(len(rsi)):
        if i < stoch_period:
            stoch_rsi.iloc[i] = 0
        else:
            rsi_min = rsi.iloc[i-stoch_period+1:i+1].min()
            rsi_max = rsi.iloc[i-stoch_period+1:i+1].max()
            stoch_rsi.iloc[i] = (rsi.iloc[i] - rsi_min) / (rsi_max - rsi_min) if rsi_max != rsi_min else 0
    # Butterworth filter (optional)
    if butterworth:
        def butterworth_filter(signal, order=filter_order, cutoff=filter_cutoff):
            signal = signal.values
            if len(signal) < 3 * order:
                return signal
            valid_mask = ~np.isnan(signal)
            if not np.any(valid_mask):
                return signal
            valid_signal = signal[valid_mask]
            if len(valid_signal) < 3 * order:
                return signal
            b, a = butter(order, cutoff, btype='low')
            try:
                filtered_valid = filtfilt(b, a, valid_signal)
                filtered_signal = signal.copy()
                filtered_signal[valid_mask] = filtered_valid
                return filtered_signal
            except:
                return signal
        stoch_rsi = pd.Series(butterworth_filter(stoch_rsi), index=stoch_rsi.index)
    # K and D lines
    k = stoch_rsi.rolling(smooth_k, min_periods=1).mean()
    d = k.rolling(smooth_d, min_periods=1).mean()
    # Rescale K and D from [0, 1] to [-40, 40]
    k_scaled = k * 80 - 40
    d_scaled = d * 80 - 40
    # Add all features to df
    df['ha_open_scaled'] = ha_open_scaled
    df['ha_close_scaled'] = ha_close_scaled
    df['ha_high_scaled'] = ha_high_scaled
    df['ha_low_scaled'] = ha_low_scaled
    df['rsi'] = rsi
    df['stoch_rsi'] = stoch_rsi
    df['k'] = k
    df['d'] = d
    df['k_scaled'] = k_scaled
    df['d_scaled'] = d_scaled
    return df

df = add_harsi_indicators(df)
df = df.dropna().reset_index(drop=True)
print(f"After feature engineering and dropna: {len(df)} rows remain.")

# Select features and normalize
FEATURES = [
    'Open', 'High', 'Low', 'Close', 'Volume',
    'ha_open_scaled', 'ha_close_scaled', 'ha_high_scaled', 'ha_low_scaled',
    'rsi', 'stoch_rsi', 'k', 'd', 'k_scaled', 'd_scaled'
]
scaler = StandardScaler()
df[FEATURES] = scaler.fit_transform(df[FEATURES])

# Train/test split
TEST_FRAC = 0.2
split_idx = int(len(df) * (1 - TEST_FRAC))
df_train = df.iloc[:split_idx].reset_index(drop=True)
df_test = df.iloc[split_idx:].reset_index(drop=True)

# Split by symbol and train/test
symbol_dfs = {sym: df[df['symbol'] == sym].reset_index(drop=True) for sym in df['symbol'].unique()}
symbol_train = {sym: sdf.iloc[:int(len(sdf)*(1-TEST_FRAC))].reset_index(drop=True) for sym, sdf in symbol_dfs.items()}
symbol_test = {sym: sdf.iloc[int(len(sdf)*(1-TEST_FRAC)):].reset_index(drop=True) for sym, sdf in symbol_dfs.items()}

# Store original (unnormalized) DataFrames for plotting
original_symbol_dfs = {sym: sdf.copy() for sym, sdf in symbol_dfs.items()}

# 2. Neural Network Definition
# ----------------------------
INPUT_SIZE = len(FEATURES)
HIDDEN_SIZE = 8
OUTPUT_SIZE = 3  # buy, sell, hold

def nn_forward(weights, x):
    w1_end = INPUT_SIZE * HIDDEN_SIZE
    w2_end = w1_end + HIDDEN_SIZE * OUTPUT_SIZE
    b1_end = w2_end + HIDDEN_SIZE
    w1 = np.array(weights[:w1_end]).reshape(INPUT_SIZE, HIDDEN_SIZE)
    w2 = np.array(weights[w1_end:w2_end]).reshape(HIDDEN_SIZE, OUTPUT_SIZE)
    b1 = np.array(weights[w2_end:b1_end])
    b2 = np.array(weights[b1_end:])
    h = np.tanh(np.dot(x, w1) + b1)
    out = np.dot(h, w2) + b2
    return out

# 3. Trading Environment (updated)
# ----------------------
def simulate_trading(weights, df, features=FEATURES, initial_balance=1000, commission=0.001, penalize_trades=True, reward_hold=True, debug_avg_gain=False):
    balance = initial_balance
    position = 0
    entry_price = 0
    trades = []
    hold_durations = []
    current_hold = 0
    equity_curve = [balance]
    for i, row in df.iterrows():
        x = row[features].values.astype(float)
        logits = nn_forward(weights, x)
        action = np.argmax(logits)
        price = row['Close']
        if action == 0 and position == 0:  # Buy
            position = 1
            entry_price = price
            balance -= price * commission  # commission on buy
            current_hold = 0
        elif action == 1 and position == 1:  # Sell
            profit = price - entry_price
            pct_gain = profit / entry_price
            balance += profit
            balance -= price * commission  # commission on sell
            trades.append(pct_gain)
            hold_durations.append(current_hold)
            position = 0
            current_hold = 0
        elif position == 1:
            current_hold += 1
        equity_curve.append(balance + (price - entry_price if position == 1 else 0))
    # Close any open position at the end
    if position == 1:
        profit = df.iloc[-1]['Close'] - entry_price
        pct_gain = profit / entry_price
        balance += profit
        balance -= df.iloc[-1]['Close'] * commission
        trades.append(pct_gain)
        hold_durations.append(current_hold)
    n_trades = len(trades)
    avg_hold = np.mean(hold_durations) if hold_durations else 0
    if trades:
        winrate = sum([1 for t in trades if t > 0]) / n_trades
        avg_gain = np.mean(trades)
        max_gain = np.max(trades)
    else:
        winrate = 0
        avg_gain = 0
        max_gain = 0
    if debug_avg_gain and abs(avg_gain) < 1e-4 and n_trades > 0:
        print(f"[DEBUG] Trades list (avg_gain={avg_gain}): {trades}")
    total_return = (balance - initial_balance) / initial_balance
    penalty = 0
    if penalize_trades and n_trades > 0:
        penalty -= 0.01 * n_trades
    bonus = 0
    if reward_hold and max_gain > 0.05:
        bonus += max_gain * 2
    fitness = total_return + 0.5 * winrate + 0.5 * avg_gain + penalty + bonus
    return (fitness, winrate, avg_gain, n_trades, avg_hold, equity_curve)

# Number of fitness coefficients to co-evolve
N_COEFFS = 4  # return, winrate, avg_gain, n_trades
COEFF_NAMES = ['w_return', 'w_winrate', 'w_avg_gain', 'w_ntrades']
COEFF_BOUNDS = (0, 10)

def eval_individual(individual, train=True):
    nn_weights = individual[:N_WEIGHTS]
    coeffs = individual[N_WEIGHTS:N_WEIGHTS+N_COEFFS]
    w_return, w_winrate, w_avg_gain, w_ntrades = coeffs
    dfs = symbol_train if train else symbol_test
    fitnesses = []
    winrates = []
    avg_gains = []
    for sym, sdf in dfs.items():
        if len(sdf) < 10:
            continue
        total_return, winrate, avg_gain, n_trades, avg_hold, equity_curve = simulate_trading(nn_weights, sdf)
        fit = (
            w_return * total_return +
            w_winrate * winrate +
            w_avg_gain * avg_gain +
            w_ntrades * (-n_trades/100)
        )
        fitnesses.append(fit)
        winrates.append(winrate)
        avg_gains.append(avg_gain)
    if not fitnesses:
        return (0.0, 0.0, 0.0)
    return (np.mean(fitnesses), np.mean(winrates), np.mean(avg_gains))

def diagnostics(individual, dfs, label=""):
    nn_weights = individual[:N_WEIGHTS]
    coeffs = individual[N_WEIGHTS:N_WEIGHTS+N_COEFFS]
    w_return, w_winrate, w_avg_gain, w_ntrades = coeffs
    print(f"{label} Diagnostics (averaged across assets):")
    fitnesses = []
    winrates = []
    avg_gains = []
    n_trades_list = []
    avg_holds = []
    total_returns = []
    for sym, sdf in dfs.items():
        if len(sdf) < 10:
            continue
        fitness, winrate, avg_gain, n_trades, avg_hold, equity_curve = simulate_trading(nn_weights, sdf)
        fitnesses.append(fitness)
        winrates.append(winrate)
        avg_gains.append(avg_gain)
        n_trades_list.append(n_trades)
        avg_holds.append(avg_hold)
        total_returns.append((equity_curve[-1] - equity_curve[0]) / equity_curve[0])
    print(f"  Fitness: {np.mean(fitnesses):.4f}")
    print(f"  Winrate: {np.mean(winrates)*100:.2f}%")
    print(f"  Avg Gain per Trade: {np.mean(avg_gains)*100:.2f}%")
    print(f"  Number of Trades: {np.mean(n_trades_list):.2f}")
    print(f"  Avg Hold Duration: {np.mean(avg_holds):.2f} bars")
    print(f"  Total Return: {np.mean(total_returns)*100:.2f}%")
    print(f"  Coefficients: return={w_return:.2f}, winrate={w_winrate:.2f}, avg_gain={w_avg_gain:.2f}, n_trades={w_ntrades:.2f}")
    # Plot equity curve and trades for the first symbol only
    first_sym = list(dfs.keys())[0]
    print(f"  Showing trade plot for: {first_sym}")
    plot_trades(individual, dfs[first_sym], label=f"{label} {first_sym}")

def plot_trades(individual, df_eval, label=""):
    nn_weights = individual[:N_WEIGHTS]
    # Simulate trading and record trade actions
    position = 0
    entry_idx = None
    entries = []
    exits = []
    # Use original price data if available
    symbol = df_eval['symbol'].iloc[0] if 'symbol' in df_eval.columns else None
    if symbol and symbol in original_symbol_dfs:
        orig_prices = original_symbol_dfs[symbol].loc[df_eval.index, 'Close'].values
    else:
        orig_prices = df_eval['Close'].values
    features = df_eval[FEATURES].values.astype(float)
    for i, x in enumerate(features):
        logits = nn_forward(nn_weights, x)
        action = np.argmax(logits)
        price = orig_prices[i]
        if action == 0 and position == 0:
            position = 1
            entry_idx = i
            entries.append((i, price))
        elif action == 1 and position == 1:
            position = 0
            exits.append((i, price))
    if position == 1 and entry_idx is not None:
        exits.append((len(orig_prices)-1, orig_prices[-1]))
    plt.figure(figsize=(12,5))
    plt.plot(orig_prices, label='Close Price')
    if entries:
        plt.scatter(*zip(*entries), marker='^', color='g', label='Buy', zorder=5)
    if exits:
        plt.scatter(*zip(*exits), marker='v', color='r', label='Sell', zorder=5)
    plt.title(f"Trade Entries/Exits {label}")
    plt.xlabel("Bar")
    plt.ylabel("Price")
    plt.legend()
    plt.tight_layout()
    plt.show()

# 4. DEAP Evolutionary Algorithm
# ------------------------------
creator.create("FitnessMax", base.Fitness, weights=(1.0, 0.5, 0.5))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
N_WEIGHTS = INPUT_SIZE*HIDDEN_SIZE + HIDDEN_SIZE*OUTPUT_SIZE + HIDDEN_SIZE + OUTPUT_SIZE
TOTAL_GENES = N_WEIGHTS + N_COEFFS

def make_individual():
    # NN weights: uniform(-1, 1), Coeffs: uniform(0, 10)
    genes = [random.uniform(-1, 1) for _ in range(N_WEIGHTS)]
    genes += [random.uniform(*COEFF_BOUNDS) for _ in range(N_COEFFS)]
    return creator.Individual(genes)

toolbox.register("individual", make_individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", lambda ind: eval_individual(ind, train=True))
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1.0, indpb=0.3)
toolbox.register("select", tools.selNSGA2)

def main():
    print("Starting neuroevolution...")
    pop = toolbox.population(n=100)
    NGEN = 30
    for gen in range(NGEN):
        print(f"\n--- Generation {gen+1}/{NGEN} ---")
        offspring = algorithms.varAnd(pop, toolbox, cxpb=0.7, mutpb=0.5)
        fits = list(map(toolbox.evaluate, offspring))
        print(f"[DEBUG] First 5 fitness values: {fits[:5]}")
        for ind, fit in zip(offspring, fits):
            ind.fitness.values = fit
        for ind in offspring + pop:
            if not hasattr(ind.fitness, 'values') or len(ind.fitness.values) != 3:
                ind.fitness.values = (0.0, 0.0, 0.0)
        pop = toolbox.select(offspring, k=len(pop))
        best = tools.selBest(pop, k=1)[0]
        print(f"Best so far: Fitness={best.fitness.values[0]:.4f}, Winrate={best.fitness.values[1]:.2f}, AvgGain={best.fitness.values[2]:.4f}")
        # Plot trades for best individual on first symbol in train set
        first_sym = list(symbol_train.keys())[0]
        plot_trades(best, symbol_train[first_sym], label=f"Gen {gen+1} (Train) {first_sym}")
    best = tools.selBest(pop, k=1)[0]
    print("\nEvolution complete!")
    print("Best Individual on TRAIN:")
    diagnostics(best, symbol_train, label="Train")
    print("Best Individual on TEST:")
    diagnostics(best, symbol_test, label="Test")

if __name__ == "__main__":
    main()
