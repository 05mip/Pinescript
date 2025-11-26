import pandas as pd
from datetime import datetime

# --- Helper to parse your lists ---
def parse_events(event_list):
    data = []
    for line in event_list.strip().split("\n"):
        date_str, price_str = line.split(" - Price: $")
        dt = datetime.strptime(date_str.strip(), "%Y-%m-%d %H:%M")
        price = float(price_str.strip())
        data.append({"time": dt, "price": price})
    return pd.DataFrame(data)

# --- Paste your raw text lists ---
buy_text = """
2025-06-21 01:30 - Price: $2.05
2025-06-23 00:00 - Price: $1.84
2025-06-25 00:00 - Price: $2.05
2025-06-25 10:30 - Price: $2.05
2025-06-27 09:00 - Price: $1.96
2025-06-29 18:00 - Price: $2.14
2025-07-02 07:30 - Price: $2.07
2025-07-05 07:30 - Price: $2.06
2025-07-08 12:00 - Price: $2.25
2025-07-10 18:00 - Price: $2.61
2025-07-13 07:30 - Price: $2.75
2025-07-15 12:00 - Price: $2.91
2025-07-18 00:00 - Price: $2.94
2025-07-18 13:30 - Price: $2.89
2025-07-19 16:30 - Price: $2.74
2025-07-23 06:00 - Price: $3.35
2025-07-24 12:00 - Price: $3.03
2025-07-25 03:00 - Price: $2.88
2025-07-25 21:00 - Price: $3.00
2025-07-27 15:00 - Price: $3.11
2025-07-31 01:30 - Price: $3.00
2025-08-02 03:00 - Price: $2.62
2025-08-03 03:00 - Price: $2.53
2025-08-06 13:30 - Price: $2.64
2025-08-09 06:00 - Price: $3.00
2025-08-12 16:30 - Price: $3.28"""
sell_text = """
2025-06-21 13:30 - Price: $2.02
2025-06-24 19:30 - Price: $2.06
2025-06-25 03:00 - Price: $2.06
2025-06-25 15:00 - Price: $2.02
2025-06-29 15:00 - Price: $2.12
2025-06-30 04:30 - Price: $2.13
2025-07-03 15:00 - Price: $2.14
2025-07-06 10:30 - Price: $2.10
2025-07-10 03:00 - Price: $2.45
2025-07-11 19:30 - Price: $2.80
2025-07-14 13:30 - Price: $2.83
2025-07-17 03:00 - Price: $2.90
2025-07-18 07:30 - Price: $2.94
2025-07-18 18:00 - Price: $2.82
2025-07-20 21:00 - Price: $2.90
2025-07-23 09:00 - Price: $3.24
2025-07-24 22:30 - Price: $3.03
2025-07-25 06:00 - Price: $2.92
2025-07-26 22:30 - Price: $3.02
2025-07-28 12:00 - Price: $3.28
2025-07-31 12:00 - Price: $2.91
2025-08-02 10:30 - Price: $2.58
2025-08-05 03:00 - Price: $2.71
2025-08-08 15:00 - Price: $2.83
2025-08-10 22:30 - Price: $3.44
2025-08-13 04:30 - Price: $3.48
"""
tp_text = """
2025-06-21 07:30 - Price: $2.10
2025-06-23 04:30 - Price: $1.87
2025-06-23 16:30 - Price: $1.90
2025-06-23 18:00 - Price: $1.95
2025-06-23 19:30 - Price: $1.99
2025-06-23 21:00 - Price: $2.08
2025-06-23 22:30 - Price: $2.06
2025-06-24 00:00 - Price: $2.06
2025-06-24 04:30 - Price: $2.09
2025-06-24 06:00 - Price: $2.09
2025-06-27 15:00 - Price: $1.99
2025-06-28 06:00 - Price: $2.04
2025-06-28 09:00 - Price: $2.06
2025-06-28 15:00 - Price: $2.09
2025-06-28 16:30 - Price: $2.13
2025-06-29 09:00 - Price: $2.16
2025-06-29 21:00 - Price: $2.20
2025-07-02 15:00 - Price: $2.12
2025-07-02 18:00 - Price: $2.14
2025-07-03 00:00 - Price: $2.17
2025-07-03 04:30 - Price: $2.23
2025-07-03 06:00 - Price: $2.23
2025-07-06 00:00 - Price: $2.10
2025-07-06 01:30 - Price: $2.13
2025-07-06 06:00 - Price: $2.18
2025-07-08 21:00 - Price: $2.28
2025-07-09 03:00 - Price: $2.33
2025-07-09 04:30 - Price: $2.42
2025-07-09 06:00 - Price: $2.40
2025-07-09 10:30 - Price: $2.41
2025-07-09 12:00 - Price: $2.49
2025-07-09 18:00 - Price: $2.49
2025-07-10 21:00 - Price: $2.68
2025-07-10 22:30 - Price: $2.71
2025-07-11 00:00 - Price: $2.73
2025-07-11 09:00 - Price: $2.85
2025-07-11 10:30 - Price: $2.81
2025-07-11 13:30 - Price: $2.90
2025-07-13 12:00 - Price: $2.83
2025-07-14 01:30 - Price: $2.84
2025-07-14 03:00 - Price: $2.91
2025-07-16 09:00 - Price: $2.96
2025-07-16 16:30 - Price: $3.01
2025-07-18 04:30 - Price: $3.06
2025-07-19 19:30 - Price: $2.80
2025-07-20 00:00 - Price: $2.83
2025-07-20 07:30 - Price: $2.88
2025-07-20 12:00 - Price: $2.96
2025-07-20 13:30 - Price: $2.96
2025-07-20 15:00 - Price: $3.01
2025-07-24 13:30 - Price: $3.16
2025-07-24 15:00 - Price: $3.13
2025-07-24 16:30 - Price: $3.18
2025-07-26 04:30 - Price: $3.06
2025-07-26 09:00 - Price: $3.13
2025-07-27 21:00 - Price: $3.17
2025-07-28 01:30 - Price: $3.28
2025-07-28 03:00 - Price: $3.27
2025-07-28 04:30 - Price: $3.36
2025-07-28 06:00 - Price: $3.39
2025-07-31 07:30 - Price: $3.05
2025-08-03 18:00 - Price: $2.58
2025-08-04 00:00 - Price: $2.63
2025-08-04 13:30 - Price: $2.65
2025-08-04 15:00 - Price: $2.72
2025-08-04 21:00 - Price: $2.76
2025-08-04 22:30 - Price: $2.79
2025-08-06 18:00 - Price: $2.68
2025-08-07 09:00 - Price: $2.75
2025-08-07 10:30 - Price: $2.77
2025-08-07 21:00 - Price: $2.83
2025-08-08 06:00 - Price: $2.85
2025-08-08 10:30 - Price: $2.93
2025-08-09 09:00 - Price: $3.07
2025-08-10 00:00 - Price: $3.17
2025-08-10 01:30 - Price: $3.47
2025-08-10 03:00 - Price: $3.49
2025-08-10 04:30 - Price: $3.42
2025-08-10 06:00 - Price: $3.40
2025-08-10 07:30 - Price: $3.35
2025-08-10 09:00 - Price: $3.41
2025-08-10 16:30 - Price: $3.42
2025-08-12 21:00 - Price: $3.33
2025-08-13 00:00 - Price: $3.48
2025-08-13 01:30 - Price: $3.53
2025-08-13 03:00 - Price: $3.63
"""

# --- Parse into DataFrames ---
buy_df = parse_events(buy_text)
buy_df["type"] = "buy"

sell_df = parse_events(sell_text)
sell_df["type"] = "sell"

tp_df = parse_events(tp_text)
tp_df["type"] = "tp"

# Merge all events & sort by time
events = pd.concat([buy_df, sell_df, tp_df]).sort_values("time").reset_index(drop=True)

# --- Simulation ---
cash = 1000.0
position = 0.0
trade_entries = []
trade_results = []
equity_curve = []
current_trade_entry_price = None

for _, row in events.iterrows():
    price = row["price"]

    if row["type"] == "buy":
        # Open new trade if flat
        if position == 0:
            current_trade_entry_price = price
        units = cash / price
        position += units
        cash = 0.0

    elif row["type"] == "tp" and position > 0:
        # Partial exit (30%)
        sell_units = position * 0.30
        cash += sell_units * price
        position -= sell_units

    elif row["type"] == "sell" and position > 0:
        # Full exit
        cash += position * price
        # Track trade P/L
        if current_trade_entry_price is not None:
            pct_change = (price - current_trade_entry_price) / current_trade_entry_price * 100
            trade_results.append(pct_change)
        position = 0.0
        current_trade_entry_price = None

    equity = cash + position * price
    equity_curve.append({"time": row["time"], "equity": equity})

# --- Stats ---
equity_df = pd.DataFrame(equity_curve)
final_equity = equity_df["equity"].iloc[-1]
total_return = (final_equity - 1000) / 1000 * 100
win_rate = sum(1 for r in trade_results if r > 0) / len(trade_results) * 100 if trade_results else 0
avg_pct_per_trade = sum(trade_results) / len(trade_results) if trade_results else 0

print(f"Final Equity: ${final_equity:,.2f}")
print(f"Total Return: {total_return:.2f}%")
print(f"Max Equity: ${equity_df['equity'].max():,.2f}")
print(f"Min Equity: ${equity_df['equity'].min():,.2f}")
print(f"Number of Trades: {len(trade_results)}")
print(f"Win Rate: {win_rate:.2f}%")
print(f"Average % per Trade: {avg_pct_per_trade:.2f}%")
