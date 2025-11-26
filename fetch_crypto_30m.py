import yfinance as yf
import pandas as pd

# List of top 10 cryptocurrencies (as of early 2024)
symbols = [
    'BTC-USD', 'ETH-USD', 'BNB-USD', 'SOL-USD', 'XRP-USD',
    'ADA-USD', 'DOGE-USD', 'AVAX-USD', 'DOT-USD'
]

period = '59d'
interval = '60m'

for symbol in symbols:
    print(f"Fetching {symbol}...")
    try:
        data = yf.download(
            tickers=symbol,
            period=period,
            interval=interval,
            progress=False
        )
        if data is not None and not data.empty:
            filename = f"{symbol}_{interval}.csv"
            data.to_csv(filename)
            print(f"Saved {filename} ({len(data)} rows)")
        else:
            print(f"No data for {symbol}")
    except Exception as e:
        print(f"Error fetching {symbol}: {e}")

print("Done.") 