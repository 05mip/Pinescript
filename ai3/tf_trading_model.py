import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.utils import to_categorical
import pytz

# --- Copy fetch_data from ai2/trading_ai.py ---
def fetch_data(symbol, start_date, end_date, interval='15m', max_retries=3, delay=2):
    import yfinance as yf
    import time
    if isinstance(start_date, datetime):
        start_date = start_date.astimezone(pytz.UTC)
    if isinstance(end_date, datetime):
        end_date = end_date.astimezone(pytz.UTC)
    for attempt in range(max_retries):
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(start=start_date, end=end_date, interval=interval)
            if not data.empty:
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

# --- Feature engineering ---
def add_indicators(df):
    df = df.copy()
    # Heikin Ashi
    ha_close = (df['Open'] + df['High'] + df['Low'] + df['Close']) / 4
    ha_open = np.zeros_like(ha_close)
    ha_open[0] = df['Open'].iloc[0]
    for i in range(1, len(ha_open)):
        ha_open[i] = (ha_open[i-1] + ha_close[i-1]) / 2
    df['ha_close'] = ha_close
    df['ha_open'] = ha_open
    # RSI
    def calc_rsi(close, period=14):
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
    df['rsi'] = calc_rsi(df['Close'], 14)
    # Stoch RSI
    def calc_stoch_rsi(rsi, length_stoch=14):
        stoch_rsi = np.zeros_like(rsi)
        for i in range(len(rsi)):
            if i < length_stoch or np.isnan(rsi[i]):
                stoch_rsi[i] = 0
            else:
                rsi_min = np.nanmin(rsi[i-length_stoch+1:i+1])
                rsi_max = np.nanmax(rsi[i-length_stoch+1:i+1])
                stoch_rsi[i] = (rsi[i] - rsi_min) / (rsi_max - rsi_min) if rsi_max != rsi_min else 0
        return stoch_rsi
    df['stoch_rsi'] = calc_stoch_rsi(df['rsi'], 14)
    df = df.dropna()
    return df

# --- Label generation ---
def make_labels(df, horizon=5, threshold=0.002):
    df = df.copy()
    df['future_return'] = df['Close'].shift(-horizon) / df['Close'] - 1
    df['label'] = 0  # hold
    df.loc[df['future_return'] > threshold, 'label'] = 1  # buy
    df.loc[df['future_return'] < -threshold, 'label'] = 2  # sell
    df = df.dropna()
    return df

# --- Main script ---
if __name__ == '__main__':
    symbols = ['XRP-USD', 'BTC-USD', 'SOL-USD', 'ETH-USD']
    end = datetime.now()
    start = end - timedelta(days=59)
    dfs = []
    for sym in symbols:
        print(f"Fetching {sym}...")
        df = fetch_data(sym, start, end)
        if df is not None and not df.empty:
            df['symbol'] = sym
            dfs.append(df)
    if not dfs:
        print("No data fetched!")
        exit(1)
    data = pd.concat(dfs)
    data = add_indicators(data)
    data = make_labels(data)
    feature_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'ha_open', 'ha_close', 'rsi', 'stoch_rsi']
    X = data[feature_cols].values
    y = to_categorical(data['label'].values, num_classes=3)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # --- Model ---
    model = keras.Sequential([
        layers.Input(shape=(len(feature_cols),)),
        layers.Dense(64, activation='relu'),
        layers.Dense(32, activation='relu'),
        layers.Dense(3, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=10, batch_size=128, validation_split=0.1)
    loss, acc = model.evaluate(X_test, y_test)
    print(f"Test accuracy: {acc:.4f}")

    # --- Example: Predict and show confusion matrix ---
    y_pred = np.argmax(model.predict(X_test), axis=1)
    y_true = np.argmax(y_test, axis=1)
    from sklearn.metrics import classification_report, confusion_matrix
    print(classification_report(y_true, y_pred, target_names=['hold','buy','sell']))
    print(confusion_matrix(y_true, y_pred))

    # --- Save model and scaler ---
    model.save('ai3/tf_trading_model.keras')
    import joblib
    joblib.dump(scaler, 'ai3/tf_scaler.save') 