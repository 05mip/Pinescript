import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.firefox.service import Service
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.common.action_chains import ActionChains
import logging
import pytz
import os
import pickle
import requests
import base64
from scipy.signal import butter, filtfilt
class HARSIStrategyDirect:
    def __init__(self):
        # Strategy parameters
        self.bottom = -17.98
        self.middle_low = -13.76
        self.middle_high = 26.82
        self.top = 30.14
        self.length_rsi = 33
        self.length_stoch = 22
        self.smooth_k = 15
        self.smooth_d = 13
        self.max_ha_cross = 34
        self.window = 249
        self.ha_smooth_period = 1
        self.use_butterworth_filter = True
        self.filter_order = 3
        self.filter_cutoff = 0.21

        # State tracking
        self.last_action = None
        self.ha_open_history = []
        self.ha_close_history = []
        self.ha_high_history = []
        self.ha_low_history = []
        self.rsi_history = []
        self.stoch_rsi_history = []
        self.k_history = []
        self.d_history = []

    def calc_rsi(self, close_prices, period=14):
        """Calculate RSI using vectorized approach"""
        if len(close_prices) < period + 1:
            return np.full(len(close_prices), np.nan)

        close = np.array(close_prices, dtype=float)
        delta = np.diff(close)

        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)

        # Use pandas-style exponential weighted mean
        gain_ema = pd.Series(gain).ewm(span=period, adjust=False).mean()
        loss_ema = pd.Series(loss).ewm(span=period, adjust=False).mean()

        rs = gain_ema / loss_ema
        rsi_values = 100 - (100 / (1 + rs))

        # Pad with NaN for first value and return
        rsi = np.full(len(close), np.nan)
        rsi[1:] = rsi_values
        rsi[:period] = np.nan

        return rsi

    def calc_stoch_rsi(self, rsi, length_stoch=14):
        """Calculate Stochastic RSI"""
        if len(rsi) < length_stoch:
            return np.zeros(len(rsi))

        stoch_rsi = np.zeros_like(rsi)
        for i in range(len(rsi)):
            if i < length_stoch or np.isnan(rsi[i]):
                stoch_rsi[i] = 0
            else:
                rsi_window = rsi[i-length_stoch+1:i+1]
                rsi_window = rsi_window[~np.isnan(rsi_window)]
                if len(rsi_window) > 0:
                    rsi_min = np.min(rsi_window)
                    rsi_max = np.max(rsi_window)
                    stoch_rsi[i] = (rsi[i] - rsi_min) / (rsi_max - rsi_min) if rsi_max != rsi_min else 0
                else:
                    stoch_rsi[i] = 0
        return stoch_rsi

    def butterworth_filter(self, signal):
        """Apply Butterworth low-pass filter"""
        min_length = 3 * self.filter_order

        if len(signal) < min_length:
            return signal

        # Remove NaN values for filtering
        valid_mask = ~np.isnan(signal)
        if not np.any(valid_mask):
            return signal

        valid_signal = signal[valid_mask]
        if len(valid_signal) < min_length:
            return signal

        # Design filter
        b, a = butter(self.filter_order, self.filter_cutoff, btype='low')

        # Apply filter
        try:
            filtered_valid = filtfilt(b, a, valid_signal)

            # Reconstruct full array
            filtered_signal = signal.copy()
            filtered_signal[valid_mask] = filtered_valid

            return filtered_signal
        except:
            return signal

    def calc_heikin_ashi(self, open_prices, high_prices, low_prices, close_prices):
        """Calculate Heikin Ashi values"""
        ha_close = (open_prices + high_prices + low_prices + close_prices) / 4
        ha_open = np.zeros_like(open_prices)
        ha_high = np.zeros_like(high_prices)
        ha_low = np.zeros_like(low_prices)

        # Initialize first values
        ha_open[0] = open_prices[0]
        ha_high[0] = max(high_prices[0], max(ha_open[0], ha_close[0]))
        ha_low[0] = min(low_prices[0], min(ha_open[0], ha_close[0]))

        # Calculate subsequent values
        for i in range(1, len(open_prices)):
            ha_open[i] = (ha_open[i-1] + ha_close[i-1]) / 2
            ha_high[i] = max(high_prices[i], max(ha_open[i], ha_close[i]))
            ha_low[i] = min(low_prices[i], min(ha_open[i], ha_close[i]))

        # Apply smoothing
        def smooth_values(values, period):
            smoothed = np.zeros_like(values)
            for i in range(len(values)):
                start_idx = max(0, i - period + 1)
                smoothed[i] = np.mean(values[start_idx:i+1])
            return smoothed

        ha_open = smooth_values(ha_open, self.ha_smooth_period)
        ha_close = smooth_values(ha_close, self.ha_smooth_period)
        ha_high = smooth_values(ha_high, self.ha_smooth_period)
        ha_low = smooth_values(ha_low, self.ha_smooth_period)

        return ha_open, ha_close, ha_high, ha_low

    def scale_ha_values(self, ha_low, ha_high, ha_open, ha_close, window=100):
        """Scale Heikin Ashi values"""
        if len(ha_low) < window:
            window = len(ha_low)

        min_val = np.min(ha_low[-window:])
        max_val = np.max(ha_high[-window:])
        scale = 1.0 / (max_val - min_val) if max_val != min_val else 1

        ha_open_scaled = (ha_open - min_val) * scale
        ha_close_scaled = (ha_close - min_val) * scale
        ha_high_scaled = (ha_high - min_val) * scale
        ha_low_scaled = (ha_low - min_val) * scale

        return ha_open_scaled, ha_close_scaled, ha_high_scaled, ha_low_scaled

    def calc_k_d_lines(self, stoch_rsi, smooth_k=3, smooth_d=3):
        """Calculate K and D lines"""
        k = np.zeros_like(stoch_rsi)
        d = np.zeros_like(stoch_rsi)

        # Calculate K line
        for i in range(len(stoch_rsi)):
            if i < smooth_k:
                k[i] = np.mean(stoch_rsi[:i+1])
            else:
                k[i] = np.mean(stoch_rsi[i-smooth_k+1:i+1])

        # Calculate D line
        for i in range(len(k)):
            if i < smooth_d:
                d[i] = np.mean(k[:i+1])
            else:
                d[i] = np.mean(k[i-smooth_d+1:i+1])

        return k, d

    def update_strategy(self, data):
        """Update strategy with new data and return signal"""
        if len(data) < max(self.length_rsi, self.length_stoch) + 1:
            return None

        # Extract OHLC data
        open_prices = data['Open'].values
        high_prices = data['High'].values
        low_prices = data['Low'].values
        close_prices = data['Close'].values

        # Calculate Heikin Ashi
        ha_open, ha_close, ha_high, ha_low = self.calc_heikin_ashi(
            open_prices, high_prices, low_prices, close_prices
        )

        # Scale Heikin Ashi values
        ha_open_scaled, ha_close_scaled, ha_high_scaled, ha_low_scaled = self.scale_ha_values(
            ha_low, ha_high, ha_open, ha_close, self.window
        )

        # Calculate RSI
        rsi = self.calc_rsi(close_prices, self.length_rsi)

        # Calculate Stochastic RSI
        stoch_rsi = self.calc_stoch_rsi(rsi, self.length_stoch)

        # Apply Butterworth filter if enabled
        if self.use_butterworth_filter:
            stoch_rsi = self.butterworth_filter(stoch_rsi)

        # Calculate K and D lines
        k, d = self.calc_k_d_lines(stoch_rsi, self.smooth_k, self.smooth_d)

        # Store history for signal generation
        self.ha_open_history = ha_open_scaled
        self.ha_close_history = ha_close_scaled
        self.ha_high_history = ha_high_scaled
        self.ha_low_history = ha_low_scaled
        self.rsi_history = rsi
        self.stoch_rsi_history = stoch_rsi
        self.k_history = k
        self.d_history = d

        # Generate signal
        return self.generate_signal()

    def generate_signal(self):
        """Generate buy/sell signal based on current state"""
        if len(self.stoch_rsi_history) < 2 or len(self.ha_close_history) < 1:
            return None

        # Check if Heikin Ashi candle is green (uptrend) or red (downtrend)
        ha_green = self.ha_close_history[-1] > self.ha_open_history[-1]
        ha_red = self.ha_close_history[-1] < self.ha_open_history[-1]

        # Check RSI direction
        rsi_rising = self.stoch_rsi_history[-1] > self.stoch_rsi_history[-2]
        rsi_falling = self.stoch_rsi_history[-1] < self.stoch_rsi_history[-2]

        # Entry condition: RSI rising AND Heikin Ashi candle is green
        long_condition = rsi_rising and ha_green

        # Exit condition: RSI falling AND Heikin Ashi candle is red
        exit_condition = rsi_falling and ha_red

        # Execute trades
        if long_condition and self.last_action != "BUY":
            self.last_action = "BUY"
            return "BUY"
        elif exit_condition and self.last_action != "SELL":
            self.last_action = "SELL"
            return "SELL"

        return None