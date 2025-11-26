import pandas as pd
import numpy as np
from backtesting import Strategy, Backtest
from backtesting.lib import crossover
from datetime import datetime, timedelta
import pytz
import time
import yfinance as yf
from ai1.mc_test import mc_optimize, print_optimization_summary
from ai1.mc_test_10 import mc_optimize_composite, print_composite_summary
from scipy.fft import fft, ifft, fftfreq
from scipy.signal import butter, filtfilt
from scipy import stats
import pandas_ta as ta
from typing import Dict, List, Tuple, Optional
# import talib
from dataclasses import dataclass
from collections import deque
import warnings
# Method 3: Suppress all FutureWarnings (most broad - use cautiously)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="backtesting.backtesting")


@dataclass
class Settings:
    source: str = 'close'
    neighbors_count: int = 8
    max_bars_back: int = 2000
    feature_count: int = 5
    color_compression: int = 1
    show_exits: bool = False
    use_dynamic_exits: bool = False

@dataclass
class FilterSettings:
    use_volatility_filter: bool = True
    use_regime_filter: bool = True
    use_adx_filter: bool = False
    regime_threshold: float = -0.1
    adx_threshold: int = 20

@dataclass
class FeatureConfig:
    name: str
    param_a: int
    param_b: int = 1

class LorentzianClassificationStrategy:
    def __init__(self, 
                 settings: Settings = None,
                 filter_settings: FilterSettings = None,
                 features: List[FeatureConfig] = None,
                 use_ema_filter: bool = False,
                 ema_period: int = 200,
                 use_sma_filter: bool = False,
                 sma_period: int = 200,
                 use_kernel_filter: bool = True,
                 kernel_h: int = 8,
                 kernel_r: float = 8.0,
                 kernel_x: int = 25,
                 kernel_lag: int = 2):
        
        self.settings = settings or Settings()
        self.filter_settings = filter_settings or FilterSettings()
        
        # Default feature configuration
        if features is None:
            self.features = [
                FeatureConfig('RSI', 14, 1),
                FeatureConfig('WT', 10, 11),
                FeatureConfig('CCI', 20, 1),
                FeatureConfig('ADX', 20, 2),
                FeatureConfig('RSI', 9, 1)
            ]
        else:
            self.features = features
            
        # Filter settings
        self.use_ema_filter = use_ema_filter
        self.ema_period = ema_period
        self.use_sma_filter = use_sma_filter
        self.sma_period = sma_period
        self.use_kernel_filter = use_kernel_filter
        self.kernel_h = kernel_h
        self.kernel_r = kernel_r
        self.kernel_x = kernel_x
        self.kernel_lag = kernel_lag
        
        # Initialize arrays for ML model
        self.y_train_array = deque(maxlen=self.settings.max_bars_back)
        self.feature_arrays = [deque(maxlen=self.settings.max_bars_back) for _ in range(5)]
        self.predictions = deque(maxlen=self.settings.neighbors_count)
        self.distances = deque(maxlen=self.settings.neighbors_count)
        
        # Trading state
        self.signal = 0
        self.bars_held = 0
        self.last_signal = 0
        
    def calculate_rsi(self, close: np.ndarray, period: int, smooth: int = 1) -> np.ndarray:
        """Calculate normalized RSI"""
        rsi = ta.rsi(close, timeperiod=period)
        if smooth > 1:
            rsi = ta.sma(rsi, timeperiod=smooth)
        return (rsi - 50) / 50  # Normalize to [-1, 1]
    
    def calculate_wt(self, hlc3: np.ndarray, period1: int, period2: int) -> np.ndarray:
        """Calculate Williams %R transformed (WT)"""
        ema1 = ta.ema(hlc3, timeperiod=period1)
        ema2 = ta.ema(np.abs(hlc3 - ema1), timeperiod=period1)
        ci = (hlc3 - ema1) / (0.015 * ema2)
        wt1 = ta.ema(ci, timeperiod=period2)
        wt2 = ta.sma(wt1, timeperiod=4)
        return (wt1 - wt2) / 100  # Normalize
    
    def calculate_cci(self, high: np.ndarray, low: np.ndarray, close: np.ndarray, 
                     period: int, smooth: int = 1) -> np.ndarray:
        """Calculate normalized CCI"""
        cci = ta.cci(high, low, close, timeperiod=period)
        if smooth > 1:
            cci = ta.sma(cci, timeperiod=smooth)
        return np.tanh(cci / 500)  # Normalize using tanh
    
    def calculate_adx(self, high: np.ndarray, low: np.ndarray, close: np.ndarray, 
                     period: int) -> np.ndarray:
        """Calculate normalized ADX"""
        adx = ta.adx(high, low, close, timeperiod=period)
        return (adx - 50) / 50  # Normalize to [-1, 1]
    
    def calculate_feature(self, feature_config: FeatureConfig, 
                         high: np.ndarray, low: np.ndarray, close: np.ndarray, 
                         hlc3: np.ndarray) -> np.ndarray:
        """Calculate individual feature based on configuration"""
        if feature_config.name == 'RSI':
            return self.calculate_rsi(close, feature_config.param_a, feature_config.param_b)
        elif feature_config.name == 'WT':
            return self.calculate_wt(hlc3, feature_config.param_a, feature_config.param_b)
        elif feature_config.name == 'CCI':
            return self.calculate_cci(high, low, close, feature_config.param_a, feature_config.param_b)
        elif feature_config.name == 'ADX':
            return self.calculate_adx(high, low, close, feature_config.param_a)
        else:
            raise ValueError(f"Unknown feature: {feature_config.name}")
    
    def get_lorentzian_distance(self, i: int, feature_count: int, 
                              current_features: List[float]) -> float:
        """Calculate Lorentzian distance between current features and historical features"""
        distance = 0.0
        for j in range(feature_count):
            if len(self.feature_arrays[j]) > i:
                historical_value = list(self.feature_arrays[j])[-1-i]
                distance += np.log(1 + abs(current_features[j] - historical_value))
        return distance
    
    def calculate_volatility_filter(self, close: np.ndarray, lookback: int = 10) -> bool:
        """Simple volatility filter"""
        if len(close) < lookback:
            return True
        returns = np.diff(np.log(close[-lookback:]))
        volatility = np.std(returns) * np.sqrt(252)
        return volatility < 0.5  # Adjust threshold as needed
    
    def calculate_regime_filter(self, ohlc4: np.ndarray, threshold: float) -> bool:
        """Regime filter based on price momentum"""
        if len(ohlc4) < 50:
            return True
        ma_fast = ta.ema(ohlc4, timeperiod=10)
        ma_slow = ta.ema(ohlc4, timeperiod=50)
        regime = (ma_fast.iloc[-1] - ma_slow.iloc[-1]) / ma_slow.iloc[-1]

        return regime > threshold
    
    def calculate_adx_filter(self, high: np.ndarray, low: np.ndarray, close: np.ndarray, 
                           period: int, threshold: int) -> bool:
        """ADX filter for trend strength"""
        if len(close) < period:
            return True
        adx = ta.ADX(high, low, close, timeperiod=period)
        return adx[-1] > threshold
    
    def rational_quadratic_kernel(self, source: np.ndarray, h: int, r: float, x: int) -> float:
        """Simplified Rational Quadratic Kernel regression"""
        if len(source) < max(h, x):
            return source[-1]
        
        weights = []
        values = []
        
        for i in range(min(h, len(source)-1)):
            weight = (1 + (i**2) / (2 * r * h**2)) ** (-r)
            weights.append(weight)
            values.append(source.iloc[-(i+1)])

        
        if not weights:
            return source[-1]
            
        weighted_sum = sum(w * v for w, v in zip(weights, values))
        weight_sum = sum(weights)
        
        return weighted_sum / weight_sum if weight_sum > 0 else source[-1]
    
    def update_ml_model(self, current_features: List[float], current_label: int):
        """Update the ML model with new data"""
        # Add current label to training array
        self.y_train_array.append(current_label)
        
        # Add current features to feature arrays
        for i, feature in enumerate(current_features):
            self.feature_arrays[i].append(feature)
        
        # Clear predictions and distances for new calculation
        self.predictions.clear()
        self.distances.clear()
        
        # Calculate predictions using approximate nearest neighbors
        last_distance = -1.0
        size_loop = min(self.settings.max_bars_back - 1, len(self.y_train_array) - 1)
        
        for i in range(0, size_loop, 4):  # Every 4 bars for chronological spacing
            if i >= len(self.y_train_array):
                break
                
            distance = self.get_lorentzian_distance(i, self.settings.feature_count, current_features)
            
            if distance >= last_distance:
                last_distance = distance
                self.distances.append(distance)
                self.predictions.append(list(self.y_train_array)[-1-i])
                
                if len(self.predictions) > self.settings.neighbors_count:
                    # Remove oldest prediction and distance
                    distances_list = list(self.distances)
                    if len(distances_list) > self.settings.neighbors_count * 3 // 4:
                        last_distance = distances_list[self.settings.neighbors_count * 3 // 4]
                    self.distances.popleft()
                    self.predictions.popleft()
        
        # Calculate final prediction
        return sum(self.predictions) if self.predictions else 0

# Strategy class for backtesting.py compatibility
class LorentzianStrategy(Strategy):

    use_ema_filter = False
    ema_period = 200
    use_sma_filter = False
    sma_period = 200
    use_kernel_filter = True
    kernel_h = 8
    kernel_r = 8.0
    kernel_x = 25
    kernel_lag = 2
    
    # Initialize the strategy with parameters
    lc_strategy = LorentzianClassificationStrategy(
        use_ema_filter=use_ema_filter,
        ema_period=ema_period,
        use_sma_filter=use_sma_filter,
        sma_period=sma_period,
        use_kernel_filter=use_kernel_filter,
        kernel_h=kernel_h,
        kernel_r=kernel_r,
        kernel_x=kernel_x,
        kernel_lag=kernel_lag
    )
    
    entry_bar = 0
    hold_bars = 10  # Hold position for 4 bars
        
    def init(self):
        """Initialize the strategy"""
        # Pre-calculate all indicators
        high = pd.Series(self.data.High)
        low = pd.Series(self.data.Low)
        close = pd.Series(self.data.Close)
        open_price = pd.Series(self.data.Open)
        
        hlc3 = (high + low + close) / 3
        ohlc4 = (open_price + high + low + close) / 4
        
        # Calculate all features for the entire dataset
        self.features_data = []
        for i, feature_config in enumerate(self.lc_strategy.features[:self.lc_strategy.settings.feature_count]):
            feature_values = self.lc_strategy.calculate_feature(feature_config, high, low, close, hlc3)

            # If DataFrame, take first column (or specific one)
            if isinstance(feature_values, pd.DataFrame):
                feature_values = feature_values.iloc[:, 0]

            self.features_data.append(feature_values)
        
        # Calculate moving averages for trend filters
        if self.use_ema_filter:
            self.ema = self.I(ta.ema, self.data.Close, timeperiod=self.ema_period)
        if self.use_sma_filter:
            self.sma = self.I(ta.sma, self.data.Close, timeperiod=self.sma_period)
        
        # Pre-generate all signals
        self.signals = np.zeros(len(self.data))
        self.predictions = np.zeros(len(self.data))
        
        # Process each bar to generate signals
        for i in range(max(50, self.lc_strategy.settings.max_bars_back // 10), len(self.data)):
            self._process_bar(i)
    
    def _process_bar(self, i):
        """Process a single bar to generate signals"""
        # Calculate current features
        current_features = []
        for j in range(self.lc_strategy.settings.feature_count):
            if i < len(self.features_data[j]) and not np.isnan(self.features_data[j].iloc[i]):
                current_features.append(self.features_data[j].iloc[i])
            else:
                return  # Skip if any feature is NaN
        
        # Calculate training label (looking forward 4 bars)
        if i + 4 < len(self.data):
            future_price = self.data.Close[i + 4]
            current_price = self.data.Close[i]
            if future_price > current_price:
                label = 1  # Long
            elif future_price < current_price:
                label = -1  # Short
            else:
                label = 0  # Neutral
        else:
            label = 0
        
        # Update ML model and get prediction
        prediction = self.lc_strategy.update_ml_model(current_features, label)
        self.predictions[i] = prediction
        
        # Apply filters
        high = pd.Series(self.data.High)
        low = pd.Series(self.data.Low)
        close = pd.Series(self.data.Close)
        open_price = pd.Series(self.data.Open)
        ohlc4 = (open_price + high + low + close) / 4
        
        volatility_filter = True
        if self.lc_strategy.filter_settings.use_volatility_filter:
            volatility_filter = self.lc_strategy.calculate_volatility_filter(close[:i+1])
        
        regime_filter = True
        if self.lc_strategy.filter_settings.use_regime_filter:
            regime_filter = self.lc_strategy.calculate_regime_filter(ohlc4[:i+1], 
                                                       self.lc_strategy.filter_settings.regime_threshold)
        
        adx_filter = True
        if self.lc_strategy.filter_settings.use_adx_filter:
            adx_filter = self.lc_strategy.calculate_adx_filter(high[:i+1], low[:i+1], close[:i+1],
                                                 14, self.lc_strategy.filter_settings.adx_threshold)
        
        filter_all = volatility_filter and regime_filter and adx_filter
        
        # Generate base signal
        if prediction > 0 and filter_all:
            new_signal = 1
        elif prediction < 0 and filter_all:
            new_signal = -1
        else:
            new_signal = self.lc_strategy.signal
        
        # Apply trend filters
        is_ema_uptrend = True
        is_ema_downtrend = True
        if self.use_ema_filter and hasattr(self, 'ema'):
            if i < len(self.ema) and not np.isnan(self.ema[i]):
                is_ema_uptrend = close[i] > self.ema[i]
                is_ema_downtrend = close[i] < self.ema[i]
        
        is_sma_uptrend = True
        is_sma_downtrend = True
        if self.use_sma_filter and hasattr(self, 'sma'):
            if i < len(self.sma) and not np.isnan(self.sma[i]):
                is_sma_uptrend = close[i] > self.sma[i]
                is_sma_downtrend = close[i] < self.sma[i]
        
        # Kernel filter
        kernel_bullish = True
        kernel_bearish = True
        if self.use_kernel_filter and i > self.kernel_h:
            kernel_estimate = self.lc_strategy.rational_quadratic_kernel(close[:i+1], 
                                                           self.kernel_h, 
                                                           self.kernel_r, 
                                                           self.kernel_x)
            kernel_prev = self.lc_strategy.rational_quadratic_kernel(close[:i], 
                                                       self.kernel_h, 
                                                       self.kernel_r, 
                                                       self.kernel_x)
            kernel_bullish = kernel_estimate > kernel_prev
            kernel_bearish = kernel_estimate <= kernel_prev
        
        # Update signal state
        signal_changed = new_signal != self.lc_strategy.signal
        self.lc_strategy.signal = new_signal
        
        if signal_changed:
            self.lc_strategy.bars_held = 0
        else:
            self.lc_strategy.bars_held += 1
        
        self.signals[i] = self.lc_strategy.signal
    
    def next(self):
        """Process next bar"""
        current_index = len(self.data) - 1
        
        # Check if we have a signal for current bar
        if current_index >= len(self.signals):
            return
        
        current_signal = self.signals[current_index]
        
        # Exit conditions (after holding for specified bars)
        # if self.position:
        #     bars_since_entry = current_index - self.entry_bar
        #     if bars_since_entry >= self.hold_bars:
        #         self.position.close()
        #         return
        
        # Entry conditions
        if not self.position and current_signal != 0:
            # Apply additional trend filter checks
            is_ema_uptrend = True
            is_ema_downtrend = True
            if self.use_ema_filter and hasattr(self, 'ema'):
                is_ema_uptrend = self.data.Close[-1] > self.ema[-1]
                is_ema_downtrend = self.data.Close[-1] < self.ema[-1]
            
            is_sma_uptrend = True
            is_sma_downtrend = True
            if self.use_sma_filter and hasattr(self, 'sma'):
                is_sma_uptrend = self.data.Close[-1] > self.sma[-1]
                is_sma_downtrend = self.data.Close[-1] < self.sma[-1]
            
            # Long entry
            if (current_signal == 1 ):
                self.buy()
                self.entry_bar = current_index
            
            # Short entry
            # elif (current_signal == -1 and is_ema_downtrend and is_sma_downtrend):
            #     self.sell()
            #     self.entry_bar = current_index

        if self.position and current_signal != 0:
            is_ema_uptrend = True
            is_ema_downtrend = True
            if self.use_ema_filter and hasattr(self, 'ema'):
                is_ema_uptrend = self.data.Close[-1] > self.ema[-1]
                is_ema_downtrend = self.data.Close[-1] < self.ema[-1]
            
            is_sma_uptrend = True
            is_sma_downtrend = True
            if self.use_sma_filter and hasattr(self, 'sma'):
                is_sma_uptrend = self.data.Close[-1] > self.sma[-1]
                is_sma_downtrend = self.data.Close[-1] < self.sma[-1]

            if (current_signal == -1  ):
                self.position.close()
                # self.entry_bar = current_index

def fetch_data(symbol, start_date, end_date, interval='15m', max_retries=3, delay=2):
    """
    Fetch data from Yahoo Finance with retry mechanism
    :param symbol: Stock symbol (e.g., 'AAPL')
    :param start_date: Start date (str or datetime)
    :param end_date: End date (str or datetime)
    :param interval: Data interval ('1d', '1h', '15m', etc.)
    :param max_retries: Maximum number of retry attempts
    :param delay: Delay between retries in seconds
    :return: DataFrame with OHLC data
    """
    # Convert dates to UTC if they're not already
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


# Example usage
if __name__ == "__main__":
    strategy = LorentzianStrategy
    param_space = {
        # RSI Parameters
        'rsi_length': lambda: np.random.randint(5, 25),
        'rsi_overbought': lambda: np.random.randint(55, 80),
        'rsi_oversold': lambda: np.random.randint(20, 45),
        'rsi_smoothing': lambda: np.random.randint(1, 5),  # 1-4 periods smoothing
        
        # MACD Parameters
        'macd_fast': lambda: np.random.randint(5, 20),
        'macd_slow': lambda: np.random.randint(21, 40),
        'macd_signal': lambda: np.random.randint(5, 15),
        
        # Strategy Conditions
        'buy_condition': lambda: np.random.randint(1, 9),   # 1-8
        'sell_condition': lambda: np.random.randint(1, 10), # 1-9 (9 is TP/SL)
        
        # TP/SL Parameters (only used when sell_condition = 9)
        'take_profit_pct': lambda: np.random.uniform(0.5, 5.0),
        'stop_loss_pct': lambda: np.random.uniform(0.5, 3.0),
    }

    if 0:
        results = mc_optimize_composite(
            strategy=strategy,
            param_space=param_space,
            n_iterations=1000,
            validation_permutations=10,
            n_threads=100,
            output_file='composite_results.json',
            verbose=True
        )
        # print_optimization_summary(results)
        print_composite_summary(results)
    else:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=59)  # 59 days of data
        data = fetch_data("RAY-USD", start_date, end_date, "30m")
        bt = Backtest(data, strategy, cash=100000, commission=.001, finalize_trades=True)
        # stats = bt.run(**results['best_params'])
        stats = bt.run()
        print(stats)
        bt.plot()