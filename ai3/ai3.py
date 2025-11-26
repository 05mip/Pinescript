import pandas as pd
import numpy as np
import pandas_ta as ta
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
import os
import warnings
warnings.filterwarnings('ignore')

class CryptoTradingModel:
    def __init__(self, data_folder=r'C:\Users\micha\OneDrive\Documents\CodeProjects\Pinescript\data_30m', label_folder=r'C:\Users\micha\OneDrive\Documents\CodeProjects\Pinescript\labelled_30'):
        self.data_folder = data_folder
        self.label_folder = label_folder
        self.scaler = StandardScaler()
        self.model = None
        
    def load_all_datasets(self):
        """Load and combine all 9 datasets"""
        all_data = []
        
        # Common crypto symbols (adjust these to match your actual files)
        symbols = ['ADA-USD', 'AVAX-USD', 'BNB-USD', 'BTC-USD', 'DOGE-USD', 
                  'DOT-USD', 'ETH-USD', 'SOL-USD', 'XRP-USD']
        
        for symbol in symbols:
            try:
                print(f"Loading {symbol}...")
                df = self.load_and_prepare_data(symbol)
                df = self.create_features(df)
                df['Symbol'] = symbol  # Add symbol identifier
                all_data.append(df)
                print(f"  - Loaded {len(df)} rows")
            except Exception as e:
                raise e
                print(f"  - Error loading {symbol}: {e}")
                continue
        
        if not all_data:
            raise ValueError("No datasets could be loaded!")
        
        # Combine all datasets
        combined_df = pd.concat(all_data, ignore_index=True)
        print(f"\nCombined dataset: {len(combined_df)} total rows")
        
        return combined_df
    
    def load_and_prepare_data(self, symbol):
        """Load price data and labels, merge them"""
        # Load price data
        price_file = f"{symbol}_30m.csv"
        price_path = os.path.join(self.data_folder, price_file)
        df = pd.read_csv(price_path, header=0, skiprows=[1,2])
        df = df.rename(columns={'Price': 'Datetime'})
        df['Datetime'] = pd.to_datetime(df['Datetime'])
        
        # Convert price columns to numeric
        price_cols = ['Close', 'High', 'Low', 'Open', 'Volume']
        for col in price_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Load labels
        label_file = f"{symbol.replace('-', '_')}_labelled.csv"
        label_path = os.path.join(self.label_folder, label_file)
        labels = pd.read_csv(label_path)
        labels['Datetime'] = pd.to_datetime(labels['Datetime'])
        
        # Merge data with labels
        df = pd.merge(df, labels, on='Datetime', how='left')
        
        return df
    
    def create_features(self, df):
        """Create technical indicators and features"""
        # Calculate MACD
        macd_data = ta.macd(df['Close'])
        df['MACD'] = macd_data['MACD_12_26_9']
        df['MACD_Signal'] = macd_data['MACDs_12_26_9']
        df['MACD_Histogram'] = macd_data['MACDh_12_26_9']
        
        # Calculate RSI
        df['RSI'] = ta.rsi(df['Close'])
        
        # MACD-specific features
        df['MACD_Signal_Distance'] = df['MACD'] - df['MACD_Signal']
        df['MACD_Histogram_Direction'] = np.where(
            df['MACD_Histogram'] > df['MACD_Histogram'].shift(1), 1, 0
        )
        
        # Volume features
        df['Volume_SMA'] = ta.sma(df['Volume'], length=10)
        df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA']
        df['High_Low_Ratio'] = df['High'] / df['Low']
        
        # Extended lag features (15-40 candles where possible)
        # Start with key periods and use what's available
        lag_periods = [1, 2, 3, 5, 10, 15, 20, 30, 40]
        
        for lag in lag_periods:
            if lag <= len(df):  # Only create if we have enough data
                df[f'Close_lag_{lag}'] = df['Close'].shift(lag)
                df[f'RSI_lag_{lag}'] = df['RSI'].shift(lag)
                df[f'MACD_lag_{lag}'] = df['MACD'].shift(lag)
                df[f'MACD_Signal_lag_{lag}'] = df['MACD_Signal'].shift(lag)
                df[f'MACD_Histogram_lag_{lag}'] = df['MACD_Histogram'].shift(lag)
                df[f'Volume_Ratio_lag_{lag}'] = df['Volume_Ratio'].shift(lag)
        
        # Derived features
        df['RSI_Change'] = df['RSI'] - df['RSI'].shift(1)
        df['MACD_Signal_Cross'] = np.where(df['MACD'] > df['MACD_Signal'], 1, 0)
        
        return df
    
    def prepare_training_data(self, df):
        """Prepare features and labels for training"""
        # Define lag periods
        lag_periods = [1, 2, 3, 5, 10, 15, 20, 30, 40]
        
        # Build feature columns dynamically based on available lags
        feature_cols = [
            'MACD', 'MACD_Signal', 'MACD_Histogram', 'RSI',
            'High_Low_Ratio', 'Volume_Ratio',
            'MACD_Signal_Distance', 'MACD_Histogram_Direction',
            'RSI_Change', 'MACD_Signal_Cross'
        ]
        
        # Add lag features that exist in the dataframe
        for lag in lag_periods:
            lag_features = [
                f'Close_lag_{lag}', f'RSI_lag_{lag}', f'MACD_lag_{lag}',
                f'MACD_Signal_lag_{lag}', f'MACD_Histogram_lag_{lag}', f'Volume_Ratio_lag_{lag}'
            ]
            for feature in lag_features:
                if feature in df.columns:
                    feature_cols.append(feature)
        
        # Only use rows with labels (not NaN)
        labeled_data = df.dropna(subset=['Label']).copy()
        
        # Remove rows with NaN in features
        labeled_data = labeled_data.dropna(subset=feature_cols)
        
        X = labeled_data[feature_cols]
        y = labeled_data['Label'].astype(int)
        
        print(f"Training data shape: {X.shape}")
        print(f"Available features: {len(feature_cols)}")
        print(f"Label distribution: {y.value_counts().sort_index()}")
        
        return X, y, labeled_data['Datetime']
    
    def train_models(self, X, y):
        """Train multiple models and compare performance"""
        # Use TimeSeriesSplit with more folds for better validation
        tscv = TimeSeriesSplit(n_splits=10)
        
        # Models with regularization
        models = {
            'Random Forest': RandomForestClassifier(
                n_estimators=100, 
                max_depth=15, 
                min_samples_split=5,
                min_samples_leaf=3,
                random_state=42, 
                class_weight='balanced'
            ),
            'XGBoost': xgb.XGBClassifier(
                n_estimators=100, 
                max_depth=8, 
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,  # L1 regularization
                reg_lambda=0.1, # L2 regularization
                random_state=42, 
                eval_metric='mlogloss'
            ),
            'Logistic Regression': LogisticRegression(
                random_state=42, 
                max_iter=2000, 
                class_weight='balanced', 
                C=0.01,  # Strong regularization
                penalty='l2'
            ),
            'Naive Bayes': GaussianNB()
        }
        
        results = {}
        
        for name, model in models.items():
            print(f"\nTraining {name}...")
            
            # Scale features for models that need it
            if name in ['Logistic Regression']:
                X_scaled = self.scaler.fit_transform(X)
                X_train, X_test, y_train, y_test = train_test_split(
                    X_scaled, y, test_size=0.2, random_state=42, stratify=y
                )
            else:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42, stratify=y
                )
            
            # Train model
            model.fit(X_train, y_train)
            
            # Predict
            y_pred = model.predict(X_test)
            
            # Store results
            results[name] = {
                'model': model,
                'accuracy': (y_pred == y_test).mean(),
                'predictions': y_pred,
                'actual': y_test
            }
            
            print(f"{name} Accuracy: {results[name]['accuracy']:.4f}")
            print(f"Classification Report:\n{classification_report(y_test, y_pred)}")
        
        return results
    
    def get_feature_importance(self, model, feature_names):
        """Get feature importance for tree-based models"""
        if hasattr(model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            return importance_df
        return None
    
    def predict_trading_signals(self, df, model, feature_cols):
        """Generate trading signals for new data"""
        # Prepare features
        X_new = df[feature_cols].dropna()
        
        if hasattr(model, 'predict_proba'):
            # Get probabilities
            probabilities = model.predict_proba(X_new)
            predictions = model.predict(X_new)
            
            # Create results DataFrame
            results = pd.DataFrame({
                'Datetime': df.loc[X_new.index, 'Datetime'],
                'Close': df.loc[X_new.index, 'Close'],
                'Prediction': predictions,
                'Sell_Prob': probabilities[:, 0],
                'Hold_Prob': probabilities[:, 1],
                'Buy_Prob': probabilities[:, 2]
            })
        else:
            predictions = model.predict(X_new)
            results = pd.DataFrame({
                'Datetime': df.loc[X_new.index, 'Datetime'],
                'Close': df.loc[X_new.index, 'Close'],
                'Prediction': predictions
            })
        
        return results

# Usage example
def main():
    # Initialize the model
    trader = CryptoTradingModel()
    
    # Load and combine all datasets
    print("Loading all datasets...")
    df = trader.load_all_datasets()
    
    # Prepare training data
    print("Preparing training data...")
    X, y, dates = trader.prepare_training_data(df)
    
    # Train models
    print("Training models...")
    results = trader.train_models(X, y)
    
    # Find best model
    best_model_name = max(results.keys(), key=lambda k: results[k]['accuracy'])
    best_model = results[best_model_name]['model']
    
    print(f"\nBest model: {best_model_name} with accuracy: {results[best_model_name]['accuracy']:.4f}")
    
    # Feature importance for tree-based models
    if best_model_name in ['Random Forest', 'XGBoost']:
        importance = trader.get_feature_importance(best_model, X.columns)
        print(f"\nTop 15 Important Features:")
        print(importance.head(15))
    
    # Generate trading signals for the entire dataset
    feature_cols = X.columns.tolist()
    signals = trader.predict_trading_signals(df, best_model, feature_cols)
    
    print(f"\nGenerated {len(signals)} trading signals")
    print("Sample signals:")
    print(signals.head())
    
    return trader, best_model, results

if __name__ == "__main__":
    trader, best_model, results = main()