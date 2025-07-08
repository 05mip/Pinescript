import os
import datetime
from datetime import timedelta
import pandas as pd
import numpy as np
import yfinance as yf
from backtesting import Backtest, Strategy
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import requests
import json

# --- CONFIG ---
# Alpaca API credentials
ALPACA_API_KEY = "PKMHEKQNW8ZI2PPFZ6XC"
ALPACA_SECRET_KEY = "dSULrEZ3PfEOCyuOMLRl0q6QKeWs67HRO35AxdEM"

ASSET = 'XRP-USD'
ASSET_SYMBOL = 'XRPUSD'  # Alpaca uses XRPUSD format for crypto
TIMEFRAME = '1d'
# Using recent dates for Yahoo Finance (last 60 days) and Alpaca
START_DATE = '2025-05-20'  # Adjust this to a date within the last 60 days
END_DATE = '2025-07-03'    # Adjust this to a date within the last 60 days

# Cache file for storing news data
NEWS_CACHE_FILE = 'alpaca_news_cache.json'

# NOTE: Alpaca provides financial news data
# For historical backtesting, you'll need a paid plan

# Initialize VADER sentiment analyzer
sentiment_analyzer = SentimentIntensityAnalyzer()

# --- 1. Download 15m price data for XRP ---
def get_price_data():
    try:
        df = yf.download(ASSET, interval=TIMEFRAME, start=START_DATE, end=END_DATE)
        if df.empty:
            print(f"No price data available for {ASSET} from {START_DATE} to {END_DATE}")
            print("Trying with a more recent date range...")
            # Try with a more recent date range
            recent_start = (datetime.datetime.now() - datetime.timedelta(days=7)).strftime('%Y-%m-%d')
            recent_end = datetime.datetime.now().strftime('%Y-%m-%d')
            df = yf.download(ASSET, interval=TIMEFRAME, start=recent_start, end=recent_end)
            if df.empty:
                raise ValueError(f"No price data available for {ASSET}")
        
        df = df.dropna()
        df.to_csv('price_data_xrp.csv')
        print(f"Successfully downloaded {len(df)} price records for {ASSET}")
        return df
    except Exception as e:
        raise e

# --- 2. News cache management ---
def load_news_cache():
    """Load cached news data from JSON file"""
    if os.path.exists(NEWS_CACHE_FILE):
        try:
            with open(NEWS_CACHE_FILE, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading news cache: {e}")
    return {}

def save_news_cache(cache_data):
    """Save news data to JSON cache file"""
    try:
        with open(NEWS_CACHE_FILE, 'w') as f:
            json.dump(cache_data, f, indent=2, default=str)
    except Exception as e:
        print(f"Error saving news cache: {e}")

def get_cached_news(date_str):
    """Get cached news for a specific date"""
    cache = load_news_cache()
    return cache.get(date_str, None)  # Return None if not found instead of empty list

def cache_news_for_date(date_str, news_data):
    """Cache news data for a specific date"""
    cache = load_news_cache()
    cache[date_str] = news_data
    save_news_cache(cache)

# --- 3. Fetch news from Alpaca API ---
def fetch_news_for_date(date_str):
    """
    Fetch financial news from Alpaca API for a specific date.
    Returns cached data if available, otherwise calls API and caches result.
    """
    # Check cache first
    cached_news = get_cached_news(date_str)
    if cached_news is not None:
        print(f"Using cached news for {date_str}")
        return cached_news
    
    api_key = ALPACA_API_KEY
    secret_key = ALPACA_SECRET_KEY
    
    if not api_key or not secret_key:
        raise Exception("Alpaca API keys not available")
    
    try:
        # Alpaca News API endpoint
        url = "https://data.alpaca.markets/v1beta1/news"
        
        params = {
            'start': (datetime.datetime.strptime(date_str, '%Y-%m-%d') - datetime.timedelta(days=1)).strftime('%Y-%m-%d'),
            'end': date_str,
            'sort': 'desc',
            'symbols': ASSET_SYMBOL,
            'limit': 50,
            'include_content': 'true',
            'exclude_contentless': 'true'
        }
        
        headers = {
            "accept": "application/json",
            "APCA-API-KEY-ID": api_key,
            "APCA-API-SECRET-KEY": secret_key
        }
        
        print(f"Fetching news for {ASSET_SYMBOL} on {date_str}")
        
        response = requests.get(url, params=params, headers=headers)
        response.raise_for_status()
        
        data = response.json()
        
        if 'news' not in data:
            print(f"No news data found for {date_str}")
            cache_news_for_date(date_str, [])
            return []
        
        # Process and cache the news data
        news_items = []
        for article in data['news']:
            try:
                # Parse the created_at timestamp (Alpaca uses 'created_at' not 'published_at')
                # Convert to timezone-naive datetime for consistency
                created_at = datetime.datetime.fromisoformat(article['created_at'].replace('Z', '+00:00'))
                created_at = created_at.replace(tzinfo=None)  # Make timezone-naive
                
                # Combine headline and content for sentiment analysis
                text = article.get('headline', '')
                if article.get('content'):
                    text += ' ' + article['content']
                elif article.get('summary'):
                    text += ' ' + article['summary']
                
                news_items.append({
                    'created_at': created_at.isoformat(),
                    'text': text,
                    'headline': article.get('headline', ''),
                    'summary': article.get('summary', ''),
                    'source': article.get('source', ''),
                    'url': article.get('url', '')
                })
            except KeyError as e:
                print(f"Missing field in article: {e}")
                print(f"Article keys: {list(article.keys())}")
                continue
            except Exception as e:
                print(f"Error processing article: {e}")
                continue
        
        print(f"Fetched {len(news_items)} news articles for {date_str}")
        
        # Cache the results
        cache_news_for_date(date_str, news_items)
        
        return news_items
        
    except requests.exceptions.RequestException as e:
        print(f"Error fetching news from Alpaca for {date_str}: {e}")
        return []
    except Exception as e:
        print(f"Unexpected error for {date_str}: {e}")
        raise e
        return []
    """
    Fetch financial news from Alpaca API for ASSET_SYMBOL between start_time and end_time.
    Alpaca provides news articles for financial assets.
    """
    api_key = ALPACA_API_KEY
    secret_key = ALPACA_SECRET_KEY
    
    if not api_key or not secret_key:
        print("Alpaca API keys not available. Using sample news for testing...")
        raise Exception("Alpaca API keys not available")
    
    try:
        # Format dates for Alpaca API (RFC-3339 format)
        start_str = start_time.strftime('%Y-%m-%d')
        end_str = end_time.strftime('%Y-%m-%d')
        
        # Alpaca News API endpoint
        url = "https://data.alpaca.markets/v1beta1/news"
        
        params = {
            'start': start_str,
            'end': end_str,
            'sort': 'desc',
            'symbols': ASSET_SYMBOL,
            'limit': 50,  # Max articles per request
            'include_content': 'true'
        }
        
        headers = {
            "accept": "application/json",
            "APCA-API-KEY-ID": api_key,
            "APCA-API-SECRET-KEY": secret_key
        }
        
        print(f"Fetching news for {ASSET_SYMBOL} from {start_str} to {end_str}")
        
        response = requests.get(url, params=params, headers=headers)
        response.raise_for_status()
        
        data = response.json()
        
        if 'news' not in data:
            print("No news data found. Using sample news for testing...")
            return []
        
        news_items = []
        for article in data['news']:
            # Parse the published_at timestamp
            published_at = datetime.datetime.fromisoformat(article['published_at'].replace('Z', '+00:00'))
            
            # Combine title and content for sentiment analysis
            text = article.get('title', '')
            if article.get('content'):
                text += ' ' + article['content']
            
            news_items.append({
                'created_at': published_at,
                'text': text,
                'sentiment_score': 0,  # Alpaca doesn't provide sentiment scores, will use VADER
                'relevance_score': 1.0  # Default relevance score
            })
        
        print(f"Fetched {len(news_items)} news articles for {ASSET_SYMBOL}")
        return news_items
        
    except requests.exceptions.RequestException as e:
        print(f"Error fetching news from Alpaca: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")

# --- 4. Sentiment analysis using VADER ---
def analyze_sentiment(texts):
    """
    Analyze sentiment of a list of texts using VADER sentiment analysis.
    Return average sentiment score between -1 (very negative) and 1 (very positive).
    """
    if not texts:
        return 0
    
    sentiment_scores = []
    for text in texts:
        # Get VADER sentiment scores
        scores = sentiment_analyzer.polarity_scores(text)
        # Use compound score which ranges from -1 to 1
        sentiment_scores.append(scores['compound'])
    
    # Return average sentiment score
    return np.mean(sentiment_scores)

# --- 5. Aggregate news and generate signals ---
def generate_signals(price_df):
    signals = []
    
    print(f"Generating signals for {len(price_df)} candles")
    
    # Generate signals for each candle
    for i in range(1, len(price_df)):
        # Get the date for this candle
        candle_date = price_df.index[i].strftime('%Y-%m-%d')
        
        # Fetch news for this specific date (will use cache if available)
        daily_news = fetch_news_for_date(candle_date)
        
        # Convert cached datetime strings back to datetime objects for processing
        processed_news = []
        for article in daily_news:
            try:
                created_at = datetime.datetime.fromisoformat(article['created_at'])
                # Ensure timezone-naive datetime for comparison
                if created_at.tzinfo is not None:
                    created_at = created_at.replace(tzinfo=None)
                processed_news.append({
                    'created_at': created_at,
                    'text': article['text']
                })
            except Exception as e:
                print(f"Error processing article datetime: {e}")
                continue
        
        # Filter news to only include articles published on the previous day
        # at any time during that day
        prev_day_start = price_df.index[i-1].replace(hour=0, minute=0, second=0, microsecond=0)
        prev_day_end = price_df.index[i-1].replace(hour=23, minute=59, second=59, microsecond=999999)
        
        relevant_news = []
        for article in processed_news:
            if prev_day_start <= article['created_at'] <= prev_day_end:
                relevant_news.append(article)
        
        # Use VADER sentiment analysis for the relevant news articles
        texts = [article['text'] for article in relevant_news]
        sentiment = analyze_sentiment(texts)
        
        # Check if previous candle had >2% move (for daily data, this might need adjustment)
        prev_close = price_df['Close'].iloc[i-1].item()
        prev_open = price_df['Open'].iloc[i-1].item()
        move = abs(prev_close - prev_open) / prev_open
        
        # Debug: Print sentiment and move info
        # if i <= 5:  # Only print first few candles to avoid spam
        print(f"Candle {i}: Date={candle_date}, Sentiment={sentiment:.3f}, Move={move:.3f}, News count={len(relevant_news)}")
        
        # Signal: 1 for buy, -1 for sell, 0 for hold
        if sentiment > 0.75:
            signals.append(1)
        elif sentiment < 0.5:
            signals.append(-1)
        else:
            signals.append(0)
    
    signals = [0] + signals  # Align with price_df
    price_df['signal'] = signals
    
    # Debug: Print signal summary
    buy_signals = sum(1 for s in signals if s == 1)
    sell_signals = sum(1 for s in signals if s == -1)
    hold_signals = sum(1 for s in signals if s == 0)
    print(f"Signal Summary: Buy={buy_signals}, Sell={sell_signals}, Hold={hold_signals}")
    
    return price_df

# --- 6. Backtesting.py Strategy ---
class NewsSentimentStrategy(Strategy):
    def init(self):
        self.signal = self.data.signal
    def next(self):
        if self.signal[-1] == 1 and not self.position:
            self.buy()
        elif self.signal[-1] == -1 and self.position:
            self.position.close()

if __name__ == '__main__':
    # Uncomment the line below to clear the news cache
    # clear_news_cache()
    
    price_df = get_price_data()
    price_df = generate_signals(price_df)
    
    # Ensure the DataFrame has the correct structure for backtesting
    # Remove any MultiIndex and ensure we have the required columns
    if isinstance(price_df.columns, pd.MultiIndex):
        price_df.columns = price_df.columns.get_level_values(0)
    
    # Ensure we have all required columns
    required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    for col in required_columns:
        if col not in price_df.columns:
            print(f"Warning: Missing column {col}")
    
    print(f"DataFrame columns: {list(price_df.columns)}")
    print(f"DataFrame shape: {price_df.shape}")
    
    bt = Backtest(
        price_df,
        NewsSentimentStrategy,
        cash=1000,
        commission=0.01,
        exclusive_orders=True,
    )
    stats = bt.run()
    print(stats)
    # bt.plot()

# --- Instructions for API keys ---
# 1. Get your Alpaca API keys at https://alpaca.markets/
# 2. Insert them above or set as environment variables ALPACA_API_KEY and ALPACA_SECRET_KEY
# 3. Alpaca provides financial news data for trading strategies
# 4. News data is cached in alpaca_news_cache.json to avoid repeated API calls
# 5. To clear the cache, delete the alpaca_news_cache.json file
