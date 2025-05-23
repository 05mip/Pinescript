import yfinance as yf
import time
from datetime import datetime

def get_xrp_price():
    xrp = yf.Ticker("XRP-USD")
    current_price = xrp.info['regularMarketPrice']
    return current_price

def main():
    print("Starting XRP-USD price monitor...")
    print("Press Ctrl+C to stop")
    
    try:
        while True:
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            price = get_xrp_price()
            print(f"[{current_time}] XRP-USD Price: ${price:.4f}")
            time.sleep(30)
    except KeyboardInterrupt:
        print("\nStopping price monitor...")

if __name__ == "__main__":
    main()
