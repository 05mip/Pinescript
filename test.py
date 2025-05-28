import requests
import pandas as pd

# Define the API endpoint and parameters
url = "https://api.pionex.com/api/v1/market/klines"
params = {
    "symbol": "XRP_USDT",  # Trading pair
    "interval": "15M",     # Valid interval
    "limit": 500           # Number of data points to retrieve
}

# Send the GET request
response = requests.get(url, params=params)

# Check if the request was successful
if response.status_code == 200:
    data = response.json()
    if data.get("result"):
        klines = data["data"]["klines"]
        # Convert the data to a pandas DataFrame
        df = pd.DataFrame(klines)
        # Convert 'time' from milliseconds to datetime
        df["time"] = pd.to_datetime(df["time"], unit='ms')
        # Rename 'time' column to 'timestamp' for consistency
        df.rename(columns={"time": "timestamp"}, inplace=True)
        print(df)
    else:
        print("API returned an error:", data)
else:
    print(f"Error fetching data: {response.status_code}")
