import requests

def list_kraken_pairs():
    url = "https://api.kraken.com/0/public/Assets"
    response = requests.get(url)
    data = response.json()
    pairs = data.get("result", {})
    
    print(f"Total pairs: {len(pairs)}")
    for pair_id, info in pairs.items():
        print(f"{pair_id} -> {info.get('altname')}")

if __name__ == "__main__":
    list_kraken_pairs()
