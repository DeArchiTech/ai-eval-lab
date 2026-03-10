import requests
import pandas as pd
import anthropic
from datetime import datetime

API_KEY = "38WMI5CCO81XF6AC"
client = anthropic.Anthropic()

def fetch_crypto(symbol: str) -> pd.DataFrame:
    """Pull daily crypto data from AlphaVantage"""
    url = f"https://www.alphavantage.co/query?function=DIGITAL_CURRENCY_DAILY&symbol={symbol}&market=USD&apikey={API_KEY}"
    r = requests.get(url).json()
    data = r.get("Time Series (Digital Currency Daily)", {})
    df = pd.DataFrame(data).T
    df = df[["4. close"]].rename(columns={"4. close": "close"})
    df["close"] = df["close"].astype(float)
    df.index = pd.to_datetime(df.index)
    return df.sort_index().tail(30)  # last 30 days

def calculate_stats(df: pd.DataFrame) -> dict:
    closes = df["close"]
    return {
        "current_price": round(closes.iloc[-1], 2),
        "30d_change_pct": round((closes.iloc[-1] - closes.iloc[0]) / closes.iloc[0] * 100, 2),
        "volatility_pct": round(closes.pct_change().std() * 100, 2),
        "7d_avg": round(closes.tail(7).mean(), 2),
        "30d_high": round(closes.max(), 2),
        "30d_low": round(closes.min(), 2),
    }

def analyze_with_claude(symbol: str, stats: dict) -> str:
    prompt = f"""You are a crypto analyst. Given these 30-day stats for {symbol}/USD:

Current Price: ${stats['current_price']}
30-Day Change: {stats['30d_change_pct']}%
Daily Volatility: {stats['volatility_pct']}%
7-Day Average: ${stats['7d_avg']}
30-Day High: ${stats['30d_high']}
30-Day Low: ${stats['30d_low']}

Give a 3-sentence analysis: trend, risk level, and a signal (Bullish / Bearish / Neutral). Be direct."""

    response = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=200,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.content[0].text

def run(symbol: str):
    print(f"\n{'='*50}")
    print(f"📊 {symbol}/USD Analysis — {datetime.now().strftime('%Y-%m-%d')}")
    print(f"{'='*50}")
    df = fetch_crypto(symbol)
    stats = calculate_stats(df)
    for k, v in stats.items():
        print(f"  {k}: {v}")
    print(f"\n🤖 Claude's Take:")
    print(analyze_with_claude(symbol, stats))

if __name__ == "__main__":
    for coin in ["SOL", "ETH"]:
        run(coin)
