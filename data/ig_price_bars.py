import pandas as pd

def bars_from_ig(prices_json):
    rows = []
    for b in prices_json.get("prices", []):
        rows.append({
            "ts": pd.to_datetime(b["snapshotTimeUTC"]),
            "open": b["openPrice"]["bid"],
            "high": b["highPrice"]["bid"],
            "low": b["lowPrice"]["bid"],
            "close": b["closePrice"]["bid"],
            "volume": b.get("lastTradedVolume", 0),
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.set_index("ts").sort_index()
