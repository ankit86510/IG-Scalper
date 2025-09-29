import pandas as pd
from strategy.ema_cross_breakout import EMACrossBreakout

def backtest(df, fast=9, slow=21, atr_period=14, rr=1.5):
    strat = EMACrossBreakout(fast, slow, atr_period, rr)
    position = 0
    entry = 0.0
    equity = 10000.0
    trades = []

    for i in range(len(df)):
        sub = df.iloc[: i+1]
        sig = strat.on_bar(sub)
        px = df["close"].iloc[i]
        ts = df.index[i]

        if sig and sig["side"] == "BUY":
            if position <= 0:
                if position < 0:
                    equity += (entry - px)
                    trades.append({"ts": ts, "exit": px, "dir": "COVER"})
                position = 1; entry = px
                trades.append({"ts": ts, "entry": px, "dir": "LONG"})
        elif sig and sig["side"] == "SELL":
            if position >= 0:
                if position > 0:
                    equity += (px - entry)
                    trades.append({"ts": ts, "exit": px, "dir": "SELL"})
                position = -1; entry = px
                trades.append({"ts": ts, "entry": px, "dir": "SHORT"})

    return {"equity": equity, "trades": trades}

if __name__ == "__main__":
    # Example CSV should have columns: ts,open,high,low,close,volume
    df = pd.read_csv("data/sample_1m.csv", parse_dates=["ts"]).set_index("ts").sort_index()
    res = backtest(df)
    print(res["equity"], len(res["trades"]))
