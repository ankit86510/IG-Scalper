import pandas as pd
import numpy as np

def ema(s, n):
    return s.ewm(span=n, adjust=False).mean()

def atr(df, n=14):
    h, l, c = df["high"], df["low"], df["close"]
    prev_c = c.shift(1)
    tr = pd.concat([(h - l).abs(), (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean()

class EMACrossBreakout:
    def __init__(self, fast=9, slow=21, atr_period=14, rr_take=1.5):
        self.fast = fast
        self.slow = slow
        self.atr_period = atr_period
        self.rr_take = rr_take

    def on_bar(self, df: pd.DataFrame) -> dict | None:
        need = max(self.slow, self.atr_period) + 3
        if len(df) < need:
            return None
        df = df.copy()
        df["ema_f"] = ema(df["close"], self.fast)
        df["ema_s"] = ema(df["close"], self.slow)
        df["atr"] = atr(df, self.atr_period)

        bull_cross = df["ema_f"].iloc[-2] <= df["ema_s"].iloc[-2] and df["ema_f"].iloc[-1] > df["ema_s"].iloc[-1]
        bear_cross = df["ema_f"].iloc[-2] >= df["ema_s"].iloc[-2] and df["ema_f"].iloc[-1] < df["ema_s"].iloc[-1]

        look = 10
        prev_high = df["high"].iloc[-(look+1):-1].max()
        prev_low  = df["low"].iloc[-(look+1):-1].min()
        c = df["close"].iloc[-1]
        a = df["atr"].iloc[-1]
        if np.isnan(a):
            return None

        if bull_cross and c > prev_high:
            stop_pts = max(a * 0.8, 0.5)
            tp_pts = stop_pts * self.rr_take
            return {"side": "BUY", "stop_pts": float(stop_pts), "tp_pts": float(tp_pts), "meta": {"break_above": float(prev_high)}}

        if bear_cross and c < prev_low:
            stop_pts = max(a * 0.8, 0.5)
            tp_pts = stop_pts * self.rr_take
            return {"side": "SELL", "stop_pts": float(stop_pts), "tp_pts": float(tp_pts), "meta": {"break_below": float(prev_low)}}

        return None
