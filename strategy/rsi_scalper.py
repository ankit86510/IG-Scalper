import pandas as pd
import numpy as np
from strategy.base import Strategy

class RSIScalper(Strategy):
    """
    RSI Pullback Scalper

    - Uses RSI to identify oversold/overbought short-term conditions within a trend.
    - Confirms trend with multiple EMAs (ma_fast, ma_med, ma_slow).
    - Entries when RSI pulls back into an oversold/overbought zone then reverts.
    """
    def __init__(self, rsi_period=14, oversold=30, overbought=70,
                 ma_fast=5, ma_med=20, ma_slow=50, atr_period=14,
                 stop_multiplier=1.0, rr_take=1.5):
        self.rsi_period = rsi_period
        self.oversold = oversold
        self.overbought = overbought
        self.ma_fast = ma_fast
        self.ma_med = ma_med
        self.ma_slow = ma_slow
        self.atr_period = atr_period
        self.stop_multiplier = stop_multiplier
        self.rr_take = rr_take

    def calculate_rsi(self, series: pd.Series) -> pd.Series:
        delta = series.diff()
        up = delta.clip(lower=0)
        down = -1 * delta.clip(upper=0)
        ma_up = up.ewm(alpha=1/self.rsi_period, adjust=False).mean()
        ma_down = down.ewm(alpha=1/self.rsi_period, adjust=False).mean()
        rs = ma_up / ma_down
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def calculate_atr(self, df):
        h, l, c = df["high"], df["low"], df["close"]
        prev_c = c.shift(1)
        tr = pd.concat([ (h - l).abs(), (h - prev_c).abs(), (l - prev_c).abs() ], axis=1).max(axis=1)
        return tr.rolling(self.atr_period).mean()

    def on_bar(self, df: pd.DataFrame) -> dict | None:
        need = max(self.ma_slow, self.rsi_period, self.atr_period) + 5
        if len(df) < need:
            return None

        df = df.copy()
        df['rsi'] = self.calculate_rsi(df['close'])
        df['ema_fast'] = df['close'].ewm(span=self.ma_fast, adjust=False).mean()
        df['ema_med'] = df['close'].ewm(span=self.ma_med, adjust=False).mean()
        df['ema_slow'] = df['close'].ewm(span=self.ma_slow, adjust=False).mean()
        df['atr'] = self.calculate_atr(df)

        if df[['rsi','ema_fast','ema_med','ema_slow','atr']].isna().any().any():
            return None

        rsi_prev = df['rsi'].iloc[-2]
        rsi_curr = df['rsi'].iloc[-1]
        price = df['close'].iloc[-1]
        atr_val = df['atr'].iloc[-1]

        # Trend determination: EMAs aligned
        if df['ema_fast'].iloc[-1] > df['ema_med'].iloc[-1] > df['ema_slow'].iloc[-1]:
            trend = "UP"
        elif df['ema_fast'].iloc[-1] < df['ema_med'].iloc[-1] < df['ema_slow'].iloc[-1]:
            trend = "DOWN"
        else:
            trend = None

        stop_pts = max(atr_val * self.stop_multiplier, 0.5)
        tp_pts = stop_pts * self.rr_take

        # BUY: in uptrend, RSI was oversold and now rising
        if trend == "UP":
            if rsi_prev <= self.oversold and rsi_curr > rsi_prev:
                return {
                    "side": "BUY",
                    "stop_pts": float(stop_pts),
                    "tp_pts": float(tp_pts),
                    "meta": {
                        "strategy": "rsi_scalp",
                        "rsi": float(rsi_curr),
                        "trend": trend
                    }
                }

        # SELL: in downtrend, RSI was overbought and now falling
        if trend == "DOWN":
            if rsi_prev >= self.overbought and rsi_curr < rsi_prev:
                return {
                    "side": "SELL",
                    "stop_pts": float(stop_pts),
                    "tp_pts": float(tp_pts),
                    "meta": {
                        "strategy": "rsi_scalp",
                        "rsi": float(rsi_curr),
                        "trend": trend
                    }
                }

        return None
