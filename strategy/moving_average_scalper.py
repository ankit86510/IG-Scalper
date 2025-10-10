import pandas as pd
import numpy as np
from strategy.base import Strategy

class MovingAverageScalper(Strategy):
    """
    Moving Average Crossover Scalper

    - Uses fast & slow MA crossover for entry.
    - Confirms trend with a longer trend MA (trend_ma).
    - Places trades in direction of trend when crossover occurs near the fast MA slope.
    """
    def __init__(self, fast_ma=5, slow_ma=20, trend_ma=200, atr_period=14,
                 stop_multiplier=1.0, rr_take=1.5):
        self.fast_ma = fast_ma
        self.slow_ma = slow_ma
        self.trend_ma = trend_ma
        self.atr_period = atr_period
        self.stop_multiplier = stop_multiplier
        self.rr_take = rr_take

    def calculate_ma(self, df):
        fast = df['close'].ewm(span=self.fast_ma, adjust=False).mean()
        slow = df['close'].ewm(span=self.slow_ma, adjust=False).mean()
        trend = df['close'].ewm(span=self.trend_ma, adjust=False).mean()
        return fast, slow, trend

    def calculate_atr(self, df):
        h, l, c = df["high"], df["low"], df["close"]
        prev_c = c.shift(1)
        tr = pd.concat([ (h - l).abs(), (h - prev_c).abs(), (l - prev_c).abs() ], axis=1).max(axis=1)
        return tr.rolling(self.atr_period).mean()

    def on_bar(self, df: pd.DataFrame) -> dict | None:
        need = max(self.slow_ma, self.trend_ma, self.atr_period) + 5
        if len(df) < need:
            return None

        df = df.copy()
        df['fast_ma'], df['slow_ma'], df['trend_ma'] = self.calculate_ma(df)
        df['atr'] = self.calculate_atr(df)

        if df['fast_ma'].isna().any() or df['slow_ma'].isna().any() or df['atr'].isna().any():
            return None

        f_prev = df['fast_ma'].iloc[-2]
        f_curr = df['fast_ma'].iloc[-1]
        s_prev = df['slow_ma'].iloc[-2]
        s_curr = df['slow_ma'].iloc[-1]
        trend_val = df['trend_ma'].iloc[-1]
        price = df['close'].iloc[-1]
        atr_val = df['atr'].iloc[-1]

        # Determine trend direction using trend_ma (price vs trend_ma and slope)
        ma_slope = df['trend_ma'].iloc[-1] - df['trend_ma'].iloc[-3]
        if price > trend_val and ma_slope > 0:
            trend = "UP"
        elif price < trend_val and ma_slope < 0:
            trend = "DOWN"
        else:
            trend = None

        # Stop & TP in points
        stop_pts = max(atr_val * self.stop_multiplier, 0.5)
        tp_pts = stop_pts * self.rr_take

        # BUY signal: fast crosses above slow and trend UP
        if trend == "UP" and f_prev <= s_prev and f_curr > s_curr:
            return {
                "side": "BUY",
                "stop_pts": float(stop_pts),
                "tp_pts": float(tp_pts),
                "meta": {
                    "strategy": "moving_average_scalp",
                    "fast_ma": float(f_curr),
                    "slow_ma": float(s_curr),
                    "trend": trend
                }
            }

        # SELL signal: fast crosses below slow and trend DOWN
        if trend == "DOWN" and f_prev >= s_prev and f_curr < s_curr:
            return {
                "side": "SELL",
                "stop_pts": float(stop_pts),
                "tp_pts": float(tp_pts),
                "meta": {
                    "strategy": "moving_average_scalp",
                    "fast_ma": float(f_curr),
                    "slow_ma": float(s_curr),
                    "trend": trend
                }
            }

        return None
