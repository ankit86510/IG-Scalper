"""
EMA Cross Breakout Strategy

Signals on EMA 9/21 crossover with ATR-based stop/TP.
Used as default strategy in run_backtest.py.
"""

import pandas as pd
import numpy as np
from strategy.base import Strategy


class EMACrossBreakout(Strategy):
    """
    EMA Crossover with breakout confirmation.
    - Fast EMA crosses above/below slow EMA
    - ATR-based stop loss and take profit
    """

    def __init__(self, fast: int = 9, slow: int = 21,
                 atr_period: int = 14, rr_take: float = 1.5,
                 stop_multiplier: float = 1.0):
        self.fast = fast
        self.slow = slow
        self.atr_period = atr_period
        self.rr_take = rr_take
        self.stop_multiplier = stop_multiplier

    def calculate_atr(self, df: pd.DataFrame) -> pd.Series:
        h, l, c = df["high"], df["low"], df["close"]
        prev_c = c.shift(1)
        tr = pd.concat([
            (h - l).abs(),
            (h - prev_c).abs(),
            (l - prev_c).abs()
        ], axis=1).max(axis=1)
        return tr.rolling(self.atr_period).mean()

    def on_bar(self, df: pd.DataFrame) -> dict | None:
        need = max(self.slow, self.atr_period) + 5
        if len(df) < need:
            return None

        df = df.copy()
        df['ema_fast'] = df['close'].ewm(span=self.fast, adjust=False).mean()
        df['ema_slow'] = df['close'].ewm(span=self.slow, adjust=False).mean()
        df['atr'] = self.calculate_atr(df)

        if df[['ema_fast', 'ema_slow', 'atr']].isna().any().any():
            return None

        fast_prev = df['ema_fast'].iloc[-2]
        fast_curr = df['ema_fast'].iloc[-1]
        slow_prev = df['ema_slow'].iloc[-2]
        slow_curr = df['ema_slow'].iloc[-1]
        atr_val = df['atr'].iloc[-1]

        stop_pts = max(atr_val * self.stop_multiplier, 0.5)
        tp_pts = stop_pts * self.rr_take

        # Bullish crossover
        if fast_prev <= slow_prev and fast_curr > slow_curr:
            return {
                "side": "BUY",
                "stop_pts": float(stop_pts),
                "tp_pts": float(tp_pts),
                "meta": {
                    "strategy": "ema_cross_breakout",
                    "ema_fast": float(fast_curr),
                    "ema_slow": float(slow_curr)
                }
            }

        # Bearish crossover
        if fast_prev >= slow_prev and fast_curr < slow_curr:
            return {
                "side": "SELL",
                "stop_pts": float(stop_pts),
                "tp_pts": float(tp_pts),
                "meta": {
                    "strategy": "ema_cross_breakout",
                    "ema_fast": float(fast_curr),
                    "ema_slow": float(slow_curr)
                }
            }

        return None
