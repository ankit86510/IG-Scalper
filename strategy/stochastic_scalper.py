import pandas as pd
import numpy as np
from strategy.base import Strategy


class StochasticScalper(Strategy):
    """
    Stochastic Oscillator Scalping Strategy

    Captures moves in trending markets by using stochastic crossovers:
    - LONG: In uptrend, buy when %K crosses above %D in oversold region (<20)
    - SHORT: In downtrend, sell when %K crosses below %D in overbought region (>80)

    Exit when stochastic reaches opposite extreme or reversal crossover occurs.
    """

    def __init__(self, k_period=14, d_period=3, oversold=20, overbought=80,
                 atr_period=14, stop_multiplier=1.0, rr_take=1.5):
        self.k_period = k_period
        self.d_period = d_period
        self.oversold = oversold
        self.overbought = overbought
        self.atr_period = atr_period
        self.stop_multiplier = stop_multiplier
        self.rr_take = rr_take

    def calculate_stochastic(self, df):
        """Calculate Stochastic Oscillator %K and %D"""
        low_min = df['low'].rolling(window=self.k_period).min()
        high_max = df['high'].rolling(window=self.k_period).max()

        k = 100 * ((df['close'] - low_min) / (high_max - low_min))
        d = k.rolling(window=self.d_period).mean()

        return k, d

    def calculate_atr(self, df):
        """Calculate Average True Range"""
        h, l, c = df["high"], df["low"], df["close"]
        prev_c = c.shift(1)
        tr = pd.concat([
            (h - l).abs(),
            (h - prev_c).abs(),
            (l - prev_c).abs()
        ], axis=1).max(axis=1)
        return tr.rolling(self.atr_period).mean()

    def detect_trend(self, df, ma_period=50):
        """Detect trend using moving average"""
        ma = df['close'].ewm(span=ma_period, adjust=False).mean()

        if len(df) < 3:
            return None

        current_price = df['close'].iloc[-1]
        ma_value = ma.iloc[-1]
        ma_slope = ma.iloc[-1] - ma.iloc[-3]

        if current_price > ma_value and ma_slope > 0:
            return "UP"
        elif current_price < ma_value and ma_slope < 0:
            return "DOWN"
        return None

    def on_bar(self, df: pd.DataFrame) -> dict | None:
        """
        Generate trading signals based on stochastic oscillator
        """
        need = max(self.k_period + self.d_period, self.atr_period, 50) + 5
        if len(df) < need:
            return None

        df = df.copy()

        # Calculate indicators
        df['stoch_k'], df['stoch_d'] = self.calculate_stochastic(df)
        df['atr'] = self.calculate_atr(df)
        trend = self.detect_trend(df)

        # Get recent values
        k_prev = df['stoch_k'].iloc[-2]
        k_curr = df['stoch_k'].iloc[-1]
        d_prev = df['stoch_d'].iloc[-2]
        d_curr = df['stoch_d'].iloc[-1]
        atr_val = df['atr'].iloc[-1]

        if np.isnan(k_curr) or np.isnan(d_curr) or np.isnan(atr_val):
            return None

        # Calculate stop loss and take profit
        stop_pts = max(atr_val * self.stop_multiplier, 0.5)
        tp_pts = stop_pts * self.rr_take

        # BULLISH SIGNAL: %K crosses above %D in oversold region during uptrend
        if trend == "UP":
            if k_prev <= d_prev and k_curr > d_curr and k_curr < self.oversold + 10:
                return {
                    "side": "BUY",
                    "stop_pts": float(stop_pts),
                    "tp_pts": float(tp_pts),
                    "meta": {
                        "strategy": "stochastic_scalp",
                        "stoch_k": float(k_curr),
                        "stoch_d": float(d_curr),
                        "trend": trend
                    }
                }

        # BEARISH SIGNAL: %K crosses below %D in overbought region during downtrend
        if trend == "DOWN":
            if k_prev >= d_prev and k_curr < d_curr and k_curr > self.overbought - 10:
                return {
                    "side": "SELL",
                    "stop_pts": float(stop_pts),
                    "tp_pts": float(tp_pts),
                    "meta": {
                        "strategy": "stochastic_scalp",
                        "stoch_k": float(k_curr),
                        "stoch_d": float(d_curr),
                        "trend": trend
                    }
                }

        return None