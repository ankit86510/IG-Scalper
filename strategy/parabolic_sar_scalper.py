import pandas as pd
import numpy as np
from strategy.base import Strategy

class ParabolicSARScalper(Strategy):
    """
    Parabolic SAR Scalper

    - Uses Parabolic SAR reversal signals to enter short-term scalps.
    - Confirms with trend using a long EMA (optional).
    - Simple Parabolic SAR implementation (iterative).
    """
    def __init__(self, acceleration=0.02, maximum=0.2, atr_period=14,
                 trend_ma=50, stop_multiplier=1.0, rr_take=1.5):
        self.af = acceleration
        self.max_af = maximum
        self.atr_period = atr_period
        self.trend_ma = trend_ma
        self.stop_multiplier = stop_multiplier
        self.rr_take = rr_take

    def calculate_atr(self, df):
        h, l, c = df["high"], df["low"], df["close"]
        prev_c = c.shift(1)
        tr = pd.concat([ (h - l).abs(), (h - prev_c).abs(), (l - prev_c).abs() ], axis=1).max(axis=1)
        return tr.rolling(self.atr_period).mean()

    def compute_parabolic_sar(self, high, low, af=0.02, max_af=0.2):
        """
        Iterative Parabolic SAR. Returns pd.Series of SAR values.
        This algorithm is standard and sufficient for scalping use.
        """
        length = len(high)
        sar = np.zeros(length)
        long = True  # start by assuming an up trend (we can refine later)
        ep = high.iloc[0]  # extreme point
        af_curr = af
        sar[0] = low.iloc[0]  # initial SAR

        for i in range(1, length):
            prev_idx = i - 1
            # calculate tentative SAR
            sar[i] = sar[prev_idx] + af_curr * (ep - sar[prev_idx])

            # if in long and SAR is greater than today's low, flip to short
            if long:
                if sar[i] > low.iloc[i]:
                    long = False
                    sar[i] = ep  # set SAR to prior EP
                    ep = low.iloc[i]
                    af_curr = af
                else:
                    # long continues
                    if high.iloc[i] > ep:
                        ep = high.iloc[i]
                        af_curr = min(af_curr + af, max_af)
                    # ensure sar does not exceed prior two lows
                    if i >= 2:
                        sar[i] = min(sar[i], low.iloc[i-1], low.iloc[i-2])
            else:
                # short
                if sar[i] < high.iloc[i]:
                    long = True
                    sar[i] = ep
                    ep = high.iloc[i]
                    af_curr = af
                else:
                    if low.iloc[i] < ep:
                        ep = low.iloc[i]
                        af_curr = min(af_curr + af, max_af)
                    if i >= 2:
                        sar[i] = max(sar[i], high.iloc[i-1], high.iloc[i-2])

        return pd.Series(sar, index=high.index)

    def on_bar(self, df: pd.DataFrame) -> dict | None:
        need = max(self.trend_ma, self.atr_period) + 10
        if len(df) < need:
            return None

        df = df.copy()
        # compute sar & atr
        try:
            df['psar'] = self.compute_parabolic_sar(df['high'], df['low'], af=self.af, max_af=self.max_af)
        except Exception:
            return None

        df['atr'] = self.calculate_atr(df)

        # basic trend via EMA
        df['trend_ma'] = df['close'].ewm(span=self.trend_ma, adjust=False).mean()

        if df['psar'].isna().any() or df['atr'].isna().any():
            return None

        psar_prev = df['psar'].iloc[-2]
        psar_curr = df['psar'].iloc[-1]
        close_prev = df['close'].iloc[-2]
        close_curr = df['close'].iloc[-1]
        trend_val = df['trend_ma'].iloc[-1]
        atr_val = df['atr'].iloc[-1]

        # stop and tp
        stop_pts = max(atr_val * self.stop_multiplier, 0.5)
        tp_pts = stop_pts * self.rr_take

        # Entry logic:
        # If PSAR flips below price -> buy (bullish flip).
        if psar_prev > close_prev and psar_curr < close_curr and close_curr > trend_val:
            return {
                "side": "BUY",
                "stop_pts": float(stop_pts),
                "tp_pts": float(tp_pts),
                "meta": {
                    "strategy": "parabolic_sar_scalp",
                    "psar": float(psar_curr),
                    "trend": "UP"
                }
            }

        # If PSAR flips above price -> sell (bearish flip).
        if psar_prev < close_prev and psar_curr > close_curr and close_curr < trend_val:
            return {
                "side": "SELL",
                "stop_pts": float(stop_pts),
                "tp_pts": float(tp_pts),
                "meta": {
                    "strategy": "parabolic_sar_scalp",
                    "psar": float(psar_curr),
                    "trend": "DOWN"
                }
            }

        return None
