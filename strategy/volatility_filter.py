"""
Volatility Regime Filter

ATR-percentile gate that blocks trading when market conditions are
too volatile (chaos) or too quiet (no edge).

Integrates before strategy.on_bar() in the epic-processing loop.
"""

import logging
from collections import deque

import numpy as np
import pandas as pd

log = logging.getLogger("ig-scalper")


class VolatilityRegimeFilter:
    """ATR-percentile gate that blocks trading in extreme volatility regimes."""

    def __init__(self, config: dict):
        """
        Initialise filter from config dict.

        Config keys:
          enabled: bool (default True)
          atr_period: int (default 14)
          lookback_bars: int (default 100)
          lower_percentile: float (default 20.0)
          upper_percentile: float (default 80.0)
        """
        self.enabled = config.get("enabled", True)
        self.atr_period = config.get("atr_period", 14)
        self.lookback_bars = config.get("lookback_bars", 100)
        self.lower_percentile = config.get("lower_percentile", 20.0)
        self.upper_percentile = config.get("upper_percentile", 80.0)

        self._history: deque = deque(maxlen=self.lookback_bars)

    def compute_atr_ratio(self, df: pd.DataFrame) -> float:
        """
        Compute ATR(period) / close for the penultimate bar.

        Uses the standard True Range definition:
          TR = max(high - low, |high - prev_close|, |low - prev_close|)
          ATR = rolling mean of TR over atr_period bars.

        Returns the ratio ATR / close at iloc[-2] (penultimate bar).
        """
        h = df["high"]
        l = df["low"]
        c = df["close"]
        prev_c = c.shift(1)

        tr = pd.concat([
            (h - l).abs(),
            (h - prev_c).abs(),
            (l - prev_c).abs(),
        ], axis=1).max(axis=1)

        atr = tr.rolling(self.atr_period).mean()

        # Use penultimate bar (last bar is forming/incomplete)
        atr_val = atr.iloc[-2]
        close_val = c.iloc[-2]

        return float(atr_val / close_val)

    def update_history(self, atr_ratio: float) -> None:
        """Append atr_ratio to rolling history, automatically trimmed to lookback_bars."""
        self._history.append(atr_ratio)

    def compute_percentile(self, current_ratio: float) -> float:
        """
        Compute percentile rank of current_ratio within history.

        Returns (count of history values <= current_ratio) / len(history) * 100.
        Range: [0, 100].
        """
        history_arr = np.array(self._history)
        count_le = np.sum(history_arr <= current_ratio)
        return float(count_le / len(history_arr) * 100)

    def allow_trading(self, df: pd.DataFrame) -> tuple[bool, dict]:
        """
        Determine whether trading should proceed.

        Returns (allowed: bool, metadata: dict).
        metadata includes: atr_ratio, percentile, reason.

        Fail-open rules:
        - If disabled: always allow.
        - If history has < 20 entries: allow (insufficient data).
        - If an error occurs: allow (fail-open philosophy).
        """
        # Disabled filter — pass through
        if not self.enabled:
            return True, {
                "atr_ratio": None,
                "percentile": None,
                "reason": "volatility filter disabled",
            }

        try:
            atr_ratio = self.compute_atr_ratio(df)
        except Exception as e:
            log.warning(f"Volatility filter error computing ATR ratio: {e}")
            return True, {
                "atr_ratio": None,
                "percentile": None,
                "reason": f"error computing ATR ratio: {e}",
            }

        # Update history with current observation
        self.update_history(atr_ratio)

        # Insufficient history — allow trading
        if len(self._history) < 20:
            log.debug(
                f"Volatility filter: insufficient history "
                f"({len(self._history)}/20), allowing trading"
            )
            return True, {
                "atr_ratio": atr_ratio,
                "percentile": None,
                "reason": "insufficient history for percentile calculation",
            }

        # Compute percentile rank
        percentile = self.compute_percentile(atr_ratio)

        # Check bounds
        if percentile > self.upper_percentile:
            reason = (
                f"volatility too high: ATR_Ratio={atr_ratio:.6f}, "
                f"percentile={percentile:.1f}, upper_bound={self.upper_percentile}"
            )
            log.info(f"🌡️ Volatility filter BLOCKED: {reason}")
            return False, {
                "atr_ratio": atr_ratio,
                "percentile": percentile,
                "reason": reason,
            }

        if percentile < self.lower_percentile:
            reason = (
                f"volatility too low: ATR_Ratio={atr_ratio:.6f}, "
                f"percentile={percentile:.1f}, lower_bound={self.lower_percentile}"
            )
            log.info(f"🌡️ Volatility filter BLOCKED: {reason}")
            return False, {
                "atr_ratio": atr_ratio,
                "percentile": percentile,
                "reason": reason,
            }

        # Within bounds — allow trading
        return True, {
            "atr_ratio": atr_ratio,
            "percentile": percentile,
            "reason": "volatility within acceptable range",
        }
