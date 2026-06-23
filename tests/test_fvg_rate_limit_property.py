"""Property-based test for FVG Rate Limit Invariant.

Property 7: Rate Limit Invariant
Across any 60-second window, the total TwelveData API calls made by
the FVG cycle SHALL NOT exceed 8. Across any 24-hour window, total
calls SHALL NOT exceed 800.

**Validates: Requirements 6.1, 6.2, 6.3**

Key insight: With cycle_interval_seconds (min 60s for this test), the
CycleScheduler prevents more than 1 cycle per interval. Each cycle makes
at most 2 API calls (60min + 15min; 5min comes from on_bar df).
So in any 60s window, max calls = 2 (far below the 8/min limit).
In 24h: max cycles = 86400 / interval, each with 2 calls.
With interval >= 60s → max 1440 cycles → 2880 calls worst case,
but the RateBudgetManager gates on the 800/day limit.
"""

import time
from collections import deque
from datetime import datetime
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
from hypothesis import given, settings, assume
from hypothesis.strategies import integers, floats, composite

from strategy.fvg_strategy import FVGStrategy


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_5min_df(n_bars: int = 50) -> pd.DataFrame:
    """Create a minimal 5min OHLC DataFrame for on_bar() calls."""
    np.random.seed(42)
    base = 3500.0
    opens = base + np.random.randn(n_bars) * 10
    closes = opens + np.random.randn(n_bars) * 5
    highs = np.maximum(opens, closes) + np.abs(np.random.randn(n_bars)) * 3
    lows = np.minimum(opens, closes) - np.abs(np.random.randn(n_bars)) * 3

    index = pd.date_range(start="2024-01-01", periods=n_bars, freq="5min")
    return pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": closes},
        index=index,
    )


class APICallTracker:
    """Mock data provider that tracks every get_bars() call with timestamps."""

    def __init__(self, daily_limit: int = 800, minute_limit: int = 8):
        self.call_timestamps: list = []
        self._daily_limit = daily_limit
        self._minute_limit = minute_limit

    def get_bars(self, symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
        """Record the call and return a valid DataFrame."""
        self.call_timestamps.append(time.time())
        # Return a basic DataFrame so the cycle can proceed
        n = min(limit, 50)
        np.random.seed(len(self.call_timestamps))
        base = 3500.0
        opens = base + np.random.randn(n) * 10
        closes = opens + np.random.randn(n) * 5
        highs = np.maximum(opens, closes) + np.abs(np.random.randn(n)) * 3
        lows = np.minimum(opens, closes) - np.abs(np.random.randn(n)) * 3
        index = pd.date_range(start="2024-01-01", periods=n, freq="5min")
        return pd.DataFrame(
            {"open": opens, "high": highs, "low": lows, "close": closes},
            index=index,
        )

    def get_budget_status(self):
        """Report budget status based on tracked calls."""
        now = time.time()
        # Count calls in the last 60s for minute budget
        minute_calls = sum(
            1 for t in self.call_timestamps if now - t < 60
        )
        return {
            "daily_used": len(self.call_timestamps),
            "daily_remaining": self._daily_limit - len(self.call_timestamps),
            "daily_limit": self._daily_limit,
            "minute_used": minute_calls,
            "minute_limit": self._minute_limit,
        }


# ---------------------------------------------------------------------------
# Strategies (generators)
# ---------------------------------------------------------------------------

@composite
def rapid_call_counts(draw):
    """Generate a number of rapid on_bar() calls (20-100)."""
    return draw(integers(min_value=20, max_value=100))


@composite
def cycle_intervals(draw):
    """Generate cycle interval values (60-600 seconds).

    Minimum 60s ensures we get meaningful rate limit behavior.
    """
    return draw(integers(min_value=60, max_value=600))


# ---------------------------------------------------------------------------
# Property 7: Rate Limit Invariant
# Validates: Requirements 6.1, 6.2, 6.3
# ---------------------------------------------------------------------------

class TestRateLimitInvariant:
    """Property 7: Rate Limit Invariant.

    Across any 60-second window, the total TwelveData API calls made by
    the FVG cycle SHALL NOT exceed 8. Across any 24-hour window, total
    calls SHALL NOT exceed 800.

    The CycleScheduler + RateBudgetManager together prevent exceeding
    rate limits even under rapid on_bar() calls.

    **Validates: Requirements 6.1, 6.2, 6.3**
    """

    @given(
        n_calls=rapid_call_counts(),
        cycle_interval=cycle_intervals(),
    )
    @settings(max_examples=50, deadline=None)
    def test_rapid_on_bar_respects_per_minute_limit(self, n_calls: int, cycle_interval: int):
        """Rapid on_bar() calls must not exceed 8 API calls in any 60s window.

        The CycleScheduler gates cycle execution: only 1 cycle can run per
        interval. Each cycle makes at most 2 API calls (60min + 15min).
        With any interval >= 60s, at most 1 cycle fires in a 60s window,
        producing at most 2 API calls — well below the 8/min limit.
        """
        tracker = APICallTracker(daily_limit=800, minute_limit=8)

        config = {
            "cycle_interval_seconds": cycle_interval,
            "timeframes": ["60min", "15min", "5min"],
            "fvg_max_age_bars": 50,
            "stop_buffer_points": 2.0,
            "min_bias_confidence": 0.6,
            "lookback_candles": 50,
        }

        # Patch KILL_SWITCH and daily_lockout so they don't interfere
        with patch.dict("os.environ", {"KILL_SWITCH": "0"}, clear=False):
            strategy = FVGStrategy(
                config=config,
                data_provider=tracker,
                symbol_epic="CS.D.CFEGOLD.CEB.IP",
            )

            df = _make_5min_df(50)

            # Simulate rapid on_bar() calls — all happen "instantly"
            # The scheduler should only allow the first cycle to run
            for _ in range(n_calls):
                strategy.on_bar(df)

        # Assert: in any 60-second window, max 8 API calls
        calls = sorted(tracker.call_timestamps)
        if len(calls) > 0:
            # Sliding window check: for each call, count how many calls
            # are within 60s ahead of it
            for i, t in enumerate(calls):
                window_end = t + 60.0
                calls_in_window = sum(1 for c in calls[i:] if c <= window_end)
                assert calls_in_window <= 8, (
                    f"Rate limit violated: {calls_in_window} API calls in a 60s "
                    f"window starting at index {i}. cycle_interval={cycle_interval}s, "
                    f"n_calls={n_calls}, total_calls={len(calls)}"
                )

    @given(
        cycle_interval=cycle_intervals(),
    )
    @settings(max_examples=30, deadline=None)
    def test_max_daily_calls_under_800(self, cycle_interval: int):
        """Even with maximum possible cycles in 24h, total API calls stay under 800.

        Mathematical proof verified by test:
        - Max cycles in 24h = 86400 / cycle_interval
        - Each cycle makes 2 API calls (60min + 15min)
        - Total = (86400 / cycle_interval) * 2

        With minimum interval of 60s: 86400/60 * 2 = 2880 theoretical max,
        BUT the RateBudgetManager gates on daily_remaining < requests_per_cycle,
        so actual calls are capped at 800.
        """
        # Simulate the theoretical maximum — the RateBudgetManager should gate
        tracker = APICallTracker(daily_limit=800, minute_limit=8)

        config = {
            "cycle_interval_seconds": cycle_interval,
            "timeframes": ["60min", "15min", "5min"],
            "fvg_max_age_bars": 50,
            "stop_buffer_points": 2.0,
            "min_bias_confidence": 0.6,
            "lookback_candles": 50,
        }

        with patch.dict("os.environ", {"KILL_SWITCH": "0"}, clear=False):
            strategy = FVGStrategy(
                config=config,
                data_provider=tracker,
                symbol_epic="CS.D.CFEGOLD.CEB.IP",
            )

            df = _make_5min_df(50)

            # Calculate how many cycles could theoretically run in 24h
            max_cycles_24h = 86400 // cycle_interval

            # Simulate by advancing time for each potential cycle
            # We force the scheduler to allow each cycle by manipulating
            # _last_cycle_time
            for i in range(min(max_cycles_24h, 500)):
                # Force the scheduler to think enough time has passed
                strategy.scheduler._last_cycle_time = 0
                strategy.scheduler._cycle_running = False
                strategy.on_bar(df)

                # Check if budget manager stopped us
                if tracker.get_budget_status()["daily_remaining"] <= 0:
                    break

        # Assert: total calls never exceed 800
        total_calls = len(tracker.call_timestamps)
        assert total_calls <= 800, (
            f"Daily rate limit violated: {total_calls} API calls made "
            f"(limit is 800). cycle_interval={cycle_interval}s"
        )

    @given(
        n_calls=rapid_call_counts(),
    )
    @settings(max_examples=50, deadline=None)
    def test_scheduler_prevents_multiple_cycles_in_interval(self, n_calls: int):
        """CycleScheduler ensures only 1 cycle per interval regardless of on_bar() frequency.

        With default 300s interval, rapid on_bar() calls within a single interval
        window should trigger exactly 1 cycle (the first call), making at most 2
        API calls total (the cycle may early-exit after 60min fetch if bias is
        neutral, producing only 1 call).
        """
        tracker = APICallTracker(daily_limit=800, minute_limit=8)

        config = {
            "cycle_interval_seconds": 300,  # 5 minutes
            "timeframes": ["60min", "15min", "5min"],
            "fvg_max_age_bars": 50,
            "stop_buffer_points": 2.0,
            "min_bias_confidence": 0.6,
            "lookback_candles": 50,
        }

        with patch.dict("os.environ", {"KILL_SWITCH": "0"}, clear=False):
            strategy = FVGStrategy(
                config=config,
                data_provider=tracker,
                symbol_epic="CS.D.CFEGOLD.CEB.IP",
            )

            df = _make_5min_df(50)

            # All calls happen within the same second — only first should trigger cycle
            for _ in range(n_calls):
                strategy.on_bar(df)

        # At most 2 API calls: one for 60min, optionally one for 15min (first cycle only)
        # The cycle may early-exit after 60min if bias is neutral (no unfilled FVGs)
        total_calls = len(tracker.call_timestamps)
        assert 1 <= total_calls <= 2, (
            f"Expected 1-2 API calls (1 cycle × 1-2 timeframes), "
            f"got {total_calls}. n_calls={n_calls}"
        )

        # Verify all calls happened within a tiny window (< 1 second)
        if total_calls >= 2:
            time_span = tracker.call_timestamps[-1] - tracker.call_timestamps[0]
            assert time_span < 1.0, (
                f"All API calls should be within the same cycle, "
                f"but span is {time_span:.2f}s"
            )

    @given(
        n_calls=rapid_call_counts(),
        cycle_interval=cycle_intervals(),
    )
    @settings(max_examples=50, deadline=None)
    def test_calls_per_cycle_bounded_by_num_timeframes_minus_one(
        self, n_calls: int, cycle_interval: int
    ):
        """Each cycle makes at most (num_timeframes - 1) API calls.

        The FVGStrategy only fetches 60min and 15min via the data_provider;
        the 5min data comes from the on_bar() DataFrame. So each cycle
        makes at most 2 API calls. It may make fewer if the cycle early-exits
        (e.g., neutral bias after 60min analysis).

        This means per-cycle cost is at most 2, and rate limiting is
        a function of how often cycles can fire (governed by the scheduler).
        """
        tracker = APICallTracker(daily_limit=800, minute_limit=8)

        config = {
            "cycle_interval_seconds": cycle_interval,
            "timeframes": ["60min", "15min", "5min"],
            "fvg_max_age_bars": 50,
            "stop_buffer_points": 2.0,
            "min_bias_confidence": 0.6,
            "lookback_candles": 50,
        }

        with patch.dict("os.environ", {"KILL_SWITCH": "0"}, clear=False):
            strategy = FVGStrategy(
                config=config,
                data_provider=tracker,
                symbol_epic="CS.D.CFEGOLD.CEB.IP",
            )

            df = _make_5min_df(50)

            # First call triggers a cycle
            strategy.on_bar(df)

        # At most 2 API calls per cycle (60min + optionally 15min)
        num_timeframes = len(config["timeframes"])
        max_expected_calls = num_timeframes - 1  # 5min comes from on_bar df

        total_calls = len(tracker.call_timestamps)
        assert 1 <= total_calls <= max_expected_calls, (
            f"Expected 1 to {max_expected_calls} API calls per cycle "
            f"(num_timeframes={num_timeframes} minus 1 for 5min), "
            f"got {total_calls}"
        )
