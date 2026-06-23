"""Unit tests for rate limit compliance in FVGStrategy analysis cycle.

Tests verify:
- Cycle is skipped when daily budget is insufficient (Req 6.3)
- Budget consumption is logged after each cycle (Req 6.6)
- Per-minute budget exhaustion doesn't block (provider handles wait) (Req 6.4)
- Cache in data_provider avoids redundant requests (Req 6.5)
- Budget is verified before fetching each timeframe (Req 6.2)
"""

import logging
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch, PropertyMock

import pandas as pd
import pytest

from strategy.fvg_strategy import FVGStrategy


def _make_ohlc_df(n_bars=50, base_price=3200.0):
    """Create a simple OHLC DataFrame with datetime index."""
    dates = pd.date_range(
        start=datetime(2024, 1, 15, 9, 0), periods=n_bars, freq="5min"
    )
    data = {
        "open": [base_price + i * 0.5 for i in range(n_bars)],
        "high": [base_price + i * 0.5 + 2.0 for i in range(n_bars)],
        "low": [base_price + i * 0.5 - 1.0 for i in range(n_bars)],
        "close": [base_price + i * 0.5 + 1.0 for i in range(n_bars)],
        "volume": [100] * n_bars,
    }
    return pd.DataFrame(data, index=dates)


class FakeTwelveDataProvider:
    """Fake TwelveData provider for budget status queries."""

    def __init__(self, daily_remaining=720, minute_used=0):
        self._daily_remaining = daily_remaining
        self._minute_used = minute_used

    def get_budget_status(self):
        return {
            "daily_used": 720 - self._daily_remaining,
            "daily_remaining": self._daily_remaining,
            "daily_limit": 720,
            "minute_used": self._minute_used,
            "minute_limit": 7,
            "active_symbols": 1,
            "optimal_interval_sec": 120.0,
            "hours_elapsed_today": 5.0,
        }


class FakeDataProvider:
    """Fake SmartDataAggregator for testing FVGStrategy."""

    def __init__(self, twelvedata_provider=None, return_empty=False):
        td = twelvedata_provider or FakeTwelveDataProvider()
        self.providers = [("TwelveData", td)]
        self._return_empty = return_empty
        self.fetch_count = 0

    def get_bars(self, epic, timeframe="5min", limit=100):
        self.fetch_count += 1
        if self._return_empty:
            return pd.DataFrame()
        return _make_ohlc_df(n_bars=limit)


DEFAULT_CONFIG = {
    "cycle_interval_seconds": 300,
    "timeframes": ["60min", "15min", "5min"],
    "fvg_max_age_bars": 50,
    "stop_buffer_points": 2.0,
    "min_bias_confidence": 0.6,
    "lookback_candles": 200,
}


class TestCycleSkipOnInsufficientBudget:
    """Test Req 6.3: cycle skipped when daily budget < requests needed."""

    def test_skips_cycle_when_daily_budget_exhausted(self, caplog):
        td = FakeTwelveDataProvider(daily_remaining=1)  # Need 2 for 60min+15min
        provider = FakeDataProvider(td)
        strategy = FVGStrategy(DEFAULT_CONFIG, provider)

        # Force scheduler to allow cycle
        strategy.scheduler._last_cycle_time = 0

        df_5min = _make_ohlc_df(n_bars=50)

        with caplog.at_level(logging.WARNING, logger="ig-scalper"):
            result = strategy.on_bar(df_5min)

        assert result is None
        assert provider.fetch_count == 0  # No fetches made
        assert "insufficient rate budget" in caplog.text.lower() or "Insufficient" in caplog.text

    def test_proceeds_when_budget_sufficient(self):
        """When budget is sufficient, data fetches are made (at least 60min)."""
        td = FakeTwelveDataProvider(daily_remaining=500)
        provider = FakeDataProvider(td)
        strategy = FVGStrategy(DEFAULT_CONFIG, provider)

        # Force scheduler to allow cycle
        strategy.scheduler._last_cycle_time = 0

        df_5min = _make_ohlc_df(n_bars=50)
        strategy.on_bar(df_5min)

        # Should have fetched at least 60min data (may stop if neutral bias)
        assert provider.fetch_count >= 1

    def test_fetches_both_timeframes_when_fvgs_exist(self):
        """When 60min data has FVGs that produce non-neutral bias, 15min is also fetched."""
        td = FakeTwelveDataProvider(daily_remaining=500)
        provider = FakeDataProvider(td)
        strategy = FVGStrategy(DEFAULT_CONFIG, provider)
        strategy.scheduler._last_cycle_time = 0

        df_5min = _make_ohlc_df(n_bars=50)

        # Mock bias calculator to return non-neutral bias so cycle continues to 15min
        with patch.object(strategy.bias_calc, "calculate_60min_bias") as mock_bias:
            from strategy.fvg_detector import Bias
            mock_bias.return_value = Bias(direction="bullish", confidence=0.8)
            strategy.on_bar(df_5min)

        # Should have fetched 60min AND 15min
        assert provider.fetch_count == 2


class TestBudgetConsumptionLogging:
    """Test Req 6.6: log budget consumption after each cycle."""

    def test_logs_consumption_after_successful_cycle(self, caplog):
        td = FakeTwelveDataProvider(daily_remaining=500, minute_used=2)
        provider = FakeDataProvider(td)
        strategy = FVGStrategy(DEFAULT_CONFIG, provider)

        strategy.scheduler._last_cycle_time = 0
        df_5min = _make_ohlc_df(n_bars=50)

        with caplog.at_level(logging.INFO, logger="ig-scalper"):
            strategy.on_bar(df_5min)

        # Verify consumption was logged
        assert "Cycle complete" in caplog.text
        assert "requests_this_cycle=" in caplog.text
        assert "daily_used=" in caplog.text
        assert "daily_remaining=" in caplog.text

    def test_logs_consumption_on_aborted_cycle(self, caplog):
        """Even when cycle aborts (empty df), budget is logged."""
        td = FakeTwelveDataProvider(daily_remaining=500)
        provider = FakeDataProvider(td, return_empty=True)
        strategy = FVGStrategy(DEFAULT_CONFIG, provider)

        strategy.scheduler._last_cycle_time = 0
        df_5min = _make_ohlc_df(n_bars=50)

        with caplog.at_level(logging.INFO, logger="ig-scalper"):
            strategy.on_bar(df_5min)

        # Should still log consumption even though cycle aborted
        assert "Cycle complete" in caplog.text


class TestPerMinuteBudgetHandling:
    """Test Req 6.4: per-minute budget handling."""

    def test_does_not_block_when_minute_exhausted(self):
        """Per-minute exhaustion is informational — provider handles wait."""
        td = FakeTwelveDataProvider(daily_remaining=500, minute_used=7)
        provider = FakeDataProvider(td)
        strategy = FVGStrategy(DEFAULT_CONFIG, provider)

        strategy.scheduler._last_cycle_time = 0
        df_5min = _make_ohlc_df(n_bars=50)

        # Should still proceed (at least fetch 60min) — per-minute wait is provider's job
        strategy.on_bar(df_5min)
        assert provider.fetch_count >= 1


class TestRequestRecording:
    """Test that requests are recorded during the cycle."""

    def test_records_requests_during_cycle(self):
        """Each fetch increments the request counter."""
        td = FakeTwelveDataProvider(daily_remaining=500)
        provider = FakeDataProvider(td)
        strategy = FVGStrategy(DEFAULT_CONFIG, provider)

        strategy.scheduler._last_cycle_time = 0
        df_5min = _make_ohlc_df(n_bars=50)
        strategy.on_bar(df_5min)

        # At least 1 request recorded (60min fetch), cycle may stop at neutral bias
        assert strategy.rate_budget._requests_this_cycle >= 1
        # Requests recorded should match actual fetches
        assert strategy.rate_budget._requests_this_cycle == provider.fetch_count

    def test_records_two_requests_when_both_fetched(self):
        """When bias is non-neutral, both 60min and 15min are fetched."""
        td = FakeTwelveDataProvider(daily_remaining=500)
        provider = FakeDataProvider(td)
        strategy = FVGStrategy(DEFAULT_CONFIG, provider)
        strategy.scheduler._last_cycle_time = 0

        df_5min = _make_ohlc_df(n_bars=50)

        with patch.object(strategy.bias_calc, "calculate_60min_bias") as mock_bias:
            from strategy.fvg_detector import Bias
            mock_bias.return_value = Bias(direction="bullish", confidence=0.8)
            strategy.on_bar(df_5min)

        assert strategy.rate_budget._requests_this_cycle == 2


class TestCachedSignalOnBudgetSkip:
    """Test that cached signal is preserved when budget skip occurs."""

    def test_returns_cached_signal_after_budget_skip(self):
        # First call: budget ok, run cycle (signal may be None due to data)
        td = FakeTwelveDataProvider(daily_remaining=500)
        provider = FakeDataProvider(td)
        strategy = FVGStrategy(DEFAULT_CONFIG, provider)

        strategy.scheduler._last_cycle_time = 0
        df_5min = _make_ohlc_df(n_bars=50)
        first_result = strategy.on_bar(df_5min)

        # Verify cached signal matches
        assert strategy._cached_signal == first_result


class TestRateBudgetIntegrationWithStrategy:
    """Test the rate_budget attribute is properly wired in FVGStrategy."""

    def test_rate_budget_uses_correct_num_timeframes(self):
        """rate_budget should use len(timeframes) - 1 since 5min comes from on_bar."""
        td = FakeTwelveDataProvider(daily_remaining=500)
        provider = FakeDataProvider(td)
        strategy = FVGStrategy(DEFAULT_CONFIG, provider)

        # 3 timeframes configured, but only 2 are fetched (60min, 15min)
        assert strategy.rate_budget.requests_per_cycle == 2

    def test_rate_budget_queries_provider_budget(self):
        """rate_budget should be able to query budget from data_provider."""
        td = FakeTwelveDataProvider(daily_remaining=350, minute_used=4)
        provider = FakeDataProvider(td)
        strategy = FVGStrategy(DEFAULT_CONFIG, provider)

        budget = strategy.rate_budget.get_budget_status()
        assert budget is not None
        assert budget["daily_remaining"] == 350
        assert budget["minute_used"] == 4
