"""Unit tests for the FVG Rate Budget Manager.

Tests verify:
- Budget calculation (requests per cycle)
- Daily budget check (sufficient/insufficient)
- Per-minute budget check
- Cycle gate logic (should_proceed_with_cycle)
- Consumption logging
- Integration with TwelveDataProvider budget status
"""

import logging
from unittest.mock import MagicMock, patch

import pytest

from strategy.fvg_rate_budget import RateBudgetManager


class FakeTwelveDataProvider:
    """Fake TwelveData provider for testing budget queries."""

    def __init__(self, daily_used=0, daily_remaining=720, minute_used=0):
        self._daily_used = daily_used
        self._daily_remaining = daily_remaining
        self._minute_used = minute_used

    def get_budget_status(self):
        return {
            "daily_used": self._daily_used,
            "daily_remaining": self._daily_remaining,
            "daily_limit": 720,
            "minute_used": self._minute_used,
            "minute_limit": 7,
            "active_symbols": 1,
            "optimal_interval_sec": 120.0,
            "hours_elapsed_today": 5.0,
        }


class FakeSmartDataAggregator:
    """Fake SmartDataAggregator wrapping a TwelveData provider."""

    def __init__(self, twelvedata_provider):
        self.providers = [("TwelveData", twelvedata_provider)]


class TestRequestsPerCycle:
    """Test Req 6.1: requests per cycle = timeframes × symbols."""

    def test_default_3_timeframes_1_symbol(self):
        provider = FakeTwelveDataProvider()
        mgr = RateBudgetManager(provider, num_timeframes=3, num_symbols=1)
        assert mgr.requests_per_cycle == 3

    def test_custom_timeframes_and_symbols(self):
        provider = FakeTwelveDataProvider()
        mgr = RateBudgetManager(provider, num_timeframes=4, num_symbols=2)
        assert mgr.requests_per_cycle == 8

    def test_single_timeframe_single_symbol(self):
        provider = FakeTwelveDataProvider()
        mgr = RateBudgetManager(provider, num_timeframes=1, num_symbols=1)
        assert mgr.requests_per_cycle == 1


class TestDailyBudgetCheck:
    """Test Req 6.2, 6.3: daily budget verification."""

    def test_sufficient_budget(self):
        provider = FakeTwelveDataProvider(daily_used=100, daily_remaining=620)
        mgr = RateBudgetManager(provider, num_timeframes=3, num_symbols=1)
        assert mgr.has_sufficient_daily_budget() is True

    def test_insufficient_budget_exactly_at_limit(self):
        # Remaining is 2, need 3 → insufficient
        provider = FakeTwelveDataProvider(daily_used=718, daily_remaining=2)
        mgr = RateBudgetManager(provider, num_timeframes=3, num_symbols=1)
        assert mgr.has_sufficient_daily_budget() is False

    def test_insufficient_budget_zero_remaining(self):
        provider = FakeTwelveDataProvider(daily_used=720, daily_remaining=0)
        mgr = RateBudgetManager(provider, num_timeframes=3, num_symbols=1)
        assert mgr.has_sufficient_daily_budget() is False

    def test_sufficient_budget_exactly_needed(self):
        # Remaining is exactly 3, need 3 → sufficient
        provider = FakeTwelveDataProvider(daily_used=717, daily_remaining=3)
        mgr = RateBudgetManager(provider, num_timeframes=3, num_symbols=1)
        assert mgr.has_sufficient_daily_budget() is True

    def test_no_provider_returns_true(self):
        """If no TwelveData provider found, assume OK (fallback providers)."""
        mgr = RateBudgetManager(object(), num_timeframes=3, num_symbols=1)
        assert mgr.has_sufficient_daily_budget() is True


class TestMinuteBudgetCheck:
    """Test Req 6.4: per-minute budget check."""

    def test_minute_budget_available(self):
        provider = FakeTwelveDataProvider(minute_used=3)
        mgr = RateBudgetManager(provider, num_timeframes=3, num_symbols=1)
        assert mgr.has_sufficient_minute_budget() is True

    def test_minute_budget_exhausted(self):
        provider = FakeTwelveDataProvider(minute_used=7)
        mgr = RateBudgetManager(provider, num_timeframes=3, num_symbols=1)
        assert mgr.has_sufficient_minute_budget() is False

    def test_minute_budget_over_limit(self):
        provider = FakeTwelveDataProvider(minute_used=10)
        mgr = RateBudgetManager(provider, num_timeframes=3, num_symbols=1)
        assert mgr.has_sufficient_minute_budget() is False

    def test_no_provider_returns_true(self):
        mgr = RateBudgetManager(object(), num_timeframes=3, num_symbols=1)
        assert mgr.has_sufficient_minute_budget() is True


class TestShouldProceedWithCycle:
    """Test pre-cycle gate: combine daily + minute checks."""

    def test_proceed_when_all_budgets_ok(self):
        provider = FakeTwelveDataProvider(daily_remaining=100, minute_used=2)
        mgr = RateBudgetManager(provider, num_timeframes=3, num_symbols=1)
        assert mgr.should_proceed_with_cycle() is True

    def test_skip_when_daily_budget_insufficient(self):
        provider = FakeTwelveDataProvider(daily_remaining=1, minute_used=0)
        mgr = RateBudgetManager(provider, num_timeframes=3, num_symbols=1)
        assert mgr.should_proceed_with_cycle() is False

    def test_proceed_even_when_minute_exhausted(self):
        """Per-minute exhaustion doesn't block — provider handles wait internally."""
        provider = FakeTwelveDataProvider(daily_remaining=100, minute_used=7)
        mgr = RateBudgetManager(provider, num_timeframes=3, num_symbols=1)
        # Still proceeds because per-minute is informational only
        assert mgr.should_proceed_with_cycle() is True


class TestSmartDataAggregatorIntegration:
    """Test budget queries through SmartDataAggregator wrapper."""

    def test_extracts_twelvedata_from_aggregator(self):
        td_provider = FakeTwelveDataProvider(daily_remaining=500)
        aggregator = FakeSmartDataAggregator(td_provider)
        mgr = RateBudgetManager(aggregator, num_timeframes=3, num_symbols=1)

        budget = mgr.get_budget_status()
        assert budget is not None
        assert budget["daily_remaining"] == 500

    def test_no_twelvedata_in_aggregator(self):
        """Aggregator without TwelveData provider."""

        class NoTwelveAggregator:
            providers = [("YahooFinance", MagicMock())]

        mgr = RateBudgetManager(NoTwelveAggregator(), num_timeframes=3, num_symbols=1)
        assert mgr.get_budget_status() is None


class TestCycleConsumptionTracking:
    """Test request recording and consumption logging."""

    def test_record_cycle_start_resets_counter(self):
        provider = FakeTwelveDataProvider()
        mgr = RateBudgetManager(provider)
        mgr.record_request()
        mgr.record_request()
        assert mgr._requests_this_cycle == 2

        mgr.record_cycle_start()
        assert mgr._requests_this_cycle == 0

    def test_record_request_increments(self):
        provider = FakeTwelveDataProvider()
        mgr = RateBudgetManager(provider)
        mgr.record_cycle_start()
        mgr.record_request()
        mgr.record_request()
        mgr.record_request()
        assert mgr._requests_this_cycle == 3

    def test_log_cycle_consumption(self, caplog):
        """Verify consumption logging includes required fields (Req 6.6)."""
        provider = FakeTwelveDataProvider(daily_used=103, daily_remaining=617)
        mgr = RateBudgetManager(provider)
        mgr.record_cycle_start()
        mgr.record_request()
        mgr.record_request()

        with caplog.at_level(logging.INFO, logger="ig-scalper"):
            mgr.log_cycle_consumption()

        assert "requests_this_cycle=2" in caplog.text
        assert "daily_used=103" in caplog.text
        assert "daily_remaining=617" in caplog.text

    def test_log_consumption_no_provider(self, caplog):
        """When no TwelveData available, still logs cycle count."""
        mgr = RateBudgetManager(object())
        mgr.record_cycle_start()
        mgr.record_request()

        with caplog.at_level(logging.INFO, logger="ig-scalper"):
            mgr.log_cycle_consumption()

        assert "requests_this_cycle=1" in caplog.text
        assert "unavailable" in caplog.text
