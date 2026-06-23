"""
Integration tests for FVG Multi-Timeframe Strategy full analysis cycle.

Tests cover:
- Complete 60min → 15min → 5min cascade producing a correct signal (Req 3.1, 3.8)
- Empty DataFrame abort behavior (Req 3.7)
- Cache behavior: second on_bar within interval returns cached signal (Req 5.2, 7.5)
- Cycle skip on rate limit exhaustion (Req 6.3)

These tests mock the DataProvider to return controlled DataFrames and patch
the FVGDetector to return controlled FVG lists, allowing end-to-end cascade
testing without depending on the fill-tracking behavior of formation bars.
"""

import time
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from strategy.fvg_detector import FVG, Bias, FVGDetector
from strategy.fvg_strategy import FVGStrategy

ROME_TZ = ZoneInfo("Europe/Rome")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_ohlc_df(
    timeframe: str = "5min", rows: int = 30, base_price: float = 3000.0
) -> pd.DataFrame:
    """Create a generic OHLC DataFrame."""
    freq = timeframe
    dates = pd.date_range(
        "2024-01-10 09:00", periods=rows, freq=freq, tz=ROME_TZ
    )
    data = []
    for i in range(rows):
        o = base_price + i * 0.5
        h = o + 2
        l = o - 1
        c = o + 1
        data.append({"open": o, "high": h, "low": l, "close": c, "volume": 100})
    return pd.DataFrame(data, index=dates)


def _make_bullish_fvg_list(timeframe: str, count: int = 3) -> list:
    """Create unfilled bullish FVGs with zones wide enough for favorable R:R.

    Zone size 25pts with stop_buffer 2.0 → stop=27, needs TP>27.
    HTF targets above entry provide TP distance > stop.
    """
    fvgs = []
    base_ts = datetime(2024, 1, 10, 10, 0, tzinfo=ROME_TZ)
    for i in range(count):
        fvgs.append(
            FVG(
                type="bullish",
                zone_upper=3070.0 + i * 30,
                zone_lower=3045.0 + i * 30,
                formation_ts=base_ts + timedelta(hours=i),
                source_tf=timeframe,
                fill_status="unfilled",
                age_bars=5 + i,
            )
        )
    return fvgs


def _make_bullish_htf_fvg_list() -> list:
    """Create HTF FVGs that provide TP targets above the 5min entry.

    Includes a bearish HTF FVG with zone_lower above the 5min trigger entry,
    giving the SignalGenerator a valid TP target for favorable R:R.
    """
    base_ts = datetime(2024, 1, 10, 10, 0, tzinfo=ROME_TZ)
    return [
        FVG(type="bullish", zone_upper=3100.0, zone_lower=3080.0,
            formation_ts=base_ts, source_tf="60min", fill_status="unfilled", age_bars=5),
        # This bearish HTF FVG provides a TP target above entry
        FVG(type="bearish", zone_upper=3180.0, zone_lower=3160.0,
            formation_ts=base_ts + timedelta(hours=1), source_tf="60min",
            fill_status="unfilled", age_bars=6),
    ]


def _make_bearish_fvg_list(timeframe: str, count: int = 3) -> list:
    """Create unfilled bearish FVGs with zones wide enough for favorable R:R."""
    fvgs = []
    base_ts = datetime(2024, 1, 10, 10, 0, tzinfo=ROME_TZ)
    for i in range(count):
        fvgs.append(
            FVG(
                type="bearish",
                zone_upper=3060.0 - i * 30,
                zone_lower=3030.0 - i * 30,
                formation_ts=base_ts + timedelta(hours=i),
                source_tf=timeframe,
                fill_status="unfilled",
                age_bars=5 + i,
            )
        )
    return fvgs


def _make_bearish_htf_fvg_list() -> list:
    """Create HTF FVGs that provide TP targets below the 5min entry (SELL)."""
    base_ts = datetime(2024, 1, 10, 10, 0, tzinfo=ROME_TZ)
    return [
        FVG(type="bearish", zone_upper=3090.0, zone_lower=3070.0,
            formation_ts=base_ts, source_tf="60min", fill_status="unfilled", age_bars=5),
        # Bullish HTF FVG below entry provides TP target for SELL
        FVG(type="bullish", zone_upper=2920.0, zone_lower=2900.0,
            formation_ts=base_ts + timedelta(hours=1), source_tf="60min",
            fill_status="unfilled", age_bars=6),
    ]


def _valid_config() -> dict:
    """Return a valid FVG strategy config for testing."""
    return {
        "cycle_interval_seconds": 300,
        "timeframes": ["60min", "15min", "5min"],
        "fvg_max_age_bars": 50,
        "stop_buffer_points": 2.0,
        "min_bias_confidence": 0.4,
        "lookback_candles": 200,
    }


class MockDataProvider:
    """Mock data provider returning controlled DataFrames per timeframe."""

    def __init__(self, frames: dict):
        self._frames = frames
        self.providers = []
        self.call_log = []

    def get_bars(self, symbol, timeframe, limit):
        self.call_log.append((symbol, timeframe, limit))
        for key, df in self._frames.items():
            if key in timeframe:
                return df
        return pd.DataFrame()


class MockDataProviderWithBudget(MockDataProvider):
    """Mock data provider with get_budget_status for rate limit tests."""

    def __init__(self, frames: dict, budget_status: dict):
        super().__init__(frames)
        self._budget_status = budget_status

    def get_budget_status(self):
        return self._budget_status


def _setup_bullish_strategy(strategy):
    """Patch strategy detector to produce bullish FVGs across all timeframes.

    The detect() returns bullish FVGs for all timeframes, and
    update_fill_status() returns them unchanged (unfilled).
    Also provides HTF targets that make the R:R favorable.
    """
    htf_fvgs = _make_bullish_htf_fvg_list()

    def mock_detect(df, timeframe):
        if "60" in timeframe:
            # Return bullish majority + HTF target for TP
            return _make_bullish_fvg_list(timeframe, count=3) + [
                FVG(type="bearish", zone_upper=3180.0, zone_lower=3160.0,
                    formation_ts=datetime(2024, 1, 10, 12, 0, tzinfo=ROME_TZ),
                    source_tf=timeframe, fill_status="unfilled", age_bars=3),
            ]
        elif "15" in timeframe:
            return _make_bullish_fvg_list(timeframe, count=3)
        else:
            return _make_bullish_fvg_list(timeframe, count=3)

    def mock_update_fill(fvgs, df, max_age):
        return fvgs

    strategy.detector.detect = mock_detect
    strategy.detector.update_fill_status = mock_update_fill


def _setup_bearish_strategy(strategy):
    """Patch strategy detector to produce bearish FVGs across all timeframes."""

    def mock_detect(df, timeframe):
        if "60" in timeframe:
            return _make_bearish_fvg_list(timeframe, count=3) + [
                FVG(type="bullish", zone_upper=2920.0, zone_lower=2900.0,
                    formation_ts=datetime(2024, 1, 10, 12, 0, tzinfo=ROME_TZ),
                    source_tf=timeframe, fill_status="unfilled", age_bars=3),
            ]
        elif "15" in timeframe:
            return _make_bearish_fvg_list(timeframe, count=3)
        else:
            return _make_bearish_fvg_list(timeframe, count=3)

    def mock_update_fill(fvgs, df, max_age):
        return fvgs

    strategy.detector.detect = mock_detect
    strategy.detector.update_fill_status = mock_update_fill


# ---------------------------------------------------------------------------
# Test: Full 60min → 15min → 5min cascade producing correct BUY signal
# ---------------------------------------------------------------------------


class TestFullCascadeBuySignal:
    """Integration test for complete cascade producing a BUY signal.

    Validates Requirements: 3.1, 3.8, 7.2
    - 60min: bullish FVGs dominate → bullish bias
    - 15min: bullish FVGs confirm → confidence boosted
    - 5min: bullish FVG present → BUY signal generated
    """

    def test_cascade_produces_buy_signal(self):
        """Full cascade with aligned bullish FVGs produces a BUY signal."""
        df_60min = _make_ohlc_df("60min", rows=30)
        df_15min = _make_ohlc_df("15min", rows=30)
        df_5min = _make_ohlc_df("5min", rows=30)

        provider = MockDataProvider({"60": df_60min, "15": df_15min})
        strategy = FVGStrategy(config=_valid_config(), data_provider=provider)
        _setup_bullish_strategy(strategy)

        signal = strategy.on_bar(df_5min)

        assert signal is not None, "Expected a BUY signal from aligned bullish cascade"
        assert signal["side"] == "BUY"
        assert signal["stop_pts"] > 0
        assert signal["tp_pts"] > 0
        assert signal["tp_pts"] > signal["stop_pts"], "TP must exceed stop (R:R > 1)"
        assert "meta" in signal
        assert signal["meta"]["bias_direction"] == "bullish"
        assert signal["meta"]["bias_confidence"] > 0

    def test_cascade_signal_has_required_meta_fields(self):
        """Signal meta field contains all required fields (Req 4.9)."""
        df_60min = _make_ohlc_df("60min", rows=30)
        df_15min = _make_ohlc_df("15min", rows=30)
        df_5min = _make_ohlc_df("5min", rows=30)

        provider = MockDataProvider({"60": df_60min, "15": df_15min})
        strategy = FVGStrategy(config=_valid_config(), data_provider=provider)
        _setup_bullish_strategy(strategy)

        signal = strategy.on_bar(df_5min)

        assert signal is not None
        meta = signal["meta"]
        assert "bias_direction" in meta
        assert "bias_confidence" in meta
        assert "trigger_fvg" in meta
        assert "fvgs_60min" in meta
        assert "fvgs_15min" in meta
        assert "fvgs_5min" in meta
        assert "entry_zone" in meta
        assert meta["trigger_fvg"]["source_tf"] == "5min"

    def test_cascade_fetches_all_timeframes_in_order(self):
        """Cascade should fetch 60min first, then 15min (Req 3.1)."""
        df_60min = _make_ohlc_df("60min", rows=30)
        df_15min = _make_ohlc_df("15min", rows=30)
        df_5min = _make_ohlc_df("5min", rows=30)

        provider = MockDataProvider({"60": df_60min, "15": df_15min})
        strategy = FVGStrategy(config=_valid_config(), data_provider=provider)
        _setup_bullish_strategy(strategy)

        strategy.on_bar(df_5min)

        # Verify fetch order: 60min first, then 15min
        assert len(provider.call_log) >= 2
        assert "60" in provider.call_log[0][1]
        assert "15" in provider.call_log[1][1]

    def test_confidence_boosted_by_confirming_15min(self):
        """15min bullish majority boosts confidence by 0.2 (Req 3.4)."""
        df_60min = _make_ohlc_df("60min", rows=30)
        df_15min = _make_ohlc_df("15min", rows=30)
        df_5min = _make_ohlc_df("5min", rows=30)

        provider = MockDataProvider({"60": df_60min, "15": df_15min})
        strategy = FVGStrategy(config=_valid_config(), data_provider=provider)
        _setup_bullish_strategy(strategy)

        signal = strategy.on_bar(df_5min)

        assert signal is not None
        # 60min: 3 bullish + 1 bearish → confidence = abs(3-1)/(3+1) = 0.5
        # 15min confirms (3 bullish): +0.2 → 0.7
        assert signal["meta"]["bias_confidence"] == pytest.approx(0.7, abs=0.01)


# ---------------------------------------------------------------------------
# Test: Full cascade producing SELL signal
# ---------------------------------------------------------------------------


class TestFullCascadeSellSignal:
    """Integration test for complete cascade producing a SELL signal.

    Validates Requirements: 3.1, 3.8
    """

    def test_cascade_produces_sell_signal(self):
        """Full cascade with aligned bearish FVGs produces a SELL signal."""
        df_60min = _make_ohlc_df("60min", rows=30)
        df_15min = _make_ohlc_df("15min", rows=30)
        df_5min = _make_ohlc_df("5min", rows=30)

        provider = MockDataProvider({"60": df_60min, "15": df_15min})
        strategy = FVGStrategy(config=_valid_config(), data_provider=provider)
        _setup_bearish_strategy(strategy)

        signal = strategy.on_bar(df_5min)

        assert signal is not None, "Expected a SELL signal from aligned bearish cascade"
        assert signal["side"] == "SELL"
        assert signal["stop_pts"] > 0
        assert signal["tp_pts"] > 0
        assert signal["tp_pts"] > signal["stop_pts"]
        assert signal["meta"]["bias_direction"] == "bearish"


# ---------------------------------------------------------------------------
# Test: Cascade aborts on empty DataFrame (Req 3.7)
# ---------------------------------------------------------------------------


class TestCascadeAbortOnEmptyData:
    """Integration tests for cycle abort when timeframe returns empty data.

    Validates Requirement 3.7.
    """

    def test_aborts_when_60min_empty(self):
        """Cycle aborts and returns None when 60min data is empty."""
        df_5min = _make_ohlc_df("5min", rows=30)
        provider = MockDataProvider(
            {"60": pd.DataFrame(), "15": _make_ohlc_df("15min")}
        )
        strategy = FVGStrategy(config=_valid_config(), data_provider=provider)

        signal = strategy.on_bar(df_5min)
        assert signal is None

    def test_aborts_when_15min_empty(self):
        """Cycle aborts and returns None when 15min data is empty."""
        df_5min = _make_ohlc_df("5min", rows=30)
        provider = MockDataProvider(
            {"60": _make_ohlc_df("60min"), "15": pd.DataFrame()}
        )
        strategy = FVGStrategy(config=_valid_config(), data_provider=provider)
        _setup_bullish_strategy(strategy)

        signal = strategy.on_bar(df_5min)
        assert signal is None

    def test_aborts_when_60min_returns_none(self):
        """Cycle aborts when get_bars returns None for 60min."""
        df_5min = _make_ohlc_df("5min", rows=30)
        provider = MockDataProvider({"15": _make_ohlc_df("15min")})
        original_get_bars = provider.get_bars

        def custom_get_bars(symbol, timeframe, limit):
            if "60" in timeframe:
                return None
            return original_get_bars(symbol, timeframe, limit)

        provider.get_bars = custom_get_bars
        strategy = FVGStrategy(config=_valid_config(), data_provider=provider)

        signal = strategy.on_bar(df_5min)
        assert signal is None


# ---------------------------------------------------------------------------
# Test: Cache behavior — second on_bar within interval returns cached signal
# ---------------------------------------------------------------------------


class TestCacheBehavior:
    """Integration tests for signal caching between cycles.

    Validates Requirements: 5.2, 7.5
    """

    def test_second_on_bar_returns_cached_signal(self):
        """Second on_bar within interval returns same cached signal (Req 7.5)."""
        df_60min = _make_ohlc_df("60min", rows=30)
        df_15min = _make_ohlc_df("15min", rows=30)
        df_5min = _make_ohlc_df("5min", rows=30)

        provider = MockDataProvider({"60": df_60min, "15": df_15min})
        strategy = FVGStrategy(config=_valid_config(), data_provider=provider)
        _setup_bullish_strategy(strategy)

        # First call triggers cycle
        first_signal = strategy.on_bar(df_5min)
        calls_after_first = len(provider.call_log)

        # Second call within interval → cached, no new fetches
        second_signal = strategy.on_bar(df_5min)
        calls_after_second = len(provider.call_log)

        assert second_signal == first_signal
        assert calls_after_second == calls_after_first, (
            "No additional data fetches should occur on cached return"
        )

    def test_multiple_rapid_calls_all_cached(self):
        """Multiple rapid calls all return same cached value (Req 5.2)."""
        df_60min = _make_ohlc_df("60min", rows=30)
        df_15min = _make_ohlc_df("15min", rows=30)
        df_5min = _make_ohlc_df("5min", rows=30)

        provider = MockDataProvider({"60": df_60min, "15": df_15min})
        strategy = FVGStrategy(config=_valid_config(), data_provider=provider)
        _setup_bullish_strategy(strategy)

        first = strategy.on_bar(df_5min)
        second = strategy.on_bar(df_5min)
        third = strategy.on_bar(df_5min)
        fourth = strategy.on_bar(df_5min)

        assert first == second == third == fourth
        # Only 2 fetches from the first cycle (60min + 15min)
        assert len(provider.call_log) == 2

    def test_cache_returns_none_when_no_signal(self):
        """Cache returns None between cycles when no signal generated."""
        flat_df = _make_ohlc_df("5min", rows=30)
        provider = MockDataProvider(
            {"60": _make_ohlc_df("60min"), "15": _make_ohlc_df("15min")}
        )
        strategy = FVGStrategy(config=_valid_config(), data_provider=provider)
        # No patching → natural detection → neutral bias → None

        first = strategy.on_bar(flat_df)
        assert first is None

        second = strategy.on_bar(flat_df)
        assert second is None

    def test_new_cycle_runs_after_interval_elapses(self):
        """After interval elapses, a new cycle runs (Req 5.2)."""
        df_60min = _make_ohlc_df("60min", rows=30)
        df_15min = _make_ohlc_df("15min", rows=30)
        df_5min = _make_ohlc_df("5min", rows=30)

        config = _valid_config()
        config["cycle_interval_seconds"] = 1

        provider = MockDataProvider({"60": df_60min, "15": df_15min})
        strategy = FVGStrategy(config=config, data_provider=provider)
        _setup_bullish_strategy(strategy)

        strategy.on_bar(df_5min)
        calls_after_first = len(provider.call_log)

        # Force interval to elapse
        strategy.scheduler._last_cycle_time = time.time() - 2

        strategy.on_bar(df_5min)
        calls_after_second = len(provider.call_log)

        assert calls_after_second > calls_after_first

    def test_cached_signal_persists_across_multiple_calls(self):
        """A generated signal stays cached until the next cycle runs."""
        df_60min = _make_ohlc_df("60min", rows=30)
        df_15min = _make_ohlc_df("15min", rows=30)
        df_5min = _make_ohlc_df("5min", rows=30)

        provider = MockDataProvider({"60": df_60min, "15": df_15min})
        strategy = FVGStrategy(config=_valid_config(), data_provider=provider)
        _setup_bullish_strategy(strategy)

        signal = strategy.on_bar(df_5min)
        assert signal is not None
        assert signal["side"] == "BUY"

        # 5 more calls — all should return same cached signal
        for _ in range(5):
            cached = strategy.on_bar(df_5min)
            assert cached == signal


# ---------------------------------------------------------------------------
# Test: Cycle skip on rate limit exhaustion (Req 6.3)
# ---------------------------------------------------------------------------


class TestCycleSkipOnRateLimitExhaustion:
    """Integration tests for cycle skip when rate budget is exhausted.

    Validates Requirement 6.3.
    """

    def test_cycle_skipped_when_daily_budget_exhausted(self):
        """Cycle is skipped when daily budget < requests needed."""
        df_5min = _make_ohlc_df("5min", rows=30)

        budget_status = {
            "daily_used": 799,
            "daily_remaining": 1,
            "daily_limit": 800,
            "minute_used": 0,
            "minute_limit": 7,
        }
        provider = MockDataProviderWithBudget(
            frames={"60": _make_ohlc_df("60min"), "15": _make_ohlc_df("15min")},
            budget_status=budget_status,
        )
        strategy = FVGStrategy(config=_valid_config(), data_provider=provider)

        signal = strategy.on_bar(df_5min)

        assert signal is None
        assert len(provider.call_log) == 0

    def test_cycle_proceeds_when_budget_sufficient(self):
        """Cycle proceeds normally when sufficient budget remains."""
        df_60min = _make_ohlc_df("60min", rows=30)
        df_15min = _make_ohlc_df("15min", rows=30)
        df_5min = _make_ohlc_df("5min", rows=30)

        budget_status = {
            "daily_used": 100,
            "daily_remaining": 700,
            "daily_limit": 800,
            "minute_used": 2,
            "minute_limit": 7,
        }
        provider = MockDataProviderWithBudget(
            frames={"60": df_60min, "15": df_15min},
            budget_status=budget_status,
        )
        strategy = FVGStrategy(config=_valid_config(), data_provider=provider)
        _setup_bullish_strategy(strategy)

        signal = strategy.on_bar(df_5min)

        assert len(provider.call_log) >= 2
        assert signal is not None

    def test_cycle_skipped_returns_cached_none(self):
        """When cycle skipped, None is returned (no prior cache)."""
        df_5min = _make_ohlc_df("5min", rows=30)

        budget_status = {
            "daily_used": 800,
            "daily_remaining": 0,
            "daily_limit": 800,
            "minute_used": 7,
            "minute_limit": 7,
        }
        provider = MockDataProviderWithBudget(
            frames={"60": _make_ohlc_df("60min"), "15": _make_ohlc_df("15min")},
            budget_status=budget_status,
        )
        strategy = FVGStrategy(config=_valid_config(), data_provider=provider)

        signal = strategy.on_bar(df_5min)
        assert signal is None
        assert strategy.scheduler._cycle_running is False

    def test_budget_exhaustion_after_successful_cycle(self):
        """After a successful cycle, budget exhaustion skips data fetching."""
        df_60min = _make_ohlc_df("60min", rows=30)
        df_15min = _make_ohlc_df("15min", rows=30)
        df_5min = _make_ohlc_df("5min", rows=30)

        budget_status = {
            "daily_used": 100,
            "daily_remaining": 700,
            "daily_limit": 800,
            "minute_used": 0,
            "minute_limit": 7,
        }
        provider = MockDataProviderWithBudget(
            frames={"60": df_60min, "15": df_15min},
            budget_status=budget_status,
        )

        config = _valid_config()
        config["cycle_interval_seconds"] = 1
        strategy = FVGStrategy(config=config, data_provider=provider)
        _setup_bullish_strategy(strategy)

        # First cycle succeeds
        first_signal = strategy.on_bar(df_5min)
        first_call_count = len(provider.call_log)
        assert first_signal is not None
        assert first_call_count >= 2

        # Simulate budget exhaustion
        provider._budget_status = {
            "daily_used": 799,
            "daily_remaining": 1,
            "daily_limit": 800,
            "minute_used": 5,
            "minute_limit": 7,
        }

        # Force interval to elapse
        strategy.scheduler._last_cycle_time = time.time() - 2

        # Second cycle: rate budget check fails inside _run_analysis_cycle
        second_signal = strategy.on_bar(df_5min)

        # No new data fetches occurred (cycle aborted at budget check)
        assert len(provider.call_log) == first_call_count
        # Signal is None because cycle was skipped (cache cleared by new cycle)
        assert second_signal is None


# ---------------------------------------------------------------------------
# Test: Neutral bias produces no signal
# ---------------------------------------------------------------------------


class TestNeutralBiasNoSignal:
    """Integration test: neutral bias → no signal (Req 3.6)."""

    def test_no_fvgs_produces_neutral_bias_no_signal(self):
        """No FVGs → neutral bias → no signal generated."""
        df_60min = _make_ohlc_df("60min", rows=30)
        df_15min = _make_ohlc_df("15min", rows=30)
        df_5min = _make_ohlc_df("5min", rows=30)

        provider = MockDataProvider({"60": df_60min, "15": df_15min})
        strategy = FVGStrategy(config=_valid_config(), data_provider=provider)

        # Patch detector to return empty lists
        strategy.detector.detect = lambda df, tf: []
        strategy.detector.update_fill_status = lambda fvgs, df, max_age: fvgs

        signal = strategy.on_bar(df_5min)
        assert signal is None

    def test_equal_bull_bear_count_produces_neutral_bias(self):
        """Equal bullish and bearish FVG counts → neutral → no signal."""
        df_60min = _make_ohlc_df("60min", rows=30)
        df_15min = _make_ohlc_df("15min", rows=30)
        df_5min = _make_ohlc_df("5min", rows=30)

        provider = MockDataProvider({"60": df_60min, "15": df_15min})
        strategy = FVGStrategy(config=_valid_config(), data_provider=provider)

        base_ts = datetime(2024, 1, 10, 10, 0, tzinfo=ROME_TZ)

        def mock_detect(df, timeframe):
            return [
                FVG(type="bullish", zone_upper=3050, zone_lower=3040,
                    formation_ts=base_ts, source_tf=timeframe,
                    fill_status="unfilled"),
                FVG(type="bearish", zone_upper=3080, zone_lower=3070,
                    formation_ts=base_ts, source_tf=timeframe,
                    fill_status="unfilled"),
            ]

        strategy.detector.detect = mock_detect
        strategy.detector.update_fill_status = lambda fvgs, df, max_age: fvgs

        signal = strategy.on_bar(df_5min)
        assert signal is None
