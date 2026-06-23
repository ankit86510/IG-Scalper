"""
Unit tests for FVGStrategy.

Tests cover:
- Configuration validation and defaults (Req 9.1-9.4)
- Strategy ABC compliance (Req 7.1)
- on_bar() cycle triggering and caching (Req 7.2, 7.5)
- Analysis cycle orchestration (Req 7.2)
"""

import time
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from strategy.fvg_strategy import FVGStrategy, _resolve_config, _validate_config


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_ohlc_df(rows: int = 20, base_price: float = 3000.0) -> pd.DataFrame:
    """Create a synthetic OHLC DataFrame with no FVG patterns (flat movement)."""
    dates = pd.date_range("2024-01-01 09:00", periods=rows, freq="5min")
    data = []
    price = base_price
    for i in range(rows):
        o = price
        h = price + 1.0
        l = price - 1.0
        c = price + 0.5
        data.append({"open": o, "high": h, "low": l, "close": c, "volume": 100})
        price += 0.3
    return pd.DataFrame(data, index=dates)


def _make_fvg_df(rows: int = 20) -> pd.DataFrame:
    """Create an OHLC DataFrame with a bullish FVG near the end that won't be filled.

    Places the FVG at rows -5, -4, -3 (near the end) so only a few bars
    follow it, and those bars have prices well below the FVG zone to avoid fill.
    FVG zone: (zone_lower=base_high_c1, zone_upper=c3_low) where subsequent
    bars stay below zone_lower.
    """
    dates = pd.date_range("2024-01-01 09:00", periods=rows, freq="5min")
    data = []
    fvg_idx = rows - 5  # Place FVG near end

    for i in range(rows):
        if i == fvg_idx:
            # Candle i: high=3020
            data.append({"open": 3015, "high": 3020, "low": 3014, "close": 3018, "volume": 100})
        elif i == fvg_idx + 1:
            # Candle i+1 (middle): big gap up, but its high stays at zone boundary
            data.append({"open": 3025, "high": 3028, "low": 3024, "close": 3026, "volume": 150})
        elif i == fvg_idx + 2:
            # Candle i+2: low=3022 > candle_i high=3020 → bullish FVG zone (3020, 3022)
            # its high stays below zone upper to avoid self-fill during update
            data.append({"open": 3023, "high": 3025, "low": 3022, "close": 3024, "volume": 120})
        else:
            # All other bars: keep high well below 3020 (zone_lower)
            base = 3000 + (i % 10) * 0.3
            data.append({"open": base, "high": base + 1.0, "low": base - 1.0, "close": base + 0.5, "volume": 100})
    return pd.DataFrame(data, index=dates)


def _valid_config() -> dict:
    """Return a valid FVG strategy config."""
    return {
        "cycle_interval_seconds": 300,
        "timeframes": ["60min", "15min", "5min"],
        "fvg_max_age_bars": 50,
        "stop_buffer_points": 2.0,
        "min_bias_confidence": 0.6,
        "lookback_candles": 200,
    }


def _mock_data_provider(df_60min=None, df_15min=None):
    """Create a mock data provider that returns specified DataFrames.

    Uses spec= to restrict attributes so RateBudgetManager's
    _get_twelvedata_provider() doesn't mistakenly find get_budget_status.
    """
    if df_60min is None:
        df_60min = _make_fvg_df(rows=30)
    if df_15min is None:
        df_15min = _make_fvg_df(rows=30)

    class MockDataProvider:
        """Minimal mock that only exposes get_bars and providers."""

        def __init__(self):
            self.providers = []  # Empty: no TwelveData found by RateBudgetManager
            self.call_count = 0

        def get_bars(self, symbol, timeframe, limit):
            self.call_count += 1
            if "60" in timeframe:
                return df_60min
            elif "15" in timeframe:
                return df_15min
            return pd.DataFrame()

    provider = MockDataProvider()
    # Wrap get_bars in a MagicMock to track calls while keeping the behavior
    original_get_bars = provider.get_bars
    provider.get_bars = MagicMock(side_effect=original_get_bars)
    return provider


# ---------------------------------------------------------------------------
# Config Resolution Tests (Req 9.3)
# ---------------------------------------------------------------------------


class TestConfigResolveDefaults:
    """Tests for _resolve_config: filling missing keys with defaults."""

    def test_all_keys_present_no_warning(self, caplog):
        """No warnings when all config keys are present."""
        import logging
        with caplog.at_level(logging.WARNING, logger="ig-scalper"):
            result = _resolve_config(_valid_config())
        assert result == _valid_config()
        assert not any("missing" in msg for msg in caplog.messages)

    def test_missing_key_uses_default(self, caplog):
        """Missing key should be filled with default and warning logged."""
        import logging
        config = {"cycle_interval_seconds": 300}
        with caplog.at_level(logging.WARNING, logger="ig-scalper"):
            result = _resolve_config(config)
        assert result["fvg_max_age_bars"] == 50
        assert result["timeframes"] == ["60min", "15min", "5min"]
        assert any("fvg_max_age_bars" in msg for msg in caplog.messages)

    def test_empty_config_all_defaults(self, caplog):
        """Empty config should result in all defaults."""
        import logging
        with caplog.at_level(logging.WARNING, logger="ig-scalper"):
            result = _resolve_config({})
        assert result["cycle_interval_seconds"] == 300
        assert result["timeframes"] == ["60min", "15min", "5min"]
        assert result["fvg_max_age_bars"] == 50
        assert result["stop_buffer_points"] == 2.0
        assert result["min_bias_confidence"] == 0.6
        assert result["lookback_candles"] == 200


# ---------------------------------------------------------------------------
# Config Validation Tests (Req 9.4)
# ---------------------------------------------------------------------------


class TestConfigValidation:
    """Tests for _validate_config: raising ValueError for invalid values."""

    def test_valid_config_passes(self):
        """Valid config should not raise."""
        _validate_config(_valid_config())

    def test_negative_cycle_interval_raises(self):
        """Negative cycle_interval_seconds should raise ValueError."""
        config = _valid_config()
        config["cycle_interval_seconds"] = -10
        with pytest.raises(ValueError, match="cycle_interval_seconds must be > 0"):
            _validate_config(config)

    def test_zero_cycle_interval_raises(self):
        """Zero cycle_interval_seconds should raise ValueError."""
        config = _valid_config()
        config["cycle_interval_seconds"] = 0
        with pytest.raises(ValueError, match="cycle_interval_seconds must be > 0"):
            _validate_config(config)

    def test_empty_timeframes_raises(self):
        """Empty timeframes list should raise ValueError."""
        config = _valid_config()
        config["timeframes"] = []
        with pytest.raises(ValueError, match="timeframes must be a non-empty list"):
            _validate_config(config)

    def test_non_list_timeframes_raises(self):
        """Non-list timeframes should raise ValueError."""
        config = _valid_config()
        config["timeframes"] = "60min"
        with pytest.raises(ValueError, match="timeframes must be a non-empty list"):
            _validate_config(config)

    def test_negative_max_age_raises(self):
        """Negative fvg_max_age_bars should raise ValueError."""
        config = _valid_config()
        config["fvg_max_age_bars"] = -1
        with pytest.raises(ValueError, match="fvg_max_age_bars must be > 0"):
            _validate_config(config)

    def test_negative_stop_buffer_raises(self):
        """Negative stop_buffer_points should raise ValueError."""
        config = _valid_config()
        config["stop_buffer_points"] = -0.5
        with pytest.raises(ValueError, match="stop_buffer_points must be >= 0"):
            _validate_config(config)

    def test_zero_stop_buffer_valid(self):
        """Zero stop_buffer_points should be valid."""
        config = _valid_config()
        config["stop_buffer_points"] = 0
        _validate_config(config)  # No exception

    def test_confidence_above_1_raises(self):
        """min_bias_confidence > 1 should raise ValueError."""
        config = _valid_config()
        config["min_bias_confidence"] = 1.5
        with pytest.raises(ValueError, match="min_bias_confidence must be between 0 and 1"):
            _validate_config(config)

    def test_confidence_below_0_raises(self):
        """min_bias_confidence < 0 should raise ValueError."""
        config = _valid_config()
        config["min_bias_confidence"] = -0.1
        with pytest.raises(ValueError, match="min_bias_confidence must be between 0 and 1"):
            _validate_config(config)

    def test_negative_lookback_raises(self):
        """Negative lookback_candles should raise ValueError."""
        config = _valid_config()
        config["lookback_candles"] = 0
        with pytest.raises(ValueError, match="lookback_candles must be > 0"):
            _validate_config(config)


# ---------------------------------------------------------------------------
# FVGStrategy Initialization Tests (Req 7.1)
# ---------------------------------------------------------------------------


class TestFVGStrategyInit:
    """Tests for FVGStrategy initialization."""

    def test_init_with_valid_config(self):
        """Strategy initializes correctly with valid config."""
        provider = _mock_data_provider()
        strategy = FVGStrategy(config=_valid_config(), data_provider=provider)
        assert strategy.config["cycle_interval_seconds"] == 300
        assert strategy.data_provider is provider
        assert strategy._cached_signal is None

    def test_init_with_partial_config_uses_defaults(self):
        """Strategy fills missing config keys with defaults."""
        provider = _mock_data_provider()
        config = {"cycle_interval_seconds": 120}
        strategy = FVGStrategy(config=config, data_provider=provider)
        assert strategy.config["cycle_interval_seconds"] == 120
        assert strategy.config["fvg_max_age_bars"] == 50

    def test_init_with_invalid_config_raises(self):
        """Strategy raises ValueError for invalid config."""
        provider = _mock_data_provider()
        config = {"cycle_interval_seconds": -5}
        with pytest.raises(ValueError):
            FVGStrategy(config=config, data_provider=provider)

    def test_extends_strategy_abc(self):
        """FVGStrategy should be a subclass of Strategy."""
        from strategy.base import Strategy
        assert issubclass(FVGStrategy, Strategy)

    def test_components_initialized(self):
        """All sub-components should be initialized."""
        provider = _mock_data_provider()
        strategy = FVGStrategy(config=_valid_config(), data_provider=provider)
        assert strategy.detector is not None
        assert strategy.bias_calc is not None
        assert strategy.signal_gen is not None
        assert strategy.scheduler is not None
        assert strategy.rate_budget is not None


# ---------------------------------------------------------------------------
# on_bar() Behavior Tests (Req 7.2, 7.5)
# ---------------------------------------------------------------------------


class TestFVGStrategyOnBar:
    """Tests for on_bar() method: cycle triggering and caching."""

    def test_first_call_triggers_cycle(self):
        """First on_bar() call should trigger analysis cycle."""
        provider = _mock_data_provider()
        strategy = FVGStrategy(config=_valid_config(), data_provider=provider)
        df = _make_ohlc_df(rows=30)

        strategy.on_bar(df)
        # Should have called data_provider.get_bars at least once (60min)
        assert provider.get_bars.call_count >= 1

    def test_second_call_within_interval_returns_cached(self):
        """Second on_bar() within interval should return cached signal (Req 7.5)."""
        provider = _mock_data_provider()
        strategy = FVGStrategy(config=_valid_config(), data_provider=provider)
        df = _make_ohlc_df(rows=30)

        # First call triggers cycle
        first_result = strategy.on_bar(df)
        call_count_after_first = provider.get_bars.call_count

        # Second call should not trigger another cycle
        second_result = strategy.on_bar(df)
        assert provider.get_bars.call_count == call_count_after_first
        assert second_result == first_result

    def test_returns_none_when_no_signal(self):
        """on_bar() should return None when no signal is generated."""
        # Use flat data with no FVGs to ensure no signal
        flat_df = _make_ohlc_df(rows=30, base_price=3000.0)
        provider = _mock_data_provider(df_60min=flat_df, df_15min=flat_df)
        strategy = FVGStrategy(config=_valid_config(), data_provider=provider)

        result = strategy.on_bar(flat_df)
        # With flat data (no real FVGs) the bias will be neutral → None
        assert result is None

    def test_cycle_runs_again_after_interval(self):
        """After interval elapses, on_bar() should trigger a new cycle."""
        provider = _mock_data_provider()
        config = _valid_config()
        config["cycle_interval_seconds"] = 1  # Short interval for testing
        strategy = FVGStrategy(config=config, data_provider=provider)
        df = _make_ohlc_df(rows=30)

        # First call
        strategy.on_bar(df)
        first_call_count = provider.get_bars.call_count

        # Simulate time passing
        strategy.scheduler._last_cycle_time = time.time() - 2

        # Second call should trigger new cycle
        strategy.on_bar(df)
        assert provider.get_bars.call_count > first_call_count

    def test_cached_signal_returned_on_repeated_calls(self):
        """Repeated on_bar() calls return the same cached result (Req 7.5)."""
        flat_df = _make_ohlc_df(rows=30)
        provider = _mock_data_provider(df_60min=flat_df, df_15min=flat_df)
        strategy = FVGStrategy(config=_valid_config(), data_provider=provider)

        # First call sets cache
        r1 = strategy.on_bar(flat_df)
        # Subsequent calls should return same cached value
        r2 = strategy.on_bar(flat_df)
        r3 = strategy.on_bar(flat_df)
        assert r1 == r2 == r3


# ---------------------------------------------------------------------------
# Analysis Cycle Tests (Req 7.2)
# ---------------------------------------------------------------------------


class TestFVGStrategyAnalysisCycle:
    """Tests for _run_analysis_cycle orchestration."""

    def test_aborts_on_empty_60min_data(self):
        """Cycle should abort and return None if 60min data is empty."""
        provider = _mock_data_provider(
            df_60min=pd.DataFrame(), df_15min=pd.DataFrame()
        )
        strategy = FVGStrategy(config=_valid_config(), data_provider=provider)

        result = strategy._run_analysis_cycle(_make_ohlc_df(rows=30))
        assert result is None

    def test_aborts_on_empty_15min_data(self):
        """Cycle should abort and return None if 15min data is empty."""
        provider = _mock_data_provider(
            df_60min=_make_fvg_df(rows=30), df_15min=pd.DataFrame()
        )
        strategy = FVGStrategy(config=_valid_config(), data_provider=provider)

        result = strategy._run_analysis_cycle(_make_ohlc_df(rows=30))
        assert result is None

    def test_fetches_correct_timeframes(self):
        """Cycle should fetch 60min and 15min data from provider."""
        provider = _mock_data_provider()
        strategy = FVGStrategy(config=_valid_config(), data_provider=provider)
        df = _make_ohlc_df(rows=30)

        # Patch bias calculator to return non-neutral bias so cycle proceeds to 15min
        with patch.object(strategy.bias_calc, "calculate_60min_bias") as mock_bias:
            from strategy.fvg_detector import Bias
            mock_bias.return_value = Bias(direction="bullish", confidence=0.8)
            strategy._run_analysis_cycle(df)

        # Check that get_bars was called with correct timeframes
        calls = provider.get_bars.call_args_list
        timeframes_called = [call[0][1] for call in calls]
        assert "60min" in timeframes_called
        assert "15min" in timeframes_called

    def test_uses_configured_lookback(self):
        """Cycle should pass lookback_candles as limit to get_bars."""
        provider = _mock_data_provider()
        config = _valid_config()
        config["lookback_candles"] = 150
        strategy = FVGStrategy(config=config, data_provider=provider)
        df = _make_ohlc_df(rows=30)

        strategy._run_analysis_cycle(df)

        # Check that get_bars was called with correct limit
        calls = provider.get_bars.call_args_list
        limits_called = [call[0][2] for call in calls]
        assert all(limit == 150 for limit in limits_called)

    def test_returns_none_on_neutral_bias(self):
        """Cycle should return None when 60min bias is neutral (no FVGs)."""
        # Flat data → no FVGs → neutral bias
        flat_df = _make_ohlc_df(rows=30)
        provider = _mock_data_provider(df_60min=flat_df, df_15min=flat_df)
        strategy = FVGStrategy(config=_valid_config(), data_provider=provider)

        result = strategy._run_analysis_cycle(flat_df)
        assert result is None

    def test_exception_in_cycle_returns_none(self):
        """Exception during cycle should be caught and None returned via on_bar."""
        provider = _mock_data_provider()
        provider.get_bars = MagicMock(side_effect=Exception("API error"))
        strategy = FVGStrategy(config=_valid_config(), data_provider=provider)
        df = _make_ohlc_df(rows=30)

        # on_bar catches exceptions in cycle
        result = strategy.on_bar(df)
        assert result is None
        # Scheduler should be marked complete despite error
        assert strategy.scheduler._cycle_running is False

    def test_uses_symbol_epic_from_init(self):
        """Cycle should use the symbol_epic passed at initialization."""
        provider = _mock_data_provider()
        strategy = FVGStrategy(
            config=_valid_config(),
            data_provider=provider,
            symbol_epic="IX.D.SPTRD.DAILY.IP",
        )
        df = _make_ohlc_df(rows=30)

        strategy._run_analysis_cycle(df)

        # Check that get_bars was called with correct symbol
        calls = provider.get_bars.call_args_list
        symbols_called = [call[0][0] for call in calls]
        assert all(s == "IX.D.SPTRD.DAILY.IP" for s in symbols_called)
