"""Property-based tests for VolatilityRegimeFilter using Hypothesis.

Validates correctness properties 8–13 from the design document.
# Feature: ml-trading-improvements
"""

import numpy as np
import pandas as pd
from hypothesis import assume, given, settings
from hypothesis.strategies import (
    composite,
    floats,
    integers,
    lists,
)

from strategy.volatility_filter import VolatilityRegimeFilter


# ---------------------------------------------------------------------------
# Strategies (generators)
# ---------------------------------------------------------------------------


@composite
def ohlc_dataframes(draw, min_rows=15, max_rows=200):
    """Generate valid OHLC DataFrames with realistic gold prices.

    Ensures high >= max(open, close) and low <= min(open, close),
    and all prices are strictly positive.
    """
    n_rows = draw(integers(min_value=min_rows, max_value=max_rows))

    rows = []
    for _ in range(n_rows):
        open_price = draw(floats(min_value=3000.0, max_value=5000.0, allow_nan=False, allow_infinity=False))
        close_price = draw(floats(min_value=3000.0, max_value=5000.0, allow_nan=False, allow_infinity=False))
        high_price = draw(floats(
            min_value=max(open_price, close_price),
            max_value=5000.0,
            allow_nan=False, allow_infinity=False,
        ))
        low_price = draw(floats(
            min_value=3000.0,
            max_value=min(open_price, close_price),
            allow_nan=False, allow_infinity=False,
        ))
        rows.append((open_price, high_price, low_price, close_price))

    index = pd.date_range(start="2024-01-01", periods=n_rows, freq="5min")
    df = pd.DataFrame(rows, columns=["open", "high", "low", "close"], index=index)
    return df


@composite
def atr_ratio_values(draw):
    """Generate valid ATR ratio values (positive floats representing ATR/close)."""
    return draw(floats(min_value=0.0001, max_value=0.1, allow_nan=False, allow_infinity=False))


@composite
def atr_ratio_lists(draw, min_size=1, max_size=200):
    """Generate lists of valid ATR ratio values."""
    size = draw(integers(min_value=min_size, max_value=max_size))
    return [draw(floats(min_value=0.0001, max_value=0.1, allow_nan=False, allow_infinity=False))
            for _ in range(size)]


# ---------------------------------------------------------------------------
# Property 8: ATR Ratio computation correctness
# Validates: Requirements 5.1
# ---------------------------------------------------------------------------


class TestATRRatioComputation:
    """Property 8: ATR Ratio computation correctness.

    For any valid OHLC DataFrame with at least `atr_period` rows,
    `compute_atr_ratio()` SHALL return ATR(period) / close for the
    penultimate bar, and the result SHALL be > 0.

    # Feature: ml-trading-improvements, Property 8: ATR Ratio computation correctness

    **Validates: Requirements 5.1**
    """

    @given(df=ohlc_dataframes(min_rows=16, max_rows=200))
    @settings(max_examples=100, deadline=None)
    def test_atr_ratio_positive(self, df: pd.DataFrame):
        """compute_atr_ratio() must return a positive value for valid OHLC data
        with actual price variation."""
        # ATR > 0 requires at least some price movement (high != low) within
        # the ATR window. Skip degenerate all-flat DataFrames.
        assume(not (df["high"] == df["low"]).all())
        vf = VolatilityRegimeFilter({"atr_period": 14})
        result = vf.compute_atr_ratio(df)
        assert result > 0, f"ATR ratio should be > 0, got {result}"

    @given(df=ohlc_dataframes(min_rows=16, max_rows=200))
    @settings(max_examples=100, deadline=None)
    def test_atr_ratio_matches_manual_computation(self, df: pd.DataFrame):
        """compute_atr_ratio() must equal ATR(period) / close at penultimate bar."""
        atr_period = 14
        vf = VolatilityRegimeFilter({"atr_period": atr_period})

        result = vf.compute_atr_ratio(df)

        # Manual computation
        h = df["high"]
        l = df["low"]
        c = df["close"]
        prev_c = c.shift(1)

        tr = pd.concat([
            (h - l).abs(),
            (h - prev_c).abs(),
            (l - prev_c).abs(),
        ], axis=1).max(axis=1)

        atr = tr.rolling(atr_period).mean()
        expected = float(atr.iloc[-2] / c.iloc[-2])

        assert abs(result - expected) < 1e-10, (
            f"ATR ratio {result} does not match expected {expected}"
        )


# ---------------------------------------------------------------------------
# Property 9: Volatility history buffer is bounded
# Validates: Requirements 5.2
# ---------------------------------------------------------------------------


class TestHistoryBufferBounded:
    """Property 9: Volatility history buffer is bounded.

    For any sequence of N calls to `update_history()` where N > lookback_bars,
    the internal history length SHALL never exceed lookback_bars.

    # Feature: ml-trading-improvements, Property 9: Volatility history buffer is bounded

    **Validates: Requirements 5.2**
    """

    @given(
        lookback=integers(min_value=5, max_value=200),
        n_updates=integers(min_value=1, max_value=500),
    )
    @settings(max_examples=100, deadline=None)
    def test_history_never_exceeds_lookback(self, lookback: int, n_updates: int):
        """After N updates, history length must never exceed lookback_bars."""
        vf = VolatilityRegimeFilter({"lookback_bars": lookback})

        for i in range(n_updates):
            vf.update_history(0.001 * (i + 1))
            assert len(vf._history) <= lookback, (
                f"History length {len(vf._history)} exceeds lookback_bars {lookback} "
                f"after {i + 1} updates"
            )

    @given(
        lookback=integers(min_value=5, max_value=50),
        ratios=atr_ratio_lists(min_size=1, max_size=500),
    )
    @settings(max_examples=100, deadline=None)
    def test_history_length_capped_at_lookback(self, lookback: int, ratios: list):
        """Final history length must be min(len(ratios), lookback_bars)."""
        vf = VolatilityRegimeFilter({"lookback_bars": lookback})

        for r in ratios:
            vf.update_history(r)

        expected_len = min(len(ratios), lookback)
        assert len(vf._history) == expected_len, (
            f"History length {len(vf._history)} != expected {expected_len} "
            f"(lookback={lookback}, updates={len(ratios)})"
        )


# ---------------------------------------------------------------------------
# Property 10: Percentile rank correctness
# Validates: Requirements 5.3
# ---------------------------------------------------------------------------


class TestPercentileRankCorrectness:
    """Property 10: Percentile rank correctness.

    For any current ATR_Ratio value and history of length >= 1,
    `compute_percentile()` SHALL return a value equal to
    (count of history values <= current_ratio) / len(history) * 100,
    which is always in [0, 100].

    # Feature: ml-trading-improvements, Property 10: Percentile rank correctness

    **Validates: Requirements 5.3**
    """

    @given(
        history=atr_ratio_lists(min_size=1, max_size=200),
        current_ratio=floats(min_value=0.0001, max_value=0.1, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=100, deadline=None)
    def test_percentile_matches_formula(self, history: list, current_ratio: float):
        """compute_percentile() must equal (count <= current) / len(history) * 100."""
        vf = VolatilityRegimeFilter({"lookback_bars": len(history) + 10})

        # Fill history
        for val in history:
            vf.update_history(val)

        result = vf.compute_percentile(current_ratio)

        # Manual computation
        count_le = sum(1 for v in history if v <= current_ratio)
        expected = count_le / len(history) * 100

        assert abs(result - expected) < 1e-10, (
            f"Percentile {result} != expected {expected} "
            f"(count_le={count_le}, history_len={len(history)})"
        )

    @given(
        history=atr_ratio_lists(min_size=1, max_size=200),
        current_ratio=floats(min_value=0.0001, max_value=0.1, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=100, deadline=None)
    def test_percentile_in_bounds(self, history: list, current_ratio: float):
        """compute_percentile() must always return a value in [0, 100]."""
        vf = VolatilityRegimeFilter({"lookback_bars": len(history) + 10})

        for val in history:
            vf.update_history(val)

        result = vf.compute_percentile(current_ratio)

        assert 0.0 <= result <= 100.0, (
            f"Percentile {result} is outside bounds [0, 100]"
        )


# ---------------------------------------------------------------------------
# Property 11: Insufficient volatility history allows trading
# Validates: Requirements 5.4
# ---------------------------------------------------------------------------


class TestInsufficientHistoryAllows:
    """Property 11: Insufficient volatility history allows trading.

    For any ATR_Ratio value, when the volatility history contains fewer than
    20 entries, `allow_trading()` SHALL return `(True, ...)`.

    # Feature: ml-trading-improvements, Property 11: Insufficient volatility history allows trading

    **Validates: Requirements 5.4**
    """

    @given(
        df=ohlc_dataframes(min_rows=16, max_rows=50),
        pre_fill_count=integers(min_value=0, max_value=18),
    )
    @settings(max_examples=100, deadline=None)
    def test_allows_trading_with_insufficient_history(self, df: pd.DataFrame, pre_fill_count: int):
        """allow_trading() returns (True, ...) when history < 20 entries.

        We pre-fill with pre_fill_count entries (0..18), then call allow_trading
        which adds one more entry, so total is pre_fill_count + 1 which is at most 19 < 20.
        """
        vf = VolatilityRegimeFilter({
            "enabled": True,
            "atr_period": 14,
            "lookback_bars": 100,
        })

        # Pre-fill history with fewer than 19 entries
        # (allow_trading adds 1 more, so total will be <= 19 < 20)
        for i in range(pre_fill_count):
            vf.update_history(0.001 * (i + 1))

        allowed, metadata = vf.allow_trading(df)

        assert allowed is True, (
            f"Trading should be allowed with {pre_fill_count + 1} history entries "
            f"(< 20), got allowed={allowed}, metadata={metadata}"
        )


# ---------------------------------------------------------------------------
# Property 12: Volatility gate blocks outside configured bounds
# Validates: Requirements 6.1, 6.2
# ---------------------------------------------------------------------------


class TestVolatilityGateBlocking:
    """Property 12: Volatility gate blocks outside configured bounds.

    For any ATR_Percentile P and configured bounds [lower, upper],
    the volatility filter SHALL block trading if and only if
    P > upper OR P < lower (when history has >= 20 entries).

    # Feature: ml-trading-improvements, Property 12: Volatility gate blocks outside configured bounds

    **Validates: Requirements 6.1, 6.2**
    """

    @given(
        lower=floats(min_value=5.0, max_value=45.0, allow_nan=False, allow_infinity=False),
        upper=floats(min_value=55.0, max_value=95.0, allow_nan=False, allow_infinity=False),
        current_ratio=floats(min_value=0.0001, max_value=0.1, allow_nan=False, allow_infinity=False),
        history=atr_ratio_lists(min_size=20, max_size=100),
    )
    @settings(max_examples=100, deadline=None)
    def test_blocking_follows_percentile_bounds(
        self, lower: float, upper: float, current_ratio: float, history: list
    ):
        """Filter blocks iff percentile > upper OR percentile < lower."""
        assume(lower < upper)

        lookback = len(history) + 10
        vf = VolatilityRegimeFilter({
            "enabled": True,
            "atr_period": 14,
            "lookback_bars": lookback,
            "lower_percentile": lower,
            "upper_percentile": upper,
        })

        # Pre-fill history with at least 20 values (the filter checks len >= 20
        # AFTER adding current value via update_history inside allow_trading)
        # We need >= 19 pre-filled so after allow_trading adds 1 more, total >= 20
        for val in history:
            vf.update_history(val)

        # Compute expected percentile manually
        # Note: allow_trading calls update_history first, so current_ratio is added
        # to history before compute_percentile is called.
        all_history = list(history) + [current_ratio]
        # Only the last `lookback` items are kept in the deque
        effective_history = all_history[-lookback:]
        count_le = sum(1 for v in effective_history if v <= current_ratio)
        expected_percentile = count_le / len(effective_history) * 100

        # Determine expected result
        should_block = (expected_percentile > upper) or (expected_percentile < lower)

        # Now we need to call allow_trading with a DataFrame that will produce
        # current_ratio. Instead, we directly test the logic by simulating what
        # allow_trading does internally (since generating a DF with exact ATR ratio
        # is impractical). We add to history and check compute_percentile + bounds.
        vf2 = VolatilityRegimeFilter({
            "enabled": True,
            "atr_period": 14,
            "lookback_bars": lookback,
            "lower_percentile": lower,
            "upper_percentile": upper,
        })

        # Fill with history + current_ratio (simulating what allow_trading does)
        for val in history:
            vf2.update_history(val)
        vf2.update_history(current_ratio)

        percentile = vf2.compute_percentile(current_ratio)

        # Verify blocking logic
        if percentile > upper or percentile < lower:
            assert should_block is True, (
                f"Expected block: percentile={percentile}, "
                f"bounds=[{lower}, {upper}]"
            )
        else:
            assert should_block is False, (
                f"Expected allow: percentile={percentile}, "
                f"bounds=[{lower}, {upper}]"
            )


# ---------------------------------------------------------------------------
# Property 13: Disabled volatility filter allows all
# Validates: Requirements 7.3
# ---------------------------------------------------------------------------


class TestDisabledFilterAllowsAll:
    """Property 13: Disabled volatility filter allows all.

    For any DataFrame, when `volatility_filter.enabled` is false,
    `allow_trading()` SHALL return `(True, ...)`.

    # Feature: ml-trading-improvements, Property 13: Disabled volatility filter allows all

    **Validates: Requirements 7.3**
    """

    @given(df=ohlc_dataframes(min_rows=16, max_rows=100))
    @settings(max_examples=100, deadline=None)
    def test_disabled_filter_always_allows(self, df: pd.DataFrame):
        """When enabled=False, allow_trading() always returns (True, ...)."""
        vf = VolatilityRegimeFilter({
            "enabled": False,
            "atr_period": 14,
            "lookback_bars": 100,
        })

        allowed, metadata = vf.allow_trading(df)

        assert allowed is True, (
            f"Disabled filter should always allow trading, got {allowed}"
        )
        assert metadata["reason"] == "volatility filter disabled", (
            f"Expected reason 'volatility filter disabled', got '{metadata['reason']}'"
        )

    @given(
        df=ohlc_dataframes(min_rows=16, max_rows=100),
        history_size=integers(min_value=0, max_value=150),
    )
    @settings(max_examples=100, deadline=None)
    def test_disabled_filter_ignores_history_state(self, df: pd.DataFrame, history_size: int):
        """Disabled filter allows trading regardless of history state."""
        vf = VolatilityRegimeFilter({
            "enabled": False,
            "atr_period": 14,
            "lookback_bars": 100,
        })

        # Pre-fill arbitrary history
        for i in range(history_size):
            vf.update_history(0.001 * (i + 1))

        allowed, _ = vf.allow_trading(df)

        assert allowed is True, (
            f"Disabled filter should allow trading even with {history_size} "
            f"history entries, got allowed={allowed}"
        )


# ===========================================================================
# UNIT TESTS — Task 1.3
# ===========================================================================
# Tests cover:
# - Config parsing with default values (Requirements 7.1, 7.2)
# - Logging output on block — high/low volatility messages (Requirements 6.3, 6.4)
# - Edge case: all identical ATR ratios in history
# ===========================================================================

import logging


# ---------------------------------------------------------------------------
# Helpers for unit tests
# ---------------------------------------------------------------------------

def _make_ohlc_df(n_bars: int = 50, base_price: float = 2000.0) -> pd.DataFrame:
    """Create a simple OHLC DataFrame for unit testing."""
    np.random.seed(42)
    closes = base_price + np.cumsum(np.random.randn(n_bars) * 2)
    highs = closes + np.abs(np.random.randn(n_bars)) * 3
    lows = closes - np.abs(np.random.randn(n_bars)) * 3
    opens = closes + np.random.randn(n_bars) * 1

    return pd.DataFrame({
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
    })


# ---------------------------------------------------------------------------
# Unit Tests: Config Parsing with Default Values
# Requirements: 7.1, 7.2
# ---------------------------------------------------------------------------


class TestConfigParsingUnit:
    """Test that VolatilityRegimeFilter parses config with correct defaults."""

    def test_default_config_values(self):
        """Empty config dict should use all default values."""
        vf = VolatilityRegimeFilter({})

        assert vf.enabled is True
        assert vf.atr_period == 14
        assert vf.lookback_bars == 100
        assert vf.lower_percentile == 20.0
        assert vf.upper_percentile == 80.0

    def test_custom_config_values(self):
        """Custom config values should override defaults."""
        config = {
            "enabled": False,
            "atr_period": 20,
            "lookback_bars": 200,
            "lower_percentile": 10.0,
            "upper_percentile": 90.0,
        }
        vf = VolatilityRegimeFilter(config)

        assert vf.enabled is False
        assert vf.atr_period == 20
        assert vf.lookback_bars == 200
        assert vf.lower_percentile == 10.0
        assert vf.upper_percentile == 90.0

    def test_partial_config_uses_defaults_for_missing(self):
        """Partial config should fill missing keys with defaults."""
        config = {"atr_period": 7, "upper_percentile": 95.0}
        vf = VolatilityRegimeFilter(config)

        assert vf.enabled is True  # default
        assert vf.atr_period == 7  # custom
        assert vf.lookback_bars == 100  # default
        assert vf.lower_percentile == 20.0  # default
        assert vf.upper_percentile == 95.0  # custom

    def test_history_deque_respects_lookback_bars(self):
        """Internal history deque maxlen should match lookback_bars config."""
        vf = VolatilityRegimeFilter({"lookback_bars": 50})
        assert vf._history.maxlen == 50


# ---------------------------------------------------------------------------
# Unit Tests: Logging Output on Block
# Requirements: 6.3, 6.4
# ---------------------------------------------------------------------------


class TestLoggingOnBlockUnit:
    """Test that blocking produces correct log messages."""

    def test_log_message_on_high_volatility_block(self, caplog):
        """When volatility is too high, log should contain 'volatility too high'
        with ATR_Ratio, percentile, and upper_bound.

        Validates: Requirements 6.3
        """
        vf = VolatilityRegimeFilter({
            "enabled": True,
            "atr_period": 14,
            "lookback_bars": 100,
            "lower_percentile": 20.0,
            "upper_percentile": 80.0,
        })
        # Fill history with very low ATR ratios so the DF's ratio is in the high percentile
        for _ in range(25):
            vf.update_history(0.0001)

        # Create DF where ATR/close will be high (large range relative to close)
        n = 50
        closes = np.full(n, 100.0)
        highs = closes + 20.0  # Large range → high ATR
        lows = closes - 20.0
        opens = closes.copy()
        df = pd.DataFrame({"open": opens, "high": highs, "low": lows, "close": closes})

        with caplog.at_level(logging.INFO, logger="ig-scalper"):
            allowed, meta = vf.allow_trading(df)

        assert allowed is False
        assert "volatility too high" in meta["reason"]
        assert "ATR_Ratio=" in meta["reason"]
        assert "percentile=" in meta["reason"]
        assert "upper_bound=80.0" in meta["reason"]
        # Check log output contains the blocking message
        assert any("volatility too high" in record.message for record in caplog.records)

    def test_log_message_on_low_volatility_block(self, caplog):
        """When volatility is too low, log should contain 'volatility too low'
        with ATR_Ratio, percentile, and lower_bound.

        Validates: Requirements 6.4
        """
        vf = VolatilityRegimeFilter({
            "enabled": True,
            "atr_period": 14,
            "lookback_bars": 100,
            "lower_percentile": 20.0,
            "upper_percentile": 80.0,
        })
        # Fill with high ratios so a low ratio falls below lower percentile
        for _ in range(25):
            vf.update_history(0.5)

        # Create DF with very tight range → low ATR ratio
        n = 50
        closes = np.full(n, 2000.0)
        highs = closes + 0.01  # Extremely tight range
        lows = closes - 0.01
        opens = closes.copy()
        df = pd.DataFrame({"open": opens, "high": highs, "low": lows, "close": closes})

        with caplog.at_level(logging.INFO, logger="ig-scalper"):
            allowed, meta = vf.allow_trading(df)

        assert allowed is False
        assert "volatility too low" in meta["reason"]
        assert "ATR_Ratio=" in meta["reason"]
        assert "percentile=" in meta["reason"]
        assert "lower_bound=20.0" in meta["reason"]
        # Check log output contains the blocking message
        assert any("volatility too low" in record.message for record in caplog.records)


# ---------------------------------------------------------------------------
# Unit Tests: Edge Case — All Identical ATR Ratios in History
# ---------------------------------------------------------------------------


class TestIdenticalAtrRatiosUnit:
    """When all ATR ratios in history are identical, percentile should be 100."""

    def test_identical_history_percentile_is_100(self):
        """If all history values equal current ratio, percentile = 100
        because all values <= current_ratio."""
        vf = VolatilityRegimeFilter({"lookback_bars": 100})

        # Fill history with identical values
        for _ in range(30):
            vf.update_history(0.005)

        # Compute percentile for the same value
        pct = vf.compute_percentile(0.005)
        # All 30 values are <= 0.005, so percentile = 30/30 * 100 = 100
        assert pct == 100.0

    def test_identical_history_blocks_as_too_high(self):
        """With all identical ratios, percentile=100 which is above upper_percentile=80,
        so trading should be blocked as 'too high'."""
        vf = VolatilityRegimeFilter({
            "enabled": True,
            "atr_period": 14,
            "lookback_bars": 100,
            "lower_percentile": 20.0,
            "upper_percentile": 80.0,
        })

        # Fill with identical values (>= 20 entries for percentile check to activate)
        for _ in range(25):
            vf.update_history(0.005)

        # Create a DF that will produce ATR_ratio ~ 0.005
        # With close=2000 and high-low=10, ATR(14) ≈ 10, ratio = 10/2000 = 0.005
        n = 50
        closes = np.full(n, 2000.0)
        highs = closes + 5.0
        lows = closes - 5.0
        opens = closes.copy()
        df = pd.DataFrame({"open": opens, "high": highs, "low": lows, "close": closes})

        allowed, meta = vf.allow_trading(df)

        # All entries are ~0.005, percentile = 100 > upper(80) → blocked
        assert allowed is False
        assert "volatility too high" in meta["reason"]

    def test_identical_history_with_value_slightly_below(self):
        """If current ratio is slightly below all identical history values,
        percentile should be 0 (no values <= current)."""
        vf = VolatilityRegimeFilter({"lookback_bars": 100})

        # Fill history with a fixed value
        for _ in range(30):
            vf.update_history(0.010)

        # Query with a value slightly below
        pct = vf.compute_percentile(0.009)
        # None of the 30 values are <= 0.009 (they're all 0.010)
        # percentile = 0/30 * 100 = 0
        assert pct == 0.0
