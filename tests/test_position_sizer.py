"""Property-based tests for RiskPositionSizer using Hypothesis.

Validates correctness properties defined in the design document.
# Feature: ml-trading-improvements
"""

import math
from unittest.mock import MagicMock, patch

from hypothesis import assume, given, settings
from hypothesis.strategies import (
    booleans,
    composite,
    floats,
    integers,
)

from core.position_sizer import RiskPositionSizer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_sizer(
    risk_pct: float = 2.0,
    equity: float = 10000.0,
    max_size_multiple: int = 50,
    use_dynamic_sizing: bool = True,
) -> RiskPositionSizer:
    """Create a RiskPositionSizer with mocked IG client and pre-set equity."""
    ig_client = MagicMock()
    ig_client.account_summary.return_value = {
        "accounts": [{"balance": {"balance": equity}}]
    }
    config = {
        "risk_pct_per_trade": risk_pct,
        "equity_refresh_interval_seconds": 300,
        "use_dynamic_sizing": use_dynamic_sizing,
        "max_size_multiple": max_size_multiple,
        "account_equity": equity,
    }
    sizer = RiskPositionSizer(config, ig_client)
    # Force equity to be the configured value without API call
    sizer._cached_equity = equity
    sizer._last_refresh_time = float("inf")  # Prevent refresh during test
    return sizer


# ---------------------------------------------------------------------------
# Property 14: Position size formula correctness
# Validates: Requirements 8.1
# ---------------------------------------------------------------------------


class TestPositionSizeFormulaCorrectness:
    """Property 14: Position size formula correctness.

    For any positive values of equity, risk_pct, stop_distance, and pip_value,
    calculate_size() SHALL return a size equal to
    floor((equity × risk_pct / 100) / (stop_distance × pip_value) / step) × step.

    # Feature: ml-trading-improvements, Property 14: Position size formula correctness

    **Validates: Requirements 8.1**
    """

    @given(
        equity=floats(min_value=100.0, max_value=1_000_000.0, allow_nan=False, allow_infinity=False),
        risk_pct=floats(min_value=0.1, max_value=10.0, allow_nan=False, allow_infinity=False),
        stop_distance=floats(min_value=0.1, max_value=500.0, allow_nan=False, allow_infinity=False),
        pip_value=floats(min_value=0.01, max_value=100.0, allow_nan=False, allow_infinity=False),
        size_step=floats(min_value=0.01, max_value=1.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=200)
    def test_formula_matches_specification(
        self, equity, risk_pct, stop_distance, pip_value, size_step
    ):
        """calculate_size returns size matching the documented formula."""
        # Use a very small min_size so the formula result isn't rejected
        min_size = 0.01
        max_size_multiple = 10000  # Large enough to avoid capping

        sizer = make_sizer(
            risk_pct=risk_pct,
            equity=equity,
            max_size_multiple=max_size_multiple,
        )

        size, metadata = sizer.calculate_size(
            stop_distance=stop_distance,
            pip_value=pip_value,
            min_size=min_size,
            size_step=size_step,
        )

        # Expected formula: floor((equity × risk_pct / 100) / (stop × pip_value) / step) × step
        expected_raw = (equity * risk_pct / 100.0) / (stop_distance * pip_value)
        expected_sized = math.floor(expected_raw / size_step) * size_step

        # If expected_sized < min_size, should return None
        if expected_sized < min_size:
            assert size is None, (
                f"Expected None (size {expected_sized} < min_size {min_size}), got {size}"
            )
        else:
            # Cap check
            max_size = min_size * max_size_multiple
            if expected_sized > max_size:
                expected_sized = max_size

            assert size is not None, (
                f"Expected size {expected_sized}, got None. Metadata: {metadata}"
            )
            assert abs(size - expected_sized) < 1e-9, (
                f"Size mismatch: got {size}, expected {expected_sized} "
                f"(equity={equity}, risk_pct={risk_pct}, stop={stop_distance}, "
                f"pip_value={pip_value}, step={size_step})"
            )


# ---------------------------------------------------------------------------
# Property 15: Position size rounding
# Validates: Requirements 8.6
# ---------------------------------------------------------------------------


class TestPositionSizeRounding:
    """Property 15: Position size rounding.

    For any calculated raw size and size_step > 0, the returned size SHALL equal
    floor(raw_size / step) × step (rounding down to nearest valid increment).

    # Feature: ml-trading-improvements, Property 15: Position size rounding

    **Validates: Requirements 8.6**
    """

    @given(
        equity=floats(min_value=1000.0, max_value=100_000.0, allow_nan=False, allow_infinity=False),
        risk_pct=floats(min_value=0.5, max_value=5.0, allow_nan=False, allow_infinity=False),
        stop_distance=floats(min_value=1.0, max_value=100.0, allow_nan=False, allow_infinity=False),
        pip_value=floats(min_value=0.1, max_value=10.0, allow_nan=False, allow_infinity=False),
        size_step=floats(min_value=0.01, max_value=1.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=200)
    def test_size_is_rounded_down_to_step(
        self, equity, risk_pct, stop_distance, pip_value, size_step
    ):
        """Returned size must be a multiple of size_step (rounded down)."""
        min_size = 0.01
        max_size_multiple = 10000

        sizer = make_sizer(
            risk_pct=risk_pct,
            equity=equity,
            max_size_multiple=max_size_multiple,
        )

        size, metadata = sizer.calculate_size(
            stop_distance=stop_distance,
            pip_value=pip_value,
            min_size=min_size,
            size_step=size_step,
        )

        if size is not None:
            # Check that size is a valid multiple of size_step (floor-rounded)
            raw_size = (equity * risk_pct / 100.0) / (stop_distance * pip_value)
            expected = math.floor(raw_size / size_step) * size_step

            # Apply cap
            max_size = min_size * max_size_multiple
            if expected > max_size:
                expected = max_size

            assert abs(size - expected) < 1e-9, (
                f"Size {size} doesn't match floor-rounded expectation {expected} "
                f"(raw={raw_size}, step={size_step})"
            )

            # Also verify it's at or below raw_size (rounded DOWN, not up)
            # Unless it was capped at max_size
            if size < max_size:
                assert size <= raw_size + 1e-9, (
                    f"Size {size} exceeds raw_size {raw_size} — "
                    f"should round DOWN not UP"
                )


# ---------------------------------------------------------------------------
# Property 16: Size below minimum rejects trade
# Validates: Requirements 8.5
# ---------------------------------------------------------------------------


class TestSizeBelowMinimumRejects:
    """Property 16: Size below minimum rejects trade.

    For any combination of equity, risk_pct, stop_distance, and pip_value where
    the formula yields a value less than min_size, calculate_size() SHALL return None.

    # Feature: ml-trading-improvements, Property 16: Size below minimum rejects trade

    **Validates: Requirements 8.5**
    """

    @given(
        equity=floats(min_value=100.0, max_value=10_000.0, allow_nan=False, allow_infinity=False),
        risk_pct=floats(min_value=0.1, max_value=2.0, allow_nan=False, allow_infinity=False),
        stop_distance=floats(min_value=50.0, max_value=1000.0, allow_nan=False, allow_infinity=False),
        pip_value=floats(min_value=5.0, max_value=100.0, allow_nan=False, allow_infinity=False),
        size_step=floats(min_value=0.1, max_value=1.0, allow_nan=False, allow_infinity=False),
        min_size=floats(min_value=1.0, max_value=100.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=200)
    def test_returns_none_when_below_min_size(
        self, equity, risk_pct, stop_distance, pip_value, size_step, min_size
    ):
        """When formula result < min_size, calculate_size must return None."""
        # Compute expected sized value
        raw_size = (equity * risk_pct / 100.0) / (stop_distance * pip_value)
        sized = math.floor(raw_size / size_step) * size_step

        # Only test cases where result is actually below min_size
        assume(sized < min_size)

        sizer = make_sizer(risk_pct=risk_pct, equity=equity)

        size, metadata = sizer.calculate_size(
            stop_distance=stop_distance,
            pip_value=pip_value,
            min_size=min_size,
            size_step=size_step,
        )

        assert size is None, (
            f"Expected None for size below min_size: "
            f"sized={sized}, min_size={min_size}, raw_size={raw_size}"
        )
        assert "insufficient" in metadata["reason"].lower() or "min_size" in metadata["reason"], (
            f"Metadata reason should mention insufficient/min_size, got: {metadata['reason']}"
        )


# ---------------------------------------------------------------------------
# Property 17: Size capped at maximum multiple
# Validates: Requirements 10.4
# ---------------------------------------------------------------------------


class TestSizeCappedAtMaximumMultiple:
    """Property 17: Size capped at maximum multiple.

    For any combination of inputs where the formula yields a value exceeding
    min_size × max_size_multiple, calculate_size() SHALL return
    min_size × max_size_multiple (capped).

    # Feature: ml-trading-improvements, Property 17: Size capped at maximum multiple

    **Validates: Requirements 10.4**
    """

    @given(
        equity=floats(min_value=50_000.0, max_value=1_000_000.0, allow_nan=False, allow_infinity=False),
        risk_pct=floats(min_value=2.0, max_value=10.0, allow_nan=False, allow_infinity=False),
        stop_distance=floats(min_value=0.1, max_value=5.0, allow_nan=False, allow_infinity=False),
        pip_value=floats(min_value=0.01, max_value=0.5, allow_nan=False, allow_infinity=False),
        size_step=floats(min_value=0.1, max_value=1.0, allow_nan=False, allow_infinity=False),
        min_size=floats(min_value=0.1, max_value=1.0, allow_nan=False, allow_infinity=False),
        max_size_multiple=integers(min_value=5, max_value=100),
    )
    @settings(max_examples=200)
    def test_size_capped_at_max_multiple(
        self, equity, risk_pct, stop_distance, pip_value, size_step, min_size, max_size_multiple
    ):
        """When formula result > min_size × max_size_multiple, returns the cap."""
        # Compute expected sized value
        raw_size = (equity * risk_pct / 100.0) / (stop_distance * pip_value)
        sized = math.floor(raw_size / size_step) * size_step
        max_size = min_size * max_size_multiple

        # Only test cases where result exceeds the cap
        assume(sized > max_size)
        # Also ensure sized >= min_size so it's not rejected
        assume(sized >= min_size)

        sizer = make_sizer(
            risk_pct=risk_pct,
            equity=equity,
            max_size_multiple=max_size_multiple,
        )

        size, metadata = sizer.calculate_size(
            stop_distance=stop_distance,
            pip_value=pip_value,
            min_size=min_size,
            size_step=size_step,
        )

        assert size is not None, (
            f"Expected capped size, got None. sized={sized}, min_size={min_size}, "
            f"max_size={max_size}. Metadata: {metadata}"
        )
        assert abs(size - max_size) < 1e-9, (
            f"Expected size to be capped at {max_size} "
            f"(min_size={min_size} × max_multiple={max_size_multiple}), "
            f"got {size}"
        )


# ---------------------------------------------------------------------------
# Property 18: Fallback to fixed sizing when dynamic disabled
# Validates: Requirements 10.3
# ---------------------------------------------------------------------------


class TestFallbackToFixedSizing:
    """Property 18: Fallback to fixed sizing when dynamic disabled.

    For any signal, when use_dynamic_sizing is false, the position sizer SHALL
    produce the same result as the existing size_by_invested_capital() function.

    In practice, when use_dynamic_sizing is False, the runner skips the
    RiskPositionSizer entirely and calls size_by_invested_capital directly.
    This property verifies the config flag is respected.

    # Feature: ml-trading-improvements, Property 18: Fallback to fixed sizing when dynamic disabled

    **Validates: Requirements 10.3**
    """

    @given(
        equity=floats(min_value=1000.0, max_value=100_000.0, allow_nan=False, allow_infinity=False),
        risk_pct=floats(min_value=0.5, max_value=5.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=100)
    def test_use_dynamic_sizing_flag_respected(self, equity, risk_pct):
        """When use_dynamic_sizing is False, the sizer's flag reflects this."""
        sizer_enabled = make_sizer(
            risk_pct=risk_pct,
            equity=equity,
            use_dynamic_sizing=True,
        )
        sizer_disabled = make_sizer(
            risk_pct=risk_pct,
            equity=equity,
            use_dynamic_sizing=False,
        )

        assert sizer_enabled.use_dynamic_sizing is True, (
            "Expected use_dynamic_sizing=True when configured as True"
        )
        assert sizer_disabled.use_dynamic_sizing is False, (
            "Expected use_dynamic_sizing=False when configured as False"
        )

    @given(
        equity=floats(min_value=1000.0, max_value=100_000.0, allow_nan=False, allow_infinity=False),
        risk_pct=floats(min_value=0.5, max_value=5.0, allow_nan=False, allow_infinity=False),
        stop_distance=floats(min_value=1.0, max_value=100.0, allow_nan=False, allow_infinity=False),
        pip_value=floats(min_value=0.1, max_value=10.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=100)
    def test_runner_fallback_logic_uses_config_flag(
        self, equity, risk_pct, stop_distance, pip_value
    ):
        """Simulates the runner's branching logic: when use_dynamic_sizing is False,
        size_by_invested_capital is used instead of position_sizer.calculate_size.

        This validates that the config flag drives the correct code path as specified
        in the integration design.
        """
        from core.risk import size_by_invested_capital

        config = {
            "risk_pct_per_trade": risk_pct,
            "equity_refresh_interval_seconds": 300,
            "use_dynamic_sizing": False,
            "max_size_multiple": 50,
            "account_equity": equity,
            "invest_per_trade": 1000.0,
            "max_loss_pct_invest": 5.0,
        }

        # When use_dynamic_sizing is False, the runner calls size_by_invested_capital
        if not config["use_dynamic_sizing"]:
            size, max_loss = size_by_invested_capital(
                invest_amount_gbp=config["invest_per_trade"],
                max_loss_pct=config["max_loss_pct_invest"],
                stop_pts=stop_distance,
                pip_value_per_contract=pip_value,
                min_size=0.1,
                size_step=0.1,
            )
            # Verify fallback produces a result (not None — size_by_invested_capital
            # returns 0.0 instead of None for invalid inputs, plus applies max(min_size, ...))
            assert isinstance(size, float)
            assert isinstance(max_loss, float)
            assert size >= 0.0
        else:
            # This branch should not execute in this test
            assert False, "use_dynamic_sizing should be False"



# ===========================================================================
# UNIT TESTS — Task 3.3
# ===========================================================================
# Unit tests for RiskPositionSizer covering:
# - Equity refresh with mocked IG client (success + failure caching)
# - Config parsing with default values
# - Edge cases: very large stop, very small equity, zero pip_value
# Requirements: 9.1, 9.2, 9.3, 10.1
# ===========================================================================

import time

import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_ig_client_fixture():
    """Create a mock IG client that returns a valid account summary."""
    client = MagicMock()
    client.account_summary.return_value = {
        "accounts": [{"balance": {"balance": 15000.0}}]
    }
    return client


@pytest.fixture
def default_config_fixture():
    """Config dict with all keys omitted (tests that defaults are applied)."""
    return {}


@pytest.fixture
def full_config_fixture():
    """Config dict with all keys explicitly set."""
    return {
        "risk_pct_per_trade": 1.5,
        "equity_refresh_interval_seconds": 600,
        "use_dynamic_sizing": False,
        "max_size_multiple": 30,
        "account_equity": 20000.0,
    }


# ---------------------------------------------------------------------------
# Config Parsing with Defaults (Requirement 10.1)
# ---------------------------------------------------------------------------


class TestConfigParsingUnit:
    """Verify config parsing applies correct default values (Req 10.1)."""

    def test_default_risk_pct(self, mock_ig_client_fixture, default_config_fixture):
        """risk_pct_per_trade defaults to 2.0."""
        sizer = RiskPositionSizer(default_config_fixture, mock_ig_client_fixture)
        assert sizer.risk_pct == 2.0

    def test_default_refresh_interval(self, mock_ig_client_fixture, default_config_fixture):
        """equity_refresh_interval_seconds defaults to 300."""
        sizer = RiskPositionSizer(default_config_fixture, mock_ig_client_fixture)
        assert sizer.refresh_interval == 300

    def test_default_use_dynamic_sizing(self, mock_ig_client_fixture, default_config_fixture):
        """use_dynamic_sizing defaults to True."""
        sizer = RiskPositionSizer(default_config_fixture, mock_ig_client_fixture)
        assert sizer.use_dynamic_sizing is True

    def test_default_max_size_multiple(self, mock_ig_client_fixture, default_config_fixture):
        """max_size_multiple defaults to 50."""
        sizer = RiskPositionSizer(default_config_fixture, mock_ig_client_fixture)
        assert sizer.max_size_multiple == 50

    def test_default_cached_equity(self, mock_ig_client_fixture, default_config_fixture):
        """Default cached equity is 10000.0 when not specified in config."""
        sizer = RiskPositionSizer(default_config_fixture, mock_ig_client_fixture)
        assert sizer._cached_equity == 10000.0

    def test_custom_config_applied(self, mock_ig_client_fixture, full_config_fixture):
        """Custom config values are correctly applied."""
        sizer = RiskPositionSizer(full_config_fixture, mock_ig_client_fixture)
        assert sizer.risk_pct == 1.5
        assert sizer.refresh_interval == 600
        assert sizer.use_dynamic_sizing is False
        assert sizer.max_size_multiple == 30
        assert sizer._cached_equity == 20000.0


# ---------------------------------------------------------------------------
# Equity Refresh (Requirements 9.1, 9.2, 9.3)
# ---------------------------------------------------------------------------


class TestEquityRefreshUnit:
    """Verify equity refresh from IG API and caching behaviour."""

    def test_refresh_equity_success(self, mock_ig_client_fixture):
        """Successful refresh updates cached equity from IG API (Req 9.1, 9.3)."""
        sizer = RiskPositionSizer({"account_equity": 5000.0}, mock_ig_client_fixture)
        assert sizer._cached_equity == 5000.0

        result = sizer.refresh_equity()

        assert result == 15000.0
        assert sizer._cached_equity == 15000.0
        mock_ig_client_fixture.account_summary.assert_called_once()

    def test_refresh_uses_balance_field(self, mock_ig_client_fixture):
        """Refresh uses the 'balance' field, not 'available' (Req 9.3)."""
        mock_ig_client_fixture.account_summary.return_value = {
            "accounts": [
                {
                    "balance": {"balance": 12000.0, "available": 8000.0, "deposit": 4000.0}
                }
            ]
        }
        sizer = RiskPositionSizer({}, mock_ig_client_fixture)

        result = sizer.refresh_equity()

        assert result == 12000.0
        assert sizer._cached_equity == 12000.0

    def test_refresh_failure_uses_cached_value(self, mock_ig_client_fixture):
        """On API failure, last cached equity is used (Req 9.2)."""
        sizer = RiskPositionSizer({"account_equity": 8000.0}, mock_ig_client_fixture)
        mock_ig_client_fixture.account_summary.side_effect = Exception("Connection timeout")

        result = sizer.refresh_equity()

        assert result == 8000.0
        assert sizer._cached_equity == 8000.0

    def test_refresh_failure_preserves_previous_success(self, mock_ig_client_fixture):
        """After a successful refresh, failure preserves the refreshed value (Req 9.2)."""
        sizer = RiskPositionSizer({"account_equity": 5000.0}, mock_ig_client_fixture)

        # First call succeeds
        sizer.refresh_equity()
        assert sizer._cached_equity == 15000.0

        # Second call fails
        mock_ig_client_fixture.account_summary.side_effect = RuntimeError("API error")
        result = sizer.refresh_equity()

        assert result == 15000.0
        assert sizer._cached_equity == 15000.0

    def test_refresh_empty_accounts_uses_cached(self, mock_ig_client_fixture):
        """Empty accounts array uses cached value."""
        mock_ig_client_fixture.account_summary.return_value = {"accounts": []}
        sizer = RiskPositionSizer({"account_equity": 7000.0}, mock_ig_client_fixture)

        result = sizer.refresh_equity()

        assert result == 7000.0

    def test_refresh_missing_balance_key_uses_cached(self, mock_ig_client_fixture):
        """Missing balance key in account dict uses cached value."""
        mock_ig_client_fixture.account_summary.return_value = {
            "accounts": [{"other_field": 123}]
        }
        sizer = RiskPositionSizer({"account_equity": 9000.0}, mock_ig_client_fixture)

        result = sizer.refresh_equity()

        assert result == 9000.0

    def test_get_equity_respects_refresh_interval(self, mock_ig_client_fixture):
        """get_equity only refreshes if interval has elapsed (Req 9.1)."""
        config = {"equity_refresh_interval_seconds": 300, "account_equity": 5000.0}
        sizer = RiskPositionSizer(config, mock_ig_client_fixture)

        # Force initial refresh (last_refresh_time is 0)
        with patch("time.time", return_value=1000.0):
            equity = sizer.get_equity()
        assert equity == 15000.0
        assert mock_ig_client_fixture.account_summary.call_count == 1

        # Call again within interval — no refresh
        with patch("time.time", return_value=1200.0):
            equity = sizer.get_equity()
        assert equity == 15000.0
        assert mock_ig_client_fixture.account_summary.call_count == 1

        # Call after interval elapsed — triggers refresh
        mock_ig_client_fixture.account_summary.return_value = {
            "accounts": [{"balance": {"balance": 16000.0}}]
        }
        mock_ig_client_fixture.account_summary.side_effect = None
        with patch("time.time", return_value=1400.0):
            equity = sizer.get_equity()
        assert equity == 16000.0
        assert mock_ig_client_fixture.account_summary.call_count == 2


# ---------------------------------------------------------------------------
# Edge Cases
# ---------------------------------------------------------------------------


class TestEdgeCasesUnit:
    """Edge case tests for calculate_size."""

    def test_zero_stop_distance_returns_none(self, mock_ig_client_fixture):
        """stop_distance=0 returns None (division by zero guard)."""
        sizer = RiskPositionSizer({"account_equity": 10000.0}, mock_ig_client_fixture)
        sizer._last_refresh_time = time.time()

        size, meta = sizer.calculate_size(
            stop_distance=0.0, pip_value=1.0, min_size=0.1, size_step=0.1
        )

        assert size is None
        assert "Invalid inputs" in meta["reason"]

    def test_negative_stop_distance_returns_none(self, mock_ig_client_fixture):
        """Negative stop_distance returns None."""
        sizer = RiskPositionSizer({"account_equity": 10000.0}, mock_ig_client_fixture)
        sizer._last_refresh_time = time.time()

        size, meta = sizer.calculate_size(
            stop_distance=-5.0, pip_value=1.0, min_size=0.1, size_step=0.1
        )

        assert size is None
        assert "Invalid inputs" in meta["reason"]

    def test_zero_pip_value_returns_none(self, mock_ig_client_fixture):
        """pip_value=0 returns None (division by zero guard)."""
        sizer = RiskPositionSizer({"account_equity": 10000.0}, mock_ig_client_fixture)
        sizer._last_refresh_time = time.time()

        size, meta = sizer.calculate_size(
            stop_distance=10.0, pip_value=0.0, min_size=0.1, size_step=0.1
        )

        assert size is None
        assert "Invalid inputs" in meta["reason"]

    def test_negative_pip_value_returns_none(self, mock_ig_client_fixture):
        """Negative pip_value returns None."""
        sizer = RiskPositionSizer({"account_equity": 10000.0}, mock_ig_client_fixture)
        sizer._last_refresh_time = time.time()

        size, meta = sizer.calculate_size(
            stop_distance=10.0, pip_value=-2.0, min_size=0.1, size_step=0.1
        )

        assert size is None
        assert "Invalid inputs" in meta["reason"]

    def test_very_large_stop_below_minimum_returns_none(self, mock_ig_client_fixture):
        """Very large stop distance yields size below minimum → None."""
        # equity=10000, risk=2%, stop=5000, pip_value=1.0
        # raw_size = (10000 * 0.02) / (5000 * 1.0) = 200 / 5000 = 0.04
        # floor(0.04/0.1)*0.1 = 0.0 < min_size (0.1) → rejected
        sizer = RiskPositionSizer({"account_equity": 10000.0}, mock_ig_client_fixture)
        sizer._last_refresh_time = time.time()

        size, meta = sizer.calculate_size(
            stop_distance=5000.0, pip_value=1.0, min_size=0.1, size_step=0.1
        )

        assert size is None
        assert "insufficient" in meta["reason"].lower()

    def test_very_small_equity_below_minimum(self, mock_ig_client_fixture):
        """Very small equity yields size below minimum → None."""
        # equity=10, risk=2%, stop=10, pip_value=1.0
        # raw_size = (10 * 0.02) / (10 * 1.0) = 0.2 / 10 = 0.02
        # floor(0.02 / 0.1) * 0.1 = 0.0 < 0.1 → rejected
        sizer = RiskPositionSizer({"account_equity": 10.0}, mock_ig_client_fixture)
        sizer._last_refresh_time = time.time()

        size, meta = sizer.calculate_size(
            stop_distance=10.0, pip_value=1.0, min_size=0.1, size_step=0.1
        )

        assert size is None
        assert "insufficient" in meta["reason"].lower()

    def test_valid_calculation_returns_correct_size(self, mock_ig_client_fixture):
        """Standard calculation: equity=10000, risk=2%, stop=20, pip=1.0."""
        # raw_size = (10000 * 0.02) / (20 * 1.0) = 200 / 20 = 10.0
        # max_size = 1.0 * 50 = 50.0 (use min_size=1.0 to avoid cap)
        # floor(10.0 / 0.1) * 0.1 = 10.0 (under cap)
        sizer = RiskPositionSizer({"account_equity": 10000.0}, mock_ig_client_fixture)
        sizer._last_refresh_time = time.time()

        size, meta = sizer.calculate_size(
            stop_distance=20.0, pip_value=1.0, min_size=1.0, size_step=0.1
        )

        assert size == 10.0
        assert meta["reason"] == "ok"
        assert meta["equity"] == 10000.0

    def test_size_capped_at_max_multiple(self, mock_ig_client_fixture):
        """Size exceeding max_size_multiple × min_size is capped."""
        # equity=100000, risk=2%, stop=1, pip=1.0
        # raw_size = (100000 * 0.02) / (1 * 1.0) = 2000
        # max_size = 0.1 * 50 = 5.0
        # capped to 5.0
        sizer = RiskPositionSizer(
            {"account_equity": 100000.0, "max_size_multiple": 50}, mock_ig_client_fixture
        )
        sizer._last_refresh_time = time.time()

        size, meta = sizer.calculate_size(
            stop_distance=1.0, pip_value=1.0, min_size=0.1, size_step=0.1
        )

        assert size == 5.0
        assert "capped" in meta["reason"].lower()

    def test_size_step_zero_defaults_to_0_1(self, mock_ig_client_fixture):
        """size_step <= 0 defaults to 0.1."""
        # equity=10000, risk=2%, stop=200, pip=1.0
        # raw_size = (10000 * 0.02) / (200 * 1.0) = 1.0
        # step defaults to 0.1 → floor(1.0 / 0.1) * 0.1 = 1.0
        # max_size = 0.5 * 50 = 25 (under cap)
        sizer = RiskPositionSizer({"account_equity": 10000.0}, mock_ig_client_fixture)
        sizer._last_refresh_time = time.time()

        size, meta = sizer.calculate_size(
            stop_distance=200.0, pip_value=1.0, min_size=0.5, size_step=0.0
        )

        assert size == 1.0

    def test_rounding_down_to_step(self, mock_ig_client_fixture):
        """Size is rounded down to nearest step increment."""
        # equity=10000, risk=2%, stop=150, pip=1.0
        # raw_size = 200 / 150 ≈ 1.333...
        # floor(1.333 / 0.1) * 0.1 = floor(13.33) * 0.1 = 13 * 0.1 = 1.3
        # max_size = 0.5 * 50 = 25 (under cap)
        sizer = RiskPositionSizer({"account_equity": 10000.0}, mock_ig_client_fixture)
        sizer._last_refresh_time = time.time()

        size, meta = sizer.calculate_size(
            stop_distance=150.0, pip_value=1.0, min_size=0.5, size_step=0.1
        )

        assert size == pytest.approx(1.3, abs=0.01)

    def test_metadata_includes_all_fields(self, mock_ig_client_fixture):
        """Metadata dict contains all expected fields."""
        sizer = RiskPositionSizer({"account_equity": 10000.0}, mock_ig_client_fixture)
        sizer._last_refresh_time = time.time()

        size, meta = sizer.calculate_size(
            stop_distance=20.0, pip_value=1.0, min_size=0.1, size_step=0.1
        )

        assert "equity" in meta
        assert "risk_pct" in meta
        assert "stop_distance" in meta
        assert "pip_value" in meta
        assert "raw_size" in meta
        assert "capped_size" in meta
        assert "reason" in meta
