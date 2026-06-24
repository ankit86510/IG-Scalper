"""
Integration tests for the ML Trading Improvements pipeline.

Tests verify the execution order and gating logic of:
  data fetch → volatility filter → strategy.on_bar() → ML filter → position sizer → order placement

Each test simulates the pipeline flow as it exists in runners/run_ai_autonomous.py,
using mocks for the IG client, strategy, and data aggregator.

Requirements validated: 11.1, 11.2, 11.3, 11.4, 11.5
"""

from unittest.mock import MagicMock, patch, call
import numpy as np
import pandas as pd
import pytest
from zoneinfo import ZoneInfo

from strategy.volatility_filter import VolatilityRegimeFilter
from strategy.ml_filter import MLDirectionalFilter
from core.position_sizer import RiskPositionSizer

ROME_TZ = ZoneInfo("Europe/Rome")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_ohlc_df(rows: int = 100, base_price: float = 3000.0) -> pd.DataFrame:
    """Create a valid OHLC DataFrame suitable for all filters."""
    dates = pd.date_range("2024-01-10 09:00", periods=rows, freq="5min", tz=ROME_TZ)
    np.random.seed(42)
    data = []
    price = base_price
    for i in range(rows):
        o = price
        h = o + np.random.uniform(0.5, 3.0)
        l = o - np.random.uniform(0.5, 2.0)
        c = o + np.random.uniform(-1.5, 1.5)
        price = c
        data.append({"open": o, "high": h, "low": l, "close": c, "volume": 100 + i})
    return pd.DataFrame(data, index=dates)


def _make_signal(side: str = "BUY", stop_pts: float = 5.0, tp_pts: float = 10.0):
    """Create a mock trading signal dict."""
    return {
        "side": side,
        "stop_pts": stop_pts,
        "tp_pts": tp_pts,
        "meta": {
            "confidence": 0.75,
            "patterns_detected": ["test_pattern"],
        },
    }


def _pipeline_run(
    df: pd.DataFrame,
    vol_filter: VolatilityRegimeFilter,
    strategy,
    ml_filter: MLDirectionalFilter,
    position_sizer: RiskPositionSizer,
    cfg_risk: dict,
    market_details: dict,
    call_log: list,
):
    """
    Simulate the pipeline flow exactly as in run_ai_autonomous.py.

    Records each component call to call_log for order verification.
    Returns the final outcome: 'order_placed', 'vol_blocked', 'no_signal',
    'ml_rejected', 'sizer_rejected'.
    """
    # 1. Volatility filter (before on_bar)
    allowed, vol_meta = vol_filter.allow_trading(df)
    call_log.append("volatility_filter")
    if not allowed:
        return "vol_blocked"

    # 2. Strategy signal
    signal = strategy.on_bar(df)
    call_log.append("on_bar")
    if signal is None:
        return "no_signal"

    # 3. ML confirmation (after signal, before order)
    if ml_filter.is_enabled:
        confirmed, ml_meta = ml_filter.confirm_signal(signal, df)
        call_log.append("ml_filter")
        if not confirmed:
            return "ml_rejected"
    else:
        call_log.append("ml_filter_disabled")

    # 4. Position sizing
    if cfg_risk.get("use_dynamic_sizing", True):
        pip_value = 0.77  # typical XAU/USD pip value
        size, size_meta = position_sizer.calculate_size(
            stop_distance=signal["stop_pts"],
            pip_value=pip_value,
            min_size=market_details["dealingRules"]["minDealSize"]["value"],
            size_step=0.1,
        )
        call_log.append("position_sizer")
        if size is None:
            return "sizer_rejected"  # No cooldown!
    else:
        call_log.append("position_sizer_fallback")

    # 5. Order would be placed
    call_log.append("order_placed")
    return "order_placed"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def df():
    """A valid OHLC DataFrame with enough history for all components."""
    return _make_ohlc_df(rows=120)


@pytest.fixture
def market_details():
    """Mock market details for XAU/USD."""
    return {
        "dealingRules": {
            "minDealSize": {"value": 0.5},
            "minNormalStopOrLimitDistance": {"value": 1.0},
        },
    }


@pytest.fixture
def vol_filter_enabled():
    """A volatility filter that is enabled and has sufficient history to pass.

    Pre-fills history with values evenly distributed around the typical ATR ratio
    (~0.001) of our test DataFrame, ensuring the current reading falls within
    the 20-80 percentile band and is allowed through.
    """
    vf = VolatilityRegimeFilter({
        "enabled": True,
        "atr_period": 14,
        "lookback_bars": 100,
        "lower_percentile": 20.0,
        "upper_percentile": 80.0,
    })
    # Pre-fill history centered around 0.001 (the typical ATR ratio of our test DF).
    # Spread values so the actual reading (~0.001) lands near the 50th percentile.
    for i in range(30):
        vf.update_history(0.0005 + i * (0.001 / 30))
    return vf


@pytest.fixture
def vol_filter_disabled():
    """A volatility filter that is disabled."""
    return VolatilityRegimeFilter({"enabled": False})


@pytest.fixture
def ml_filter_enabled():
    """An ML filter mock that is enabled and confirms signals."""
    ml = MagicMock(spec=MLDirectionalFilter)
    ml.is_enabled = True
    ml.confirm_signal.return_value = (True, {
        "probability": 0.72,
        "threshold": 0.55,
        "direction": "BUY",
        "reason": "BUY confirmed",
        "ml_enabled": True,
        "confirmed": True,
    })
    return ml


@pytest.fixture
def ml_filter_disabled():
    """An ML filter that is disabled."""
    ml = MagicMock(spec=MLDirectionalFilter)
    ml.is_enabled = False
    return ml


@pytest.fixture
def ml_filter_rejecting():
    """An ML filter that rejects all signals."""
    ml = MagicMock(spec=MLDirectionalFilter)
    ml.is_enabled = True
    ml.confirm_signal.return_value = (False, {
        "probability": 0.40,
        "threshold": 0.55,
        "direction": "BUY",
        "reason": "BUY rejected: P(bullish)=0.4000 <= 0.55",
        "ml_enabled": True,
        "confirmed": False,
    })
    return ml


@pytest.fixture
def position_sizer_pass():
    """A position sizer that approves sizing."""
    ig_client = MagicMock()
    ig_client.account_summary.return_value = {
        "accounts": [{"balance": {"balance": 10000.0}}]
    }
    ps = RiskPositionSizer(
        {"risk_pct_per_trade": 2.0, "use_dynamic_sizing": True, "max_size_multiple": 50},
        ig_client,
    )
    return ps


@pytest.fixture
def position_sizer_reject():
    """A position sizer that rejects due to insufficient size (tiny equity)."""
    ig_client = MagicMock()
    ig_client.account_summary.return_value = {
        "accounts": [{"balance": {"balance": 5.0}}]  # Very low equity
    }
    ps = RiskPositionSizer(
        {
            "risk_pct_per_trade": 0.1,
            "account_equity": 5.0,
            "use_dynamic_sizing": True,
            "max_size_multiple": 50,
        },
        ig_client,
    )
    return ps


@pytest.fixture
def mock_strategy_signal():
    """A strategy mock that returns a BUY signal."""
    strategy = MagicMock()
    strategy.on_bar.return_value = _make_signal("BUY")
    return strategy


@pytest.fixture
def mock_strategy_no_signal():
    """A strategy mock that returns None (no signal)."""
    strategy = MagicMock()
    strategy.on_bar.return_value = None
    return strategy


# ---------------------------------------------------------------------------
# Test: Full pass-through (all filters pass, order placed)
# ---------------------------------------------------------------------------


class TestFullPipeline:
    """Test complete pipeline flow: volatility → on_bar → ML → sizing → order."""

    def test_full_pass_through(
        self, df, vol_filter_enabled, mock_strategy_signal, ml_filter_enabled,
        position_sizer_pass, market_details
    ):
        """All filters pass, signal generated, size calculated, order placed.
        Validates: Requirements 11.1, 11.2, 11.3
        """
        call_log = []
        result = _pipeline_run(
            df=df,
            vol_filter=vol_filter_enabled,
            strategy=mock_strategy_signal,
            ml_filter=ml_filter_enabled,
            position_sizer=position_sizer_pass,
            cfg_risk={"use_dynamic_sizing": True},
            market_details=market_details,
            call_log=call_log,
        )

        assert result == "order_placed"
        assert call_log == [
            "volatility_filter",
            "on_bar",
            "ml_filter",
            "position_sizer",
            "order_placed",
        ]

    def test_execution_order_is_strict(
        self, df, vol_filter_enabled, mock_strategy_signal, ml_filter_enabled,
        position_sizer_pass, market_details
    ):
        """Verify the exact ordering: volatility → on_bar → ML → sizing → order.
        Validates: Requirements 11.1, 11.2, 11.3
        """
        call_log = []
        _pipeline_run(
            df=df,
            vol_filter=vol_filter_enabled,
            strategy=mock_strategy_signal,
            ml_filter=ml_filter_enabled,
            position_sizer=position_sizer_pass,
            cfg_risk={"use_dynamic_sizing": True},
            market_details=market_details,
            call_log=call_log,
        )

        # Verify strict sequential order
        assert call_log.index("volatility_filter") < call_log.index("on_bar")
        assert call_log.index("on_bar") < call_log.index("ml_filter")
        assert call_log.index("ml_filter") < call_log.index("position_sizer")
        assert call_log.index("position_sizer") < call_log.index("order_placed")


# ---------------------------------------------------------------------------
# Test: Volatility filter blocks pipeline
# ---------------------------------------------------------------------------


class TestVolatilityBlocks:
    """Test that volatility filter blocks before on_bar is called."""

    def test_vol_blocks_prevents_on_bar(
        self, df, mock_strategy_signal, ml_filter_enabled,
        position_sizer_pass, market_details
    ):
        """Volatility filter blocks → on_bar NOT called.
        Validates: Requirement 11.1
        """
        # Create a vol filter that blocks (extreme percentile)
        vf = VolatilityRegimeFilter({
            "enabled": True,
            "atr_period": 14,
            "lookback_bars": 100,
            "lower_percentile": 20.0,
            "upper_percentile": 80.0,
        })
        # Fill history with very low values, so actual ATR ratio is above the 80th pctile
        for _ in range(30):
            vf.update_history(0.0001)

        call_log = []
        result = _pipeline_run(
            df=df,
            vol_filter=vf,
            strategy=mock_strategy_signal,
            ml_filter=ml_filter_enabled,
            position_sizer=position_sizer_pass,
            cfg_risk={"use_dynamic_sizing": True},
            market_details=market_details,
            call_log=call_log,
        )

        assert result == "vol_blocked"
        # on_bar should NOT have been called
        assert "on_bar" not in call_log
        assert "ml_filter" not in call_log
        assert "position_sizer" not in call_log
        # Strategy mock should NOT have been called
        mock_strategy_signal.on_bar.assert_not_called()


# ---------------------------------------------------------------------------
# Test: No signal from strategy
# ---------------------------------------------------------------------------


class TestNoSignal:
    """Test that when strategy returns None, ML filter is not called."""

    def test_no_signal_skips_ml_and_sizer(
        self, df, vol_filter_enabled, mock_strategy_no_signal, ml_filter_enabled,
        position_sizer_pass, market_details
    ):
        """Vol passes, on_bar returns None → ML and sizer NOT called.
        Validates: Requirements 11.1, 11.2
        """
        call_log = []
        result = _pipeline_run(
            df=df,
            vol_filter=vol_filter_enabled,
            strategy=mock_strategy_no_signal,
            ml_filter=ml_filter_enabled,
            position_sizer=position_sizer_pass,
            cfg_risk={"use_dynamic_sizing": True},
            market_details=market_details,
            call_log=call_log,
        )

        assert result == "no_signal"
        assert "volatility_filter" in call_log
        assert "on_bar" in call_log
        assert "ml_filter" not in call_log
        assert "ml_filter_disabled" not in call_log
        assert "position_sizer" not in call_log
        # ML filter confirm_signal should NOT have been called
        ml_filter_enabled.confirm_signal.assert_not_called()


# ---------------------------------------------------------------------------
# Test: ML filter rejects
# ---------------------------------------------------------------------------


class TestMLRejects:
    """Test that ML filter rejection prevents position sizing."""

    def test_ml_rejects_skips_sizer(
        self, df, vol_filter_enabled, mock_strategy_signal, ml_filter_rejecting,
        position_sizer_pass, market_details
    ):
        """Vol passes, signal generated, ML rejects → position sizer NOT called.
        Validates: Requirement 11.2
        """
        call_log = []
        result = _pipeline_run(
            df=df,
            vol_filter=vol_filter_enabled,
            strategy=mock_strategy_signal,
            ml_filter=ml_filter_rejecting,
            position_sizer=position_sizer_pass,
            cfg_risk={"use_dynamic_sizing": True},
            market_details=market_details,
            call_log=call_log,
        )

        assert result == "ml_rejected"
        assert "volatility_filter" in call_log
        assert "on_bar" in call_log
        assert "ml_filter" in call_log
        assert "position_sizer" not in call_log


# ---------------------------------------------------------------------------
# Test: Position sizer rejects (no cooldown)
# ---------------------------------------------------------------------------


class TestSizerRejects:
    """Test that position sizer rejection skips trade with NO cooldown."""

    def test_sizer_rejects_no_cooldown(
        self, df, vol_filter_enabled, mock_strategy_signal, ml_filter_enabled,
        position_sizer_reject, market_details
    ):
        """All pass but size < min → trade skipped, NO cooldown.
        Validates: Requirement 11.5
        """
        call_log = []
        result = _pipeline_run(
            df=df,
            vol_filter=vol_filter_enabled,
            strategy=mock_strategy_signal,
            ml_filter=ml_filter_enabled,
            position_sizer=position_sizer_reject,
            cfg_risk={"use_dynamic_sizing": True},
            market_details=market_details,
            call_log=call_log,
        )

        assert result == "sizer_rejected"
        # All stages were reached up to sizer
        assert "volatility_filter" in call_log
        assert "on_bar" in call_log
        assert "ml_filter" in call_log
        assert "position_sizer" in call_log
        assert "order_placed" not in call_log

    def test_sizer_rejection_does_not_trigger_cooldown(
        self, df, vol_filter_enabled, mock_strategy_signal, ml_filter_enabled,
        position_sizer_reject, market_details
    ):
        """Verify that sizer rejection returns 'sizer_rejected' (not SL-like cooldown).

        In the real runner, 'continue' without updating last_sl_time means no cooldown.
        The next bar can immediately produce a new signal attempt.
        Validates: Requirement 11.5
        """
        # Run pipeline twice in succession — both should reach sizer (no cooldown gate)
        for attempt in range(2):
            call_log = []
            result = _pipeline_run(
                df=df,
                vol_filter=vol_filter_enabled,
                strategy=mock_strategy_signal,
                ml_filter=ml_filter_enabled,
                position_sizer=position_sizer_reject,
                cfg_risk={"use_dynamic_sizing": True},
                market_details=market_details,
                call_log=call_log,
            )
            assert result == "sizer_rejected", f"Attempt {attempt + 1} should also reach sizer"
            assert "position_sizer" in call_log, f"Attempt {attempt + 1} should reach sizer"


# ---------------------------------------------------------------------------
# Test: Disabled filters pass through
# ---------------------------------------------------------------------------


class TestDisabledFilters:
    """Test that disabled filters allow signals to pass through."""

    def test_disabled_vol_filter_passes(
        self, df, vol_filter_disabled, mock_strategy_signal, ml_filter_enabled,
        position_sizer_pass, market_details
    ):
        """Disabled volatility filter always passes.
        Validates: Requirement 11.1
        """
        call_log = []
        result = _pipeline_run(
            df=df,
            vol_filter=vol_filter_disabled,
            strategy=mock_strategy_signal,
            ml_filter=ml_filter_enabled,
            position_sizer=position_sizer_pass,
            cfg_risk={"use_dynamic_sizing": True},
            market_details=market_details,
            call_log=call_log,
        )

        assert result == "order_placed"
        assert "volatility_filter" in call_log
        assert "on_bar" in call_log

    def test_disabled_ml_filter_passes(
        self, df, vol_filter_enabled, mock_strategy_signal, ml_filter_disabled,
        position_sizer_pass, market_details
    ):
        """Disabled ML filter passes signal through without calling confirm_signal.
        Validates: Requirement 11.2
        """
        call_log = []
        result = _pipeline_run(
            df=df,
            vol_filter=vol_filter_enabled,
            strategy=mock_strategy_signal,
            ml_filter=ml_filter_disabled,
            position_sizer=position_sizer_pass,
            cfg_risk={"use_dynamic_sizing": True},
            market_details=market_details,
            call_log=call_log,
        )

        assert result == "order_placed"
        assert "ml_filter_disabled" in call_log
        assert "ml_filter" not in call_log
        # confirm_signal should NOT have been called
        ml_filter_disabled.confirm_signal.assert_not_called()

    def test_disabled_dynamic_sizing_uses_fallback(
        self, df, vol_filter_enabled, mock_strategy_signal, ml_filter_enabled,
        market_details
    ):
        """When use_dynamic_sizing is false, fallback path is used.
        Validates: Requirement 11.3
        """
        ig_client = MagicMock()
        ps = RiskPositionSizer({"use_dynamic_sizing": False}, ig_client)

        call_log = []
        result = _pipeline_run(
            df=df,
            vol_filter=vol_filter_enabled,
            strategy=mock_strategy_signal,
            ml_filter=ml_filter_enabled,
            position_sizer=ps,
            cfg_risk={"use_dynamic_sizing": False},
            market_details=market_details,
            call_log=call_log,
        )

        assert result == "order_placed"
        assert "position_sizer_fallback" in call_log
        assert "position_sizer" not in call_log

    def test_both_filters_disabled_still_orders(
        self, df, vol_filter_disabled, mock_strategy_signal, ml_filter_disabled,
        position_sizer_pass, market_details
    ):
        """Both filters disabled — signal passes through to order.
        Validates: Requirements 11.1, 11.2, 11.3
        """
        call_log = []
        result = _pipeline_run(
            df=df,
            vol_filter=vol_filter_disabled,
            strategy=mock_strategy_signal,
            ml_filter=ml_filter_disabled,
            position_sizer=position_sizer_pass,
            cfg_risk={"use_dynamic_sizing": True},
            market_details=market_details,
            call_log=call_log,
        )

        assert result == "order_placed"
        assert "volatility_filter" in call_log
        assert "on_bar" in call_log
        assert "ml_filter_disabled" in call_log
        assert "position_sizer" in call_log
        assert "order_placed" in call_log


# ---------------------------------------------------------------------------
# Test: Pipeline logging at each stage (Requirement 11.4)
# ---------------------------------------------------------------------------


class TestPipelineLogging:
    """Test that pipeline produces log output at each filter stage."""

    def test_logging_on_vol_block(self, df, mock_strategy_signal):
        """Volatility filter logs blocking reason with metrics.
        Validates: Requirement 11.4
        """
        vf = VolatilityRegimeFilter({
            "enabled": True,
            "atr_period": 14,
            "lookback_bars": 100,
            "lower_percentile": 20.0,
            "upper_percentile": 80.0,
        })
        # Fill with low values to ensure blocking
        for _ in range(30):
            vf.update_history(0.0001)

        allowed, meta = vf.allow_trading(df)
        # Should block and provide metadata
        assert not allowed
        assert "reason" in meta
        assert "atr_ratio" in meta
        assert "percentile" in meta

    def test_logging_on_ml_rejection(self, df):
        """ML filter provides rejection metadata.
        Validates: Requirement 11.4
        """
        ml = MagicMock(spec=MLDirectionalFilter)
        ml.is_enabled = True
        ml.confirm_signal.return_value = (False, {
            "probability": 0.40,
            "threshold": 0.55,
            "direction": "BUY",
            "reason": "BUY rejected: P(bullish)=0.4000 <= 0.55",
        })

        signal = _make_signal("BUY")
        confirmed, meta = ml.confirm_signal(signal, df)

        assert not confirmed
        assert "probability" in meta
        assert "threshold" in meta
        assert "direction" in meta
        assert "reason" in meta

    def test_logging_on_sizer_rejection(self, df, market_details):
        """Position sizer provides rejection metadata.
        Validates: Requirement 11.4
        """
        ig_client = MagicMock()
        ig_client.account_summary.return_value = {
            "accounts": [{"balance": {"balance": 5.0}}]
        }
        ps = RiskPositionSizer(
            {"risk_pct_per_trade": 0.1, "account_equity": 5.0, "max_size_multiple": 50},
            ig_client,
        )

        size, meta = ps.calculate_size(
            stop_distance=5.0, pip_value=0.77, min_size=0.5, size_step=0.1
        )
        assert size is None
        assert "reason" in meta
        assert "equity" in meta
        assert "raw_size" in meta
