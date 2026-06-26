"""
Integration tests for the Sentiment Filter pipeline ordering.

Tests verify the execution order and gating logic of:
  ML filter → Sentiment filter → Position sizer

These tests exercise the filter components in sequence (not the actual runner loop),
verifying that the pipeline ordering behaves correctly.

Requirements validated: 8.1, 8.2, 8.3, 8.5, 9.4
"""

from unittest.mock import MagicMock, patch
import numpy as np
import pandas as pd
import pytest
from zoneinfo import ZoneInfo

from strategy.ml_filter import MLDirectionalFilter
from strategy.sentiment_filter import SentimentFilter
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


def _pipeline_run_with_sentiment(
    df: pd.DataFrame,
    signal: dict,
    ml_filter: MLDirectionalFilter,
    sentiment_filter: SentimentFilter,
    position_sizer: RiskPositionSizer | None,
    market_details: dict | None = None,
    call_log: list | None = None,
) -> str:
    """
    Simulate the pipeline flow as in run_ai_autonomous.py (ML → Sentiment → Sizer).

    Records each component call to call_log for order verification.
    Returns the final outcome: 'order_placed', 'ml_rejected', 'sentiment_rejected',
    'sizer_rejected'.
    """
    if call_log is None:
        call_log = []
    if market_details is None:
        market_details = {
            "dealingRules": {
                "minDealSize": {"value": 0.5},
                "minNormalStopOrLimitDistance": {"value": 1.0},
            }
        }

    # 1. ML Filter
    if ml_filter.is_enabled:
        confirmed, ml_meta = ml_filter.confirm_signal(signal, df)
        call_log.append("ml_filter")
        if not confirmed:
            return "ml_rejected"
    else:
        call_log.append("ml_filter_disabled")

    # 2. Sentiment Filter (after ML, before position sizer)
    sent_confirmed, sentiment_meta = sentiment_filter.confirm_signal(signal, df)
    call_log.append("sentiment_filter")
    # Append sentiment metadata to signal's meta dict (as runner does)
    signal.setdefault("meta", {}).update({"sentiment": sentiment_meta})
    if not sent_confirmed:
        return "sentiment_rejected"

    # 3. Position sizer
    if position_sizer is not None:
        size, size_meta = position_sizer.calculate_size(
            stop_distance=signal["stop_pts"],
            pip_value=0.77,
            min_size=market_details["dealingRules"]["minDealSize"]["value"],
            size_step=0.1,
        )
        call_log.append("position_sizer")
        if size is None:
            return "sizer_rejected"

    call_log.append("order_placed")
    return "order_placed"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def df():
    """A valid OHLC DataFrame with enough history."""
    return _make_ohlc_df(rows=120)


@pytest.fixture
def ml_filter_confirms():
    """An ML filter mock that confirms signals."""
    ml = MagicMock(spec=MLDirectionalFilter)
    ml.is_enabled = True
    ml.confirm_signal.return_value = (True, {
        "probability": 0.72,
        "threshold": 0.55,
        "direction": "BUY",
        "reason": "BUY confirmed: P(bullish)=0.7200 > 0.55",
        "ml_enabled": True,
        "confirmed": True,
    })
    return ml


@pytest.fixture
def ml_filter_rejects():
    """An ML filter mock that rejects signals."""
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
def sentiment_filter_confirms():
    """A SentimentFilter that confirms all signals (neutral score)."""
    sf = SentimentFilter(config={"enabled": True, "sentiment_threshold": 0.3}, ig_client=None)
    # No sources configured means score=0.0 → always confirms
    return sf


@pytest.fixture
def sentiment_filter_rejects():
    """A SentimentFilter that rejects signals (very negative score)."""
    sf = SentimentFilter(config={"enabled": True, "sentiment_threshold": 0.3}, ig_client=None)
    # Monkey-patch get_sentiment_score to return a very negative score
    sf.get_sentiment_score = lambda: (-0.8, {"ig_client": -0.8})
    sf._has_fetched_successfully = True
    return sf


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


# ---------------------------------------------------------------------------
# TestPipelineOrdering
# ---------------------------------------------------------------------------


class TestPipelineOrdering:
    """Test correct ordering: ML → Sentiment → Position Sizer.
    Validates: Requirements 8.1, 8.2, 8.3
    """

    def test_ml_confirms_then_sentiment_called(
        self, df, ml_filter_confirms, sentiment_filter_confirms, position_sizer_pass
    ):
        """ML confirms → sentiment is called → sizer is called.
        Validates: Requirements 8.1, 8.3
        """
        signal = _make_signal("BUY")
        call_log = []

        result = _pipeline_run_with_sentiment(
            df=df,
            signal=signal,
            ml_filter=ml_filter_confirms,
            sentiment_filter=sentiment_filter_confirms,
            position_sizer=position_sizer_pass,
            call_log=call_log,
        )

        assert result == "order_placed"
        assert call_log == ["ml_filter", "sentiment_filter", "position_sizer", "order_placed"]
        # Verify ordering
        assert call_log.index("ml_filter") < call_log.index("sentiment_filter")
        assert call_log.index("sentiment_filter") < call_log.index("position_sizer")

    def test_ml_rejects_then_sentiment_not_called(
        self, df, ml_filter_rejects, sentiment_filter_confirms, position_sizer_pass
    ):
        """ML rejects → sentiment is NOT called → sizer is NOT called.
        Validates: Requirement 8.2
        """
        signal = _make_signal("BUY")
        call_log = []

        result = _pipeline_run_with_sentiment(
            df=df,
            signal=signal,
            ml_filter=ml_filter_rejects,
            sentiment_filter=sentiment_filter_confirms,
            position_sizer=position_sizer_pass,
            call_log=call_log,
        )

        assert result == "ml_rejected"
        assert "ml_filter" in call_log
        assert "sentiment_filter" not in call_log
        assert "position_sizer" not in call_log
        assert "order_placed" not in call_log

    def test_sentiment_confirms_then_proceeds_to_sizer(
        self, df, ml_filter_confirms, sentiment_filter_confirms, position_sizer_pass
    ):
        """ML confirms, sentiment confirms → position sizer is reached.
        Validates: Requirement 8.3
        """
        signal = _make_signal("BUY")
        call_log = []

        result = _pipeline_run_with_sentiment(
            df=df,
            signal=signal,
            ml_filter=ml_filter_confirms,
            sentiment_filter=sentiment_filter_confirms,
            position_sizer=position_sizer_pass,
            call_log=call_log,
        )

        assert result == "order_placed"
        assert "position_sizer" in call_log


# ---------------------------------------------------------------------------
# TestEndToEnd
# ---------------------------------------------------------------------------


class TestEndToEnd:
    """End-to-end tests with mocked sentiment sources.
    Validates: Requirements 8.1, 8.3, 8.5
    """

    def test_full_signal_flow_mocked_apis(self, df, ml_filter_confirms, position_sizer_pass):
        """Full mini-pipeline with mocked sources:
        ML confirms → IG sentiment (neutral) → news (None) → signal confirmed with metadata.
        Validates: Requirements 8.1, 8.3, 8.5
        """
        # Create a SentimentFilter with mocked sources
        sf = SentimentFilter(
            config={"enabled": True, "sentiment_threshold": 0.3},
            ig_client=None,
        )

        # Mock IG source returning 50/50 (neutral → 0.0)
        mock_ig_source = MagicMock()
        mock_ig_source.name = "ig_client"
        mock_ig_source.fetch_score.return_value = 0.0

        # Mock news source returning None (not enough articles)
        mock_news_source = MagicMock()
        mock_news_source.name = "news"
        mock_news_source.fetch_score.return_value = None

        sf._sources = [mock_ig_source, mock_news_source]
        sf._has_fetched_successfully = False  # Reset so first fetch marks success

        signal = _make_signal("BUY")
        call_log = []

        result = _pipeline_run_with_sentiment(
            df=df,
            signal=signal,
            ml_filter=ml_filter_confirms,
            sentiment_filter=sf,
            position_sizer=position_sizer_pass,
            call_log=call_log,
        )

        assert result == "order_placed"
        # Verify both ML and sentiment metadata are in signal
        assert "sentiment" in signal["meta"]
        sentiment_meta = signal["meta"]["sentiment"]
        assert "sentiment_score" in sentiment_meta
        assert "sentiment_sources" in sentiment_meta
        assert sentiment_meta["sentiment_confirmed"] is True

        # Verify sources were called
        mock_ig_source.fetch_score.assert_called_once()
        mock_news_source.fetch_score.assert_called_once()

    def test_sentiment_rejection_stops_pipeline(
        self, df, ml_filter_confirms, position_sizer_pass
    ):
        """ML confirms, sentiment rejects (very negative score) → sizer NOT called.
        Validates: Requirements 8.3, 8.5
        """
        # Create a SentimentFilter with a strongly negative IG score
        sf = SentimentFilter(
            config={"enabled": True, "sentiment_threshold": 0.3},
            ig_client=None,
        )

        # Mock IG source returning -0.8 (strong bearish sentiment for a BUY signal)
        mock_ig_source = MagicMock()
        mock_ig_source.name = "ig_client"
        mock_ig_source.fetch_score.return_value = -0.8

        sf._sources = [mock_ig_source]
        sf._has_fetched_successfully = True

        signal = _make_signal("BUY")
        call_log = []

        result = _pipeline_run_with_sentiment(
            df=df,
            signal=signal,
            ml_filter=ml_filter_confirms,
            sentiment_filter=sf,
            position_sizer=position_sizer_pass,
            call_log=call_log,
        )

        assert result == "sentiment_rejected"
        assert "ml_filter" in call_log
        assert "sentiment_filter" in call_log
        assert "position_sizer" not in call_log
        assert "order_placed" not in call_log

    def test_sell_signal_flow_with_positive_sentiment_rejects(
        self, df, ml_filter_confirms, position_sizer_pass
    ):
        """SELL signal with positive sentiment score > threshold → rejected.
        Validates: Requirements 8.3, 8.5
        """
        # ML filter confirms SELL
        ml_filter_confirms.confirm_signal.return_value = (True, {
            "probability": 0.30,
            "threshold": 0.55,
            "direction": "SELL",
            "reason": "SELL confirmed: P(bearish)=0.7000 > 0.55",
            "ml_enabled": True,
            "confirmed": True,
        })

        # Create sentiment filter with positive score (bullish) → rejects SELL
        sf = SentimentFilter(
            config={"enabled": True, "sentiment_threshold": 0.3},
            ig_client=None,
        )
        mock_ig_source = MagicMock()
        mock_ig_source.name = "ig_client"
        mock_ig_source.fetch_score.return_value = 0.8  # bullish → rejects SELL

        sf._sources = [mock_ig_source]
        sf._has_fetched_successfully = True

        signal = _make_signal("SELL")
        call_log = []

        result = _pipeline_run_with_sentiment(
            df=df,
            signal=signal,
            ml_filter=ml_filter_confirms,
            sentiment_filter=sf,
            position_sizer=position_sizer_pass,
            call_log=call_log,
        )

        assert result == "sentiment_rejected"


# ---------------------------------------------------------------------------
# TestAnalyticsDecisionLog
# ---------------------------------------------------------------------------


class TestAnalyticsDecisionLog:
    """Test that sentiment metadata is appended to the signal for analytics.
    Validates: Requirements 8.5, 9.4
    """

    def test_metadata_appended_to_signal(self, df, ml_filter_confirms, position_sizer_pass):
        """Run a signal through ML+sentiment, verify signal['meta']['sentiment']
        contains the sentiment metadata dict.
        Validates: Requirements 8.5, 9.4
        """
        # Create sentiment filter with real sources mocked
        sf = SentimentFilter(
            config={
                "enabled": True,
                "sentiment_threshold": 0.3,
                "analytics": {"save_all_decisions": True},
            },
            ig_client=None,
        )

        mock_ig_source = MagicMock()
        mock_ig_source.name = "ig_client"
        mock_ig_source.fetch_score.return_value = 0.1  # slightly bullish

        sf._sources = [mock_ig_source]
        sf._has_fetched_successfully = True

        signal = _make_signal("BUY")
        call_log = []

        result = _pipeline_run_with_sentiment(
            df=df,
            signal=signal,
            ml_filter=ml_filter_confirms,
            sentiment_filter=sf,
            position_sizer=position_sizer_pass,
            call_log=call_log,
        )

        assert result == "order_placed"

        # Verify sentiment metadata is in signal
        assert "sentiment" in signal["meta"]
        sentiment_meta = signal["meta"]["sentiment"]

        # Verify required metadata keys exist
        assert "sentiment_score" in sentiment_meta
        assert "sentiment_threshold" in sentiment_meta
        assert "sentiment_confirmed" in sentiment_meta
        assert "sentiment_cache_hit" in sentiment_meta
        assert "sentiment_sources" in sentiment_meta
        assert "sentiment_reason" in sentiment_meta

        # Verify values make sense
        assert sentiment_meta["sentiment_confirmed"] is True
        assert sentiment_meta["sentiment_score"] == pytest.approx(0.1, abs=0.01)
        assert sentiment_meta["sentiment_threshold"] == 0.3
        assert "ig_client" in sentiment_meta["sentiment_sources"]

    def test_rejection_metadata_also_appended(self, df, ml_filter_confirms):
        """Even when sentiment rejects, metadata is still appended to signal.
        Validates: Requirement 9.4
        """
        sf = SentimentFilter(
            config={"enabled": True, "sentiment_threshold": 0.3},
            ig_client=None,
        )

        mock_ig_source = MagicMock()
        mock_ig_source.name = "ig_client"
        mock_ig_source.fetch_score.return_value = -0.9  # very bearish → rejects BUY

        sf._sources = [mock_ig_source]
        sf._has_fetched_successfully = True

        signal = _make_signal("BUY")
        call_log = []

        result = _pipeline_run_with_sentiment(
            df=df,
            signal=signal,
            ml_filter=ml_filter_confirms,
            sentiment_filter=sf,
            position_sizer=None,
            call_log=call_log,
        )

        assert result == "sentiment_rejected"

        # Metadata should still be appended even on rejection
        assert "sentiment" in signal["meta"]
        sentiment_meta = signal["meta"]["sentiment"]
        assert sentiment_meta["sentiment_confirmed"] is False
        assert sentiment_meta["sentiment_score"] == pytest.approx(-0.9, abs=0.01)
        assert "rejected" in sentiment_meta["sentiment_reason"].lower()

    def test_decision_log_includes_sentiment_when_analytics_enabled(
        self, df, ml_filter_confirms, position_sizer_pass
    ):
        """When analytics.save_all_decisions is true, metadata includes analytics flag.
        Validates: Requirement 9.4
        """
        sf = SentimentFilter(
            config={
                "enabled": True,
                "sentiment_threshold": 0.3,
                "analytics": {"save_all_decisions": True},
            },
            ig_client=None,
        )

        mock_ig_source = MagicMock()
        mock_ig_source.name = "ig_client"
        mock_ig_source.fetch_score.return_value = 0.05

        sf._sources = [mock_ig_source]
        sf._has_fetched_successfully = True

        signal = _make_signal("BUY")
        call_log = []

        _pipeline_run_with_sentiment(
            df=df,
            signal=signal,
            ml_filter=ml_filter_confirms,
            sentiment_filter=sf,
            position_sizer=position_sizer_pass,
            call_log=call_log,
        )

        sentiment_meta = signal["meta"]["sentiment"]
        assert sentiment_meta.get("sentiment_analytics") is True
