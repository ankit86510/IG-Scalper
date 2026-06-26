"""Unit tests for SentimentFilter.confirm_signal() decision logic.

Tests the signal confirmation logic including BUY/SELL rules,
boundary conditions, neutral score behavior, and metadata structure.
"""

import pytest

from strategy.sentiment_filter import SentimentFilter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_filter(threshold: float = 0.3, enabled: bool = True) -> SentimentFilter:
    """Create a SentimentFilter with the given config."""
    return SentimentFilter({"sentiment_threshold": threshold, "enabled": enabled})


def _with_score(sf: SentimentFilter, score: float, sources: dict | None = None) -> SentimentFilter:
    """Monkey-patch get_sentiment_score to return a fixed score."""
    sources = sources or {}
    sf.get_sentiment_score = lambda: (score, sources)
    return sf


# ---------------------------------------------------------------------------
# BUY signal confirmation
# ---------------------------------------------------------------------------


class TestBuyConfirmation:
    """Tests for BUY signal decision logic."""

    def test_buy_confirmed_positive_score(self):
        """BUY with positive score (above -threshold) is confirmed."""
        sf = _with_score(_make_filter(threshold=0.3), 0.5)
        confirmed, meta = sf.confirm_signal({"side": "BUY"})
        assert confirmed is True

    def test_buy_confirmed_slightly_negative(self):
        """BUY with slightly negative score (above -threshold) is confirmed."""
        sf = _with_score(_make_filter(threshold=0.3), -0.2)
        confirmed, meta = sf.confirm_signal({"side": "BUY"})
        assert confirmed is True

    def test_buy_confirmed_at_boundary(self):
        """BUY with score exactly at -threshold is confirmed (>= boundary)."""
        sf = _with_score(_make_filter(threshold=0.3), -0.3)
        confirmed, meta = sf.confirm_signal({"side": "BUY"})
        assert confirmed is True

    def test_buy_rejected_below_threshold(self):
        """BUY with score below -threshold is rejected."""
        sf = _with_score(_make_filter(threshold=0.3), -0.5)
        confirmed, meta = sf.confirm_signal({"side": "BUY"})
        assert confirmed is False

    def test_buy_rejected_extreme_negative(self):
        """BUY with extremely negative score is rejected."""
        sf = _with_score(_make_filter(threshold=0.3), -1.0)
        confirmed, meta = sf.confirm_signal({"side": "BUY"})
        assert confirmed is False


# ---------------------------------------------------------------------------
# SELL signal confirmation
# ---------------------------------------------------------------------------


class TestSellConfirmation:
    """Tests for SELL signal decision logic."""

    def test_sell_confirmed_negative_score(self):
        """SELL with negative score (below +threshold) is confirmed."""
        sf = _with_score(_make_filter(threshold=0.3), -0.5)
        confirmed, meta = sf.confirm_signal({"side": "SELL"})
        assert confirmed is True

    def test_sell_confirmed_slightly_positive(self):
        """SELL with slightly positive score (below +threshold) is confirmed."""
        sf = _with_score(_make_filter(threshold=0.3), 0.2)
        confirmed, meta = sf.confirm_signal({"side": "SELL"})
        assert confirmed is True

    def test_sell_confirmed_at_boundary(self):
        """SELL with score exactly at +threshold is confirmed (<= boundary)."""
        sf = _with_score(_make_filter(threshold=0.3), 0.3)
        confirmed, meta = sf.confirm_signal({"side": "SELL"})
        assert confirmed is True

    def test_sell_rejected_above_threshold(self):
        """SELL with score above +threshold is rejected."""
        sf = _with_score(_make_filter(threshold=0.3), 0.5)
        confirmed, meta = sf.confirm_signal({"side": "SELL"})
        assert confirmed is False

    def test_sell_rejected_extreme_positive(self):
        """SELL with extremely positive score is rejected."""
        sf = _with_score(_make_filter(threshold=0.3), 1.0)
        confirmed, meta = sf.confirm_signal({"side": "SELL"})
        assert confirmed is False


# ---------------------------------------------------------------------------
# Neutral score (0.0) always confirms
# ---------------------------------------------------------------------------


class TestNeutralScore:
    """Score exactly 0.0 always confirms regardless of direction."""

    def test_neutral_confirms_buy(self):
        """Score 0.0 confirms BUY signal."""
        sf = _with_score(_make_filter(threshold=0.3), 0.0)
        confirmed, meta = sf.confirm_signal({"side": "BUY"})
        assert confirmed is True
        assert "neutral" in meta["sentiment_reason"]

    def test_neutral_confirms_sell(self):
        """Score 0.0 confirms SELL signal."""
        sf = _with_score(_make_filter(threshold=0.3), 0.0)
        confirmed, meta = sf.confirm_signal({"side": "SELL"})
        assert confirmed is True
        assert "neutral" in meta["sentiment_reason"]

    def test_neutral_confirms_with_tiny_threshold(self):
        """Score 0.0 confirms even with a very small threshold."""
        sf = _with_score(_make_filter(threshold=0.01), 0.0)
        confirmed, _ = sf.confirm_signal({"side": "BUY"})
        assert confirmed is True
        confirmed, _ = sf.confirm_signal({"side": "SELL"})
        assert confirmed is True


# ---------------------------------------------------------------------------
# Metadata structure
# ---------------------------------------------------------------------------


class TestMetadataStructure:
    """Return tuple[bool, dict] with required sentiment metadata keys."""

    def test_metadata_keys_present(self):
        """Metadata dict contains all required keys."""
        sf = _with_score(_make_filter(), 0.5, {"ig_client": 0.5})
        confirmed, meta = sf.confirm_signal({"side": "BUY"})

        required_keys = {
            "sentiment_score",
            "sentiment_threshold",
            "sentiment_confirmed",
            "sentiment_cache_hit",
            "sentiment_sources",
            "sentiment_reason",
        }
        assert required_keys.issubset(meta.keys())

    def test_metadata_values_match_state(self):
        """Metadata values accurately reflect the decision state."""
        sf = _with_score(_make_filter(threshold=0.3), -0.5, {"ig_client": -0.5})
        confirmed, meta = sf.confirm_signal({"side": "BUY"})

        assert confirmed is False
        assert meta["sentiment_score"] == -0.5
        assert meta["sentiment_threshold"] == 0.3
        assert meta["sentiment_confirmed"] is False
        assert meta["sentiment_cache_hit"] is False
        assert meta["sentiment_sources"] == {"ig_client": -0.5}
        assert "rejected" in meta["sentiment_reason"].lower()

    def test_return_type_is_tuple(self):
        """confirm_signal returns a tuple of (bool, dict)."""
        sf = _with_score(_make_filter(), 0.0)
        result = sf.confirm_signal({"side": "BUY"})
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], bool)
        assert isinstance(result[1], dict)


# ---------------------------------------------------------------------------
# Disabled filter pass-through
# ---------------------------------------------------------------------------


class TestDisabledFilter:
    """When disabled, confirm all signals without fetching sentiment."""

    def test_disabled_passes_buy(self):
        """Disabled filter confirms BUY without computing score."""
        sf = _make_filter(enabled=False)
        confirmed, meta = sf.confirm_signal({"side": "BUY"})
        assert confirmed is True
        assert meta["sentiment_reason"] == "sentiment filter disabled"

    def test_disabled_passes_sell(self):
        """Disabled filter confirms SELL without computing score."""
        sf = _make_filter(enabled=False)
        confirmed, meta = sf.confirm_signal({"side": "SELL"})
        assert confirmed is True


# ---------------------------------------------------------------------------
# Different threshold values
# ---------------------------------------------------------------------------


class TestCustomThresholds:
    """Tests with various threshold values."""

    def test_tight_threshold_rejects_more(self):
        """A tight threshold (0.1) rejects scores that a loose one (0.5) would accept."""
        sf_tight = _with_score(_make_filter(threshold=0.1), -0.2)
        sf_loose = _with_score(_make_filter(threshold=0.5), -0.2)

        confirmed_tight, _ = sf_tight.confirm_signal({"side": "BUY"})
        confirmed_loose, _ = sf_loose.confirm_signal({"side": "BUY"})

        assert confirmed_tight is False  # -0.2 < -0.1
        assert confirmed_loose is True   # -0.2 >= -0.5

    def test_default_threshold_is_0_3(self):
        """Default threshold is 0.3 when not specified in config."""
        sf = SentimentFilter({})
        assert sf._threshold == 0.3


# ---------------------------------------------------------------------------
# Error handling and edge cases (Task 6.3)
# ---------------------------------------------------------------------------

import time
from unittest.mock import MagicMock, patch, PropertyMock

import requests as requests_lib

from strategy.sentiment_filter import (
    IGClientSentimentSource,
    AlphaVantageNewsSource,
)


class TestIGAPITimeout:
    """IG API timeout returns neutral score. (Req 7.1)"""

    def test_ig_timeout_returns_none(self):
        """When IG session.get raises Timeout, fetch_score returns None."""
        mock_ig_client = MagicMock()
        mock_ig_client.base = "https://demo-api.ig.com/gateway/deal"
        mock_ig_client._hv.return_value = {"Version": "1"}
        mock_ig_client.verify_ssl = False
        mock_ig_client.s.get.side_effect = requests_lib.exceptions.Timeout(
            "Connection timed out"
        )

        source = IGClientSentimentSource(mock_ig_client, market_id="GOLD")
        result = source.fetch_score()

        assert result is None

    def test_ig_timeout_overall_score_neutral(self):
        """When IG times out, overall sentiment score is neutral (0.0)."""
        mock_ig_client = MagicMock()
        mock_ig_client.base = "https://demo-api.ig.com/gateway/deal"
        mock_ig_client._hv.return_value = {"Version": "1"}
        mock_ig_client.verify_ssl = False
        mock_ig_client.s.get.side_effect = requests_lib.exceptions.Timeout(
            "Connection timed out"
        )

        config = {
            "enabled": True,
            "sentiment_threshold": 0.3,
            "cache_ttl_seconds": 300,
            "sources": {"ig_client": {"enabled": True, "market_id": "GOLD"}},
            "source_weights": {"ig_client": 1.0},
        }
        sf = SentimentFilter(config, ig_client=mock_ig_client)
        score, details = sf.get_sentiment_score()

        # IG returned None, so aggregated score should be 0.0
        assert score == 0.0
        assert details.get("ig_client") is None


class TestNewsAPIFailure:
    """News API failure excludes source from aggregation. (Req 7.2)"""

    def test_news_connection_error_returns_none(self):
        """When requests.get raises ConnectionError, news source returns None."""
        source = AlphaVantageNewsSource(
            api_key="test-key", lookback_hours=4, min_articles=3
        )
        with patch("strategy.sentiment_filter.requests.get") as mock_get:
            mock_get.side_effect = requests_lib.exceptions.ConnectionError(
                "Connection refused"
            )
            result = source.fetch_score()

        assert result is None

    def test_news_failure_ig_score_used_alone(self):
        """When news fails but IG works, IG score is used as the overall score."""
        mock_ig_client = MagicMock()
        mock_ig_client.base = "https://demo-api.ig.com/gateway/deal"
        mock_ig_client._hv.return_value = {"Version": "1"}
        mock_ig_client.verify_ssl = False

        # IG returns a valid response with 80% long (contrarian bearish)
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            "longPositionPercentage": 80.0,
            "shortPositionPercentage": 20.0,
        }
        mock_ig_client.s.get.return_value = mock_response

        config = {
            "enabled": True,
            "sentiment_threshold": 0.3,
            "cache_ttl_seconds": 300,
            "sources": {
                "ig_client": {"enabled": True, "market_id": "GOLD"},
                "news": {"enabled": True, "api_key": "test-key"},
            },
            "source_weights": {"ig_client": 0.6, "news": 0.4},
        }

        sf = SentimentFilter(config, ig_client=mock_ig_client)

        # Mock the news source to raise an exception
        for source in sf._sources:
            if source.name == "news":
                source.fetch_score = MagicMock(
                    side_effect=requests_lib.exceptions.ConnectionError("fail")
                )

        score, details = sf.get_sentiment_score()

        # IG score: -(80-50)/50 = -0.6, news is excluded → score = -0.6
        assert score == pytest.approx(-0.6, abs=0.01)
        assert details.get("ig_client") is not None
        assert details.get("news") is None


class TestFirstCallPassthrough:
    """First call after startup passes through until first fetch succeeds. (Req 7.3)"""

    def test_first_call_passes_through(self):
        """Before first successful fetch, confirm_signal returns True."""
        mock_ig_client = MagicMock()
        mock_ig_client.base = "https://demo-api.ig.com/gateway/deal"
        mock_ig_client._hv.return_value = {"Version": "1"}
        mock_ig_client.verify_ssl = False
        # IG returns None (all sources fail)
        mock_ig_client.s.get.side_effect = requests_lib.exceptions.Timeout("timeout")

        config = {
            "enabled": True,
            "sentiment_threshold": 0.3,
            "cache_ttl_seconds": 300,
            "sources": {"ig_client": {"enabled": True, "market_id": "GOLD"}},
            "source_weights": {"ig_client": 1.0},
        }
        sf = SentimentFilter(config, ig_client=mock_ig_client)

        # Sources are configured, but haven't fetched successfully yet
        assert sf._has_fetched_successfully is False

        confirmed, meta = sf.confirm_signal({"side": "BUY"})

        assert confirmed is True
        assert "awaiting first sentiment data" in meta["sentiment_reason"]

    def test_no_sources_means_already_fetched(self):
        """When no sources are configured, _has_fetched_successfully is True."""
        sf = SentimentFilter({"enabled": True})
        # No sources configured → treated as "already fetched"
        assert sf._has_fetched_successfully is True


class TestInitializationFailure:
    """Initialization failure results in permanent pass-through. (Req 7.4)"""

    def test_invalid_config_triggers_permanent_passthrough(self):
        """Config that causes initialization exception → permanent pass-through."""
        # Force an exception during source initialization by providing
        # sources config with a type that causes an error during processing
        config = {
            "enabled": True,
            "sentiment_threshold": 0.3,
            "sources": {
                "ig_client": {"enabled": True, "market_id": "GOLD"},
            },
        }

        # Pass an ig_client whose attribute access causes an exception
        # during IGClientSentimentSource initialization
        bad_ig_client = MagicMock()
        # Make the IGClientSentimentSource constructor fail by making
        # the sources config trigger an exception
        with patch(
            "strategy.sentiment_filter.IGClientSentimentSource",
            side_effect=RuntimeError("init failed"),
        ):
            sf = SentimentFilter(config, ig_client=bad_ig_client)

        assert sf._permanent_passthrough is True
        assert sf._sources == []

    def test_permanent_passthrough_confirms_signal(self):
        """In permanent pass-through mode, confirm_signal always returns True."""
        config = {
            "enabled": True,
            "sentiment_threshold": 0.3,
            "sources": {
                "ig_client": {"enabled": True, "market_id": "GOLD"},
            },
        }

        with patch(
            "strategy.sentiment_filter.IGClientSentimentSource",
            side_effect=RuntimeError("init failed"),
        ):
            sf = SentimentFilter(config, ig_client=MagicMock())

        confirmed, meta = sf.confirm_signal({"side": "SELL"})

        assert confirmed is True
        assert "permanent pass-through" in meta["sentiment_reason"]


class TestConfigDefaults:
    """Config defaults are used when keys are missing. (Req 5.3, 5.4, 5.5)"""

    def test_empty_config_defaults(self):
        """Empty config produces correct default values."""
        sf = SentimentFilter({})

        assert sf._threshold == 0.3
        assert sf._enabled is True
        assert sf._cache_ttl == 300

    def test_empty_config_no_sources(self):
        """Empty config initializes with no sources."""
        sf = SentimentFilter({})
        assert sf._sources == []

    def test_partial_config_fills_defaults(self):
        """Partial config uses defaults for missing keys."""
        sf = SentimentFilter({"sentiment_threshold": 0.5})

        assert sf._threshold == 0.5
        assert sf._enabled is True  # default
        assert sf._cache_ttl == 300  # default


class TestRateLimiter:
    """Rate limiter respects per-source limits for IG. (Req 5.3)"""

    def test_ig_rate_limited_after_first_fetch(self):
        """IG source is rate-limited after a successful fetch (within 5 min window)."""
        mock_ig_client = MagicMock()
        mock_ig_client.base = "https://demo-api.ig.com/gateway/deal"
        mock_ig_client._hv.return_value = {"Version": "1"}
        mock_ig_client.verify_ssl = False

        # IG returns a valid response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            "longPositionPercentage": 50.0,
            "shortPositionPercentage": 50.0,
        }
        mock_ig_client.s.get.return_value = mock_response

        config = {
            "enabled": True,
            "sentiment_threshold": 0.3,
            "cache_ttl_seconds": 1,  # Very short TTL so cache expires quickly
            "sources": {"ig_client": {"enabled": True, "market_id": "GOLD"}},
            "source_weights": {"ig_client": 1.0},
        }
        sf = SentimentFilter(config, ig_client=mock_ig_client)

        # First fetch → should call IG API
        score1, details1 = sf.get_sentiment_score()
        assert details1.get("ig_client") == 0.0  # neutral: 50/50
        assert mock_ig_client.s.get.call_count == 1

        # Wait for cache to expire
        time.sleep(1.1)

        # Second fetch → IG should be rate-limited (within 5 min window)
        score2, details2 = sf.get_sentiment_score()

        # IG should return None due to rate limiting (no second API call)
        assert details2.get("ig_client") is None
        # Still only 1 call to the API
        assert mock_ig_client.s.get.call_count == 1

    def test_ig_not_rate_limited_on_first_call(self):
        """IG source is NOT rate-limited on the very first call."""
        mock_ig_client = MagicMock()
        mock_ig_client.base = "https://demo-api.ig.com/gateway/deal"
        mock_ig_client._hv.return_value = {"Version": "1"}
        mock_ig_client.verify_ssl = False

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            "longPositionPercentage": 75.0,
            "shortPositionPercentage": 25.0,
        }
        mock_ig_client.s.get.return_value = mock_response

        config = {
            "enabled": True,
            "sentiment_threshold": 0.3,
            "cache_ttl_seconds": 300,
            "sources": {"ig_client": {"enabled": True, "market_id": "GOLD"}},
            "source_weights": {"ig_client": 1.0},
        }
        sf = SentimentFilter(config, ig_client=mock_ig_client)

        # First call should pass through (not rate limited)
        assert sf._is_ig_rate_limited() is False

        score, details = sf.get_sentiment_score()

        # Should have called the API
        assert mock_ig_client.s.get.call_count == 1
        # IG score: -(75-50)/50 = -0.5
        assert details.get("ig_client") == pytest.approx(-0.5, abs=0.01)
