"""Sentiment Analysis Filter for Gold trading signals.

This module provides a sentiment-based signal filter that evaluates market
sentiment from multiple sources (IG client positioning, news APIs) and
confirms or rejects trade signals based on sentiment alignment.

Integrates into the pipeline after the ML Directional Filter, following
the same confirm_signal() interface and fail-open philosophy.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Protocol

import requests

if TYPE_CHECKING:
    from broker.ig_client import IGClient

logger = logging.getLogger(__name__)


class SentimentSource(Protocol):
    """Protocol defining the interface for sentiment data providers.

    All sentiment sources must implement this protocol to be used
    by the SentimentFilter aggregation system.
    """

    @property
    def name(self) -> str:
        """Unique identifier for this sentiment source."""
        ...

    def fetch_score(self) -> float | None:
        """Fetch current sentiment score.

        Returns:
            Sentiment score in [-1.0, +1.0], or None if unavailable.
            Must handle its own timeouts and errors internally.
        """
        ...


@dataclass
class CacheEntry:
    """Stores a cached sentiment score with metadata.

    Attributes:
        score: The cached sentiment score in [-1.0, +1.0].
        source_details: Dict mapping source names to their individual scores.
        timestamp: Unix timestamp (time.time()) when the entry was created.
        ttl_seconds: Time-to-live in seconds for this cache entry.
    """

    score: float
    source_details: dict
    timestamp: float
    ttl_seconds: int


class SentimentCache:
    """Simple in-memory TTL cache for sentiment data.

    Stores the most recent sentiment score and source details with
    a configurable TTL to avoid redundant API calls.
    """

    def __init__(self, ttl_seconds: int = 300) -> None:
        """Initialize the cache.

        Args:
            ttl_seconds: Time-to-live for cached entries (default 300s / 5 min).
        """
        self._ttl_seconds = ttl_seconds
        self._entry: CacheEntry | None = None

    def get(self) -> tuple[float, dict] | None:
        """Retrieve cached sentiment data if still valid.

        Returns:
            Tuple of (score, source_details) if cache is valid, else None.
        """
        if self._entry is None:
            return None
        if not self.is_valid():
            return None
        return (self._entry.score, self._entry.source_details)

    def set(self, score: float, details: dict) -> None:
        """Store a new sentiment score in the cache.

        Args:
            score: Aggregated sentiment score in [-1.0, +1.0].
            details: Dict of source names to their individual scores.
        """
        self._entry = CacheEntry(
            score=score,
            source_details=details,
            timestamp=time.time(),
            ttl_seconds=self._ttl_seconds,
        )

    def is_valid(self) -> bool:
        """Check if the cache contains data within the TTL window.

        Returns:
            True if cache has data and it's still within TTL, else False.
        """
        if self._entry is None:
            return False
        elapsed = time.time() - self._entry.timestamp
        return elapsed < self._entry.ttl_seconds


class IGClientSentimentSource:
    """Fetches IG client positioning and applies contrarian interpretation.

    Uses the IG REST API `/clientsentiment/{marketId}` endpoint to retrieve
    the percentage of retail clients holding long vs short positions.
    Applies contrarian logic: extreme retail long positioning is interpreted
    as bearish, and extreme short positioning as bullish.
    """

    def __init__(self, ig_client: IGClient, market_id: str = "GOLD") -> None:
        """Initialize the IG client sentiment source.

        Args:
            ig_client: Authenticated IGClient instance with active session.
            market_id: IG market identifier for sentiment lookup (default "GOLD").
        """
        self._ig_client = ig_client
        self._market_id = market_id

    @property
    def name(self) -> str:
        """Unique identifier for this sentiment source."""
        return "ig_client"

    def fetch_score(self) -> float | None:
        """Fetch IG client sentiment and apply contrarian logic.

        Contrarian interpretation:
        - long_pct > 70%: score = -(long_pct - 50) / 50  (bearish)
        - short_pct > 70%: score = +(short_pct - 50) / 50 (bullish)
        - Both in [30%, 70%]: score = 0.0 (neutral)

        Returns:
            Sentiment score in [-1.0, +1.0], or None on error/timeout.
        """
        try:
            url = f"{self._ig_client.base}/clientsentiment/{self._market_id}"
            response = self._ig_client.s.get(
                url,
                headers=self._ig_client._hv("1"),
                timeout=10,
                verify=self._ig_client.verify_ssl,
            )
            response.raise_for_status()
            data = response.json()

            long_pct = float(data["longPositionPercentage"])
            short_pct = float(data["shortPositionPercentage"])

            if long_pct > 70:
                return -(long_pct - 50) / 50
            elif short_pct > 70:
                return +(short_pct - 50) / 50
            else:
                return 0.0

        except Exception as exc:
            logger.warning(
                "IGClientSentimentSource: failed to fetch sentiment for %s: %s",
                self._market_id,
                exc,
            )
            return None


class WeightedAggregator:
    """Combines multiple sentiment source scores using configurable weights.

    Computes a weighted average of all non-None scores, re-normalized by the
    sum of active weights. Returns 0.0 when all sources are unavailable, and
    clamps the final result to [-1.0, +1.0].
    """

    def __init__(self, weights: dict[str, float]) -> None:
        """Initialize the aggregator with source weights.

        Args:
            weights: Dict mapping source names to their weights.
                     e.g. {"ig_client": 0.6, "news": 0.4}
        """
        self._weights = weights

    def aggregate(self, scores: dict[str, float | None]) -> float:
        """Compute the weighted average of non-None scores.

        Filters out sources with None scores, computes the weighted average
        using only active sources, re-normalizes by the sum of active weights,
        and clamps the result to [-1.0, +1.0].

        Args:
            scores: Dict mapping source names to their scores.
                    Values may be None if a source is unavailable.

        Returns:
            Aggregated sentiment score in [-1.0, +1.0].
            Returns 0.0 if all sources are None.
        """
        weighted_sum = 0.0
        active_weight_sum = 0.0

        for source_name, score in scores.items():
            if score is None:
                continue
            weight = self._weights.get(source_name, 0.0)
            weighted_sum += score * weight
            active_weight_sum += weight

        if active_weight_sum == 0.0:
            return 0.0

        result = weighted_sum / active_weight_sum
        return max(-1.0, min(1.0, result))



class SentimentFilter:
    """Main orchestrator for sentiment-based signal confirmation.

    Evaluates market sentiment and confirms or rejects trade signals
    based on sentiment alignment with the proposed direction. Follows
    the same confirm_signal() interface as MLDirectionalFilter.
    """

    # IG rate limit: 1 request per 5 minutes (300 seconds)
    _IG_RATE_LIMIT_SECONDS = 300

    def __init__(self, config: dict, ig_client: "IGClient | None" = None) -> None:
        """Initialize the SentimentFilter.

        Args:
            config: Configuration dict (from settings_ai.yaml sentiment_filter section).
                    Keys: enabled, sentiment_threshold, cache_ttl_seconds, sources, source_weights.
            ig_client: Optional authenticated IGClient instance for IG sentiment source.
        """
        self._config = config
        self._ig_client = ig_client
        self._enabled = config.get("enabled", True)
        self._threshold = config.get("sentiment_threshold", 0.3)
        self._cache_ttl = config.get("cache_ttl_seconds", 300)

        # Initialize cache
        self._cache = SentimentCache(ttl_seconds=self._cache_ttl)

        # Track whether we've ever successfully fetched sentiment data
        self._has_fetched_successfully = False

        # Permanent pass-through mode (set on initialization failure)
        self._permanent_passthrough = False

        # IG rate limit tracking (1 request per 5 minutes)
        self._ig_last_fetch_time: float = 0.0

        # Initialize sources based on config
        self._sources: list = []

        try:
            sources_config = config.get("sources", {})

            # IG Client Sentiment Source
            ig_source_config = sources_config.get("ig_client", {})
            if ig_source_config.get("enabled", False) and ig_client is not None:
                market_id = ig_source_config.get("market_id", "GOLD")
                self._sources.append(IGClientSentimentSource(ig_client, market_id=market_id))

            # Alpha Vantage News Source
            news_source_config = sources_config.get("news", {})
            if news_source_config.get("enabled", False):
                raw_api_key = news_source_config.get("api_key", "")
                # Resolve env var expansion
                api_key = self._resolve_env_var(raw_api_key)
                lookback_hours = news_source_config.get("lookback_hours", 4)
                min_articles = news_source_config.get("min_articles", 3)
                timeout_seconds = news_source_config.get("timeout_seconds", 15)
                max_requests_per_hour = news_source_config.get("max_requests_per_hour", 5)
                self._sources.append(
                    AlphaVantageNewsSource(
                        api_key=api_key,
                        lookback_hours=lookback_hours,
                        min_articles=min_articles,
                        timeout_seconds=timeout_seconds,
                        max_requests_per_hour=max_requests_per_hour,
                    )
                )

            # ForexFactory Economic Calendar Source (always enabled, no API key needed)
            ff_config = sources_config.get("forex_factory", {})
            if ff_config.get("enabled", True):  # Enabled by default
                proximity_hours = ff_config.get("proximity_hours", 2)
                self._sources.append(
                    ForexFactoryCalendarSource(
                        proximity_hours=proximity_hours,
                        timeout_seconds=ff_config.get("timeout_seconds", 10),
                    )
                )

        except Exception as exc:
            logger.error(
                "SentimentFilter initialization failed, entering permanent pass-through mode: %s: %s",
                type(exc).__name__,
                exc,
            )
            self._permanent_passthrough = True
            self._sources = []

        # Initialize weighted aggregator
        default_weights = {"ig_client": 0.4, "news": 0.3, "forex_factory": 0.3}
        weights = config.get("source_weights", default_weights)
        self._aggregator = WeightedAggregator(weights=weights)

        # If no sources are configured, there's nothing to await —
        # treat as "already fetched" so confirm_signal uses normal logic
        if not self._sources:
            self._has_fetched_successfully = True

    @staticmethod
    def _resolve_env_var(value: str) -> str:
        """Resolve ${VAR} env var expansion in a config value.

        Args:
            value: Raw string, possibly with ${VAR} syntax.

        Returns:
            Resolved value, or empty string if env var not found.
        """
        if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
            var_name = value[2:-1]
            return os.environ.get(var_name, "")
        return value

    @property
    def is_enabled(self) -> bool:
        """Whether the sentiment filter is active."""
        return self._enabled

    def _is_ig_rate_limited(self) -> bool:
        """Check if the IG source is rate-limited (1 req per 5 min).

        Returns:
            True if less than 5 minutes have passed since last IG fetch.
        """
        if self._ig_last_fetch_time == 0.0:
            return False
        elapsed = time.time() - self._ig_last_fetch_time
        return elapsed < self._IG_RATE_LIMIT_SECONDS

    def get_sentiment_score(self) -> tuple[float, dict]:
        """Fetch/compute aggregated sentiment score.

        Coordinates the cache check → source fetch → aggregation flow.
        Respects rate limits per source (IG: 1/5min, Alpha Vantage: 5/hour).

        Returns:
            Tuple of (score in [-1.0, +1.0], source_details dict).
        """
        # Check cache first
        cached = self._cache.get()
        if cached is not None:
            return cached

        # Fetch from all initialized sources, respecting rate limits
        scores: dict[str, float | None] = {}

        for source in self._sources:
            try:
                # Apply IG-specific rate limit
                if source.name == "ig_client" and self._is_ig_rate_limited():
                    logger.debug(
                        "SentimentFilter: IG source rate-limited, skipping fetch"
                    )
                    scores[source.name] = None
                    continue

                score = source.fetch_score()
                scores[source.name] = score

                # Log each source fetch at DEBUG level (Requirement 9.3)
                logger.debug(
                    "SentimentFilter: source '%s' returned score=%s",
                    source.name,
                    score,
                )

                # Track IG fetch timestamp for rate limiting
                if source.name == "ig_client" and score is not None:
                    self._ig_last_fetch_time = time.time()

            except Exception as exc:
                logger.warning(
                    "SentimentFilter: error fetching from %s: %s",
                    source.name,
                    exc,
                )
                scores[source.name] = None

        # Log warning when all sources fail but at least one is configured
        if self._sources and all(v is None for v in scores.values()):
            logger.warning(
                "SentimentFilter: all sentiment sources unavailable, using neutral score"
            )

        # Aggregate scores
        aggregated_score = self._aggregator.aggregate(scores)

        # Mark successful if any source returned non-None
        if any(v is not None for v in scores.values()):
            self._has_fetched_successfully = True

        # Store in cache
        self._cache.set(aggregated_score, scores)
        # Log cache store at DEBUG level (Requirement 9.3)
        logger.debug(
            "SentimentFilter: cached score=%.4f, TTL=%ds remaining",
            aggregated_score,
            self._cache_ttl,
        )

        return (aggregated_score, scores)

    def confirm_signal(self, signal: dict, df=None) -> tuple[bool, dict]:
        """Confirm or reject a trade signal based on sentiment alignment.

        Decision logic:
        - BUY: confirmed if score >= -threshold, rejected if score < -threshold
        - SELL: confirmed if score <= +threshold, rejected if score > +threshold
        - Score exactly 0.0: always confirm regardless of direction

        First-fetch pass-through: if we have never successfully fetched sentiment
        data, pass the signal through (return True) until first successful fetch.

        Fail-open: any unhandled exception → return (True, {error details}).

        Args:
            signal: Trade signal dict with at minimum a "side" key ("BUY" or "SELL").
            df: Optional DataFrame (unused for now, kept for interface compatibility).

        Returns:
            Tuple of (confirmed: bool, metadata: dict) where metadata includes:
            sentiment_score, sentiment_threshold, sentiment_confirmed,
            sentiment_cache_hit, sentiment_sources, sentiment_reason.
        """
        if not self._enabled:
            return (True, {
                "sentiment_score": None,
                "sentiment_threshold": self._threshold,
                "sentiment_confirmed": True,
                "sentiment_cache_hit": False,
                "sentiment_sources": {},
                "sentiment_reason": "sentiment filter disabled",
            })

        # Permanent pass-through mode (initialization failed)
        if self._permanent_passthrough:
            return (True, {
                "sentiment_score": None,
                "sentiment_threshold": self._threshold,
                "sentiment_confirmed": True,
                "sentiment_cache_hit": False,
                "sentiment_sources": {},
                "sentiment_reason": "permanent pass-through: initialization failed",
            })

        try:
            # Determine if this is a cache hit by checking cache before get_sentiment_score
            cache_hit = self._cache.is_valid()

            score, source_details = self.get_sentiment_score()

            # First-fetch pass-through: haven't successfully fetched yet
            if not self._has_fetched_successfully:
                logger.debug(
                    "SentimentFilter: sentiment data pending, passing through"
                )
                return (True, {
                    "sentiment_score": score,
                    "sentiment_threshold": self._threshold,
                    "sentiment_confirmed": True,
                    "sentiment_cache_hit": cache_hit,
                    "sentiment_sources": source_details,
                    "sentiment_reason": "awaiting first sentiment data",
                })

            direction = signal.get("side", "BUY")
            threshold = self._threshold

            # Score exactly 0.0 always confirms
            if score == 0.0:
                confirmed = True
                reason = f"{direction} confirmed: sentiment score is neutral (0.0)"
            elif direction == "BUY":
                # BUY confirmed if score >= -threshold
                confirmed = score >= -threshold
                if confirmed:
                    reason = (
                        f"BUY confirmed: sentiment score {score:.4f} >= "
                        f"-threshold ({-threshold:.4f})"
                    )
                else:
                    reason = (
                        f"BUY rejected: sentiment score {score:.4f} < "
                        f"-threshold ({-threshold:.4f})"
                    )
            else:  # SELL
                # SELL confirmed if score <= +threshold
                confirmed = score <= threshold
                if confirmed:
                    reason = (
                        f"SELL confirmed: sentiment score {score:.4f} <= "
                        f"+threshold ({threshold:.4f})"
                    )
                else:
                    reason = (
                        f"SELL rejected: sentiment score {score:.4f} > "
                        f"+threshold ({threshold:.4f})"
                    )

            metadata = {
                "sentiment_score": score,
                "sentiment_threshold": threshold,
                "sentiment_confirmed": confirmed,
                "sentiment_cache_hit": cache_hit,
                "sentiment_sources": source_details,
                "sentiment_reason": reason,
            }

            # Log the decision at INFO level (Requirements 9.1, 9.2)
            if confirmed:
                logger.info(
                    "SentimentFilter: %s confirmed — score=%.4f, sources=%s",
                    direction,
                    score,
                    source_details,
                )
            else:
                logger.info(
                    "SentimentFilter: %s rejected — score=%.4f, threshold=%.4f, reason=%s",
                    direction,
                    score,
                    threshold,
                    reason,
                )

            # Include sentiment metadata in analytics decision log (Requirement 9.4)
            # When analytics.save_all_decisions is true, ensure metadata is
            # available in signal's meta key (pipeline integration task 7.2 will
            # handle the actual appending to the signal dict)
            analytics_config = self._config.get("analytics", {})
            if analytics_config.get("save_all_decisions", False):
                metadata["sentiment_analytics"] = True

            return (confirmed, metadata)

        except Exception as exc:
            logger.warning(
                "SentimentFilter: unhandled exception in confirm_signal, failing open: %s: %s",
                type(exc).__name__,
                exc,
            )
            return (True, {
                "sentiment_score": None,
                "sentiment_threshold": self._threshold,
                "sentiment_confirmed": True,
                "sentiment_cache_hit": False,
                "sentiment_sources": {},
                "sentiment_reason": f"fail-open: {type(exc).__name__}: {exc}",
            })


class ForexFactoryCalendarSource:
    """Fetches high-impact USD economic events from ForexFactory calendar.

    Implements the SentimentSource protocol. Checks for upcoming or recent
    high-impact USD events and returns a sentiment score based on proximity
    to major events. High-impact events create uncertainty → bullish Gold.

    Uses the free faireconomy.media JSON endpoint (no API key required).
    """

    FF_CALENDAR_URL = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"

    def __init__(self, proximity_hours: int = 2, timeout_seconds: int = 10) -> None:
        """Initialize the ForexFactory calendar source.

        Args:
            proximity_hours: Hours before/after an event to consider it "active".
            timeout_seconds: HTTP request timeout.
        """
        self._proximity_hours = proximity_hours
        self._timeout_seconds = timeout_seconds

    @property
    def name(self) -> str:
        """Unique identifier for this sentiment source."""
        return "forex_factory"

    def fetch_score(self) -> float | None:
        """Fetch current sentiment score based on proximity to high-impact USD events.

        Returns:
            Positive score (bullish Gold) when high-impact event is imminent/recent
            (uncertainty drives safe-haven demand). None if no relevant events or error.
        """
        try:
            response = requests.get(self.FF_CALENDAR_URL, timeout=self._timeout_seconds)
            if response.status_code != 200:
                logger.warning(
                    "ForexFactoryCalendar: API returned status %d", response.status_code
                )
                return None

            events = response.json()
            if not events:
                return None

            now_utc = datetime.now(timezone.utc)
            proximity_window = timedelta(hours=self._proximity_hours)

            # Filter USD high-impact events within proximity window
            active_events = []
            for event in events:
                if event.get("country") != "USD":
                    continue
                if event.get("impact") not in ("High", "Medium"):
                    continue

                event_date_str = event.get("date", "")
                if not event_date_str:
                    continue

                try:
                    # Format: "2026-07-08T14:00:00-04:00"
                    event_dt = datetime.fromisoformat(event_date_str)
                    event_dt_utc = event_dt.astimezone(timezone.utc)
                except (ValueError, TypeError):
                    continue

                # Check if event is within proximity window (before or after)
                time_diff = abs((event_dt_utc - now_utc).total_seconds())
                if time_diff <= proximity_window.total_seconds():
                    impact_weight = 1.0 if event.get("impact") == "High" else 0.5
                    # Closer to event = stronger signal
                    proximity_factor = 1.0 - (time_diff / proximity_window.total_seconds())
                    active_events.append((event, impact_weight * proximity_factor))

            if not active_events:
                return None

            # High-impact USD events create uncertainty → bullish Gold (safe haven)
            total_weight = sum(w for _, w in active_events)
            # Score between 0.1 and 0.4 based on event proximity and impact
            score = min(0.4, total_weight * 0.2)

            event_names = [e.get("title", "?") for e, _ in active_events[:3]]
            logger.info(
                "ForexFactoryCalendar: %d active USD events (score=%.3f): %s",
                len(active_events), score, ", ".join(event_names),
            )
            return score

        except requests.Timeout:
            logger.warning("ForexFactoryCalendar: request timed out")
            return None
        except Exception as exc:
            logger.warning("ForexFactoryCalendar: error: %s", exc)
            return None


class AlphaVantageNewsSource:
    """Fetches Gold-related news sentiment from Alpha Vantage News & Sentiment API.

    Implements the SentimentSource protocol. Computes a weighted average
    sentiment polarity from recent Gold news articles within a configurable
    lookback window.

    Rate limiting is enforced by tracking request timestamps and refusing
    to fetch if the configured max_requests_per_hour has been reached.
    """

    def __init__(
        self,
        api_key: str,
        lookback_hours: int = 4,
        min_articles: int = 3,
        timeout_seconds: int = 15,
        max_requests_per_hour: int = 5,
    ) -> None:
        """Initialize the Alpha Vantage news source.

        Args:
            api_key: Alpha Vantage API key. Supports ${VAR} env var expansion.
            lookback_hours: Only include articles published within this window.
            min_articles: Minimum articles required; returns None if fewer found.
            timeout_seconds: HTTP request timeout in seconds.
            max_requests_per_hour: Maximum API requests allowed per hour.
        """
        self._api_key = self._resolve_api_key(api_key)
        self._lookback_hours = lookback_hours
        self._min_articles = min_articles
        self._timeout_seconds = timeout_seconds
        self._max_requests_per_hour = max_requests_per_hour
        self._request_timestamps: list[float] = []

    @property
    def name(self) -> str:
        """Unique identifier for this sentiment source."""
        return "news"

    @staticmethod
    def _resolve_api_key(api_key: str) -> str:
        """Resolve ${VAR} env var expansion in the API key.

        If api_key starts with '${' and ends with '}', extracts the variable
        name and looks it up in os.environ.

        Args:
            api_key: Raw API key string, possibly with env var syntax.

        Returns:
            Resolved API key value, or empty string if env var not found.
        """
        if api_key.startswith("${") and api_key.endswith("}"):
            var_name = api_key[2:-1]
            return os.environ.get(var_name, "")
        return api_key

    def _is_rate_limited(self) -> bool:
        """Check if we've exceeded the max requests per hour.

        Prunes timestamps older than 1 hour, then checks count.

        Returns:
            True if at or above the rate limit, False otherwise.
        """
        now = time.time()
        one_hour_ago = now - 3600.0
        # Prune old timestamps
        self._request_timestamps = [
            ts for ts in self._request_timestamps if ts > one_hour_ago
        ]
        return len(self._request_timestamps) >= self._max_requests_per_hour

    def _record_request(self) -> None:
        """Record the current time as a request timestamp."""
        self._request_timestamps.append(time.time())

    def fetch_score(self) -> float | None:
        """Fetch current news sentiment score for Gold.

        Calls the Alpha Vantage News & Sentiment API, filters articles
        to those within the lookback window, and computes the weighted
        average sentiment polarity.

        Returns:
            Sentiment score in [-1.0, +1.0], or None if:
            - Rate limited
            - Fewer than min_articles found
            - API error or timeout
        """
        try:
            if self._is_rate_limited():
                logger.warning(
                    "AlphaVantageNewsSource: rate limited (%d requests in last hour)",
                    self._max_requests_per_hour,
                )
                return None

            url = (
                f"https://www.alphavantage.co/query"
                f"?function=NEWS_SENTIMENT"
                f"&tickers=FOREX:USD"
                f"&apikey={self._api_key}"
            )

            response = requests.get(url, timeout=self._timeout_seconds)
            self._record_request()

            # Second call: economy_macro topic for geopolitical/war news
            geo_url = (
                f"https://www.alphavantage.co/query"
                f"?function=NEWS_SENTIMENT"
                f"&topics=economy_macro"
                f"&apikey={self._api_key}"
            )
            geo_response = requests.get(geo_url, timeout=self._timeout_seconds)
            self._record_request()

            if response.status_code != 200:
                logger.warning(
                    "AlphaVantageNewsSource: API returned status %d",
                    response.status_code,
                )
                return None

            data = response.json()
            feed = data.get("feed", [])

            if not feed:
                logger.warning("AlphaVantageNewsSource: no articles in feed")
                return None

            # Filter articles within lookback window
            cutoff = datetime.now(timezone.utc) - timedelta(hours=self._lookback_hours)
            qualifying_articles = []

            for article in feed:
                time_published = article.get("time_published", "")
                if not time_published:
                    continue

                try:
                    # Alpha Vantage format: "20231215T143000"
                    article_dt = datetime.strptime(
                        time_published, "%Y%m%dT%H%M%S"
                    ).replace(tzinfo=timezone.utc)
                except (ValueError, TypeError):
                    continue

                if article_dt >= cutoff:
                    # Extract USD-specific ticker sentiment (not overall)
                    # Bullish USD → bearish Gold, so we INVERT the score
                    for ticker_info in article.get("ticker_sentiment", []):
                        ticker = ticker_info.get("ticker", "")
                        if "USD" in ticker.upper():
                            score = ticker_info.get("ticker_sentiment_score")
                            if score is not None:
                                try:
                                    # Invert: bullish USD (+) → bearish Gold (-)
                                    qualifying_articles.append(-float(score))
                                except (ValueError, TypeError):
                                    continue
                            break

            if len(qualifying_articles) < self._min_articles:
                logger.debug(
                    "AlphaVantageNewsSource: only %d USD articles (need %d)",
                    len(qualifying_articles),
                    self._min_articles,
                )
                # Fall through to geopolitical check even if USD articles insufficient

            # --- Geopolitical / war news (safe-haven demand → bullish Gold) ---
            # Scan economy_macro feed for conflict/war keywords
            # Geopolitical tension is always bullish for Gold regardless of USD direction
            # Use TWO tiers: strong keywords (always match) and weak keywords (need 2+ to match)
            strong_geo_keywords = (
                "missile", "invasion", "nuclear", "bomb", "troops",
                "nato", "weapon", "terror", "strait", "hormuz",
                "iran", "airstrike", "ceasefire", "escalation",
            )
            weak_geo_keywords = (
                "war", "conflict", "strike", "military", "sanction",
                "tension", "escalat", "attack", "defense", "geopolit",
                "tariff", "embargo", "russia", "china",
            )
            # Exclude false positives (stock "price war", "trade war" in non-geo context)
            false_positive_phrases = (
                "price war", "streaming war", "browser war", "format war",
                "star wars", "bidding war", "turf war", "war chest",
            )
            geo_boost = 0.0
            geo_count = 0

            geo_feed = geo_response.json().get("feed", []) if geo_response.status_code == 200 else []

            for article in geo_feed:
                title = article.get("title", "").lower()
                summary = article.get("summary", "").lower()
                text_to_check = title + " " + summary

                # Skip false positives
                if any(fp in text_to_check for fp in false_positive_phrases):
                    continue

                # Check strong keywords (any single match qualifies)
                strong_matches = [kw for kw in strong_geo_keywords if kw in text_to_check]
                weak_matches = [kw for kw in weak_geo_keywords if kw in text_to_check]

                # Qualify if: any strong keyword OR 2+ weak keywords
                is_geopolitical = len(strong_matches) > 0 or len(weak_matches) >= 2

                if is_geopolitical:
                    time_published = article.get("time_published", "")
                    if not time_published:
                        continue
                    try:
                        article_dt = datetime.strptime(
                            time_published, "%Y%m%dT%H%M%S"
                        ).replace(tzinfo=timezone.utc)
                    except (ValueError, TypeError):
                        continue

                    if article_dt >= cutoff:
                        overall = article.get("overall_sentiment_score")
                        if overall is not None:
                            try:
                                # Negative overall sentiment (fear/conflict) → positive for Gold
                                # Positive geo news (de-escalation) → slightly negative for Gold
                                geo_boost += -float(overall)
                                geo_count += 1
                            except (ValueError, TypeError):
                                continue

            if geo_count > 0:
                # Average geo score — not capped, strong geopolitical events should dominate
                avg_geo = geo_boost / geo_count
                # Weight geo signal: more articles = stronger conviction
                geo_weight = min(1.0, geo_count / 5.0)  # Full weight at 5+ articles
                qualifying_articles.append(avg_geo * geo_weight)
                logger.debug(
                    "AlphaVantageNewsSource: %d geopolitical articles, avg_score=%.4f, weight=%.2f",
                    geo_count, avg_geo, geo_weight,
                )

            if not qualifying_articles:
                return None

            # Compute weighted average (equal weight per article)
            avg_score = sum(qualifying_articles) / len(qualifying_articles)
            # Clamp to [-1.0, +1.0]
            return max(-1.0, min(1.0, avg_score))

        except requests.Timeout:
            logger.warning(
                "AlphaVantageNewsSource: request timed out after %ds",
                self._timeout_seconds,
            )
            return None
        except Exception as exc:
            logger.warning("AlphaVantageNewsSource: error fetching news: %s", exc)
            return None
