"""Property-based tests for Sentiment Analysis Filter using Hypothesis.

Validates correctness properties defined in the design document.
"""

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch, call

from hypothesis import given, settings, assume
from hypothesis.strategies import (
    booleans,
    composite,
    dictionaries,
    floats,
    integers,
    just,
    lists,
    none,
    one_of,
    sampled_from,
    text,
)

from strategy.sentiment_filter import (
    AlphaVantageNewsSource,
    IGClientSentimentSource,
    SentimentCache,
    SentimentFilter,
    WeightedAggregator,
)


# ---------------------------------------------------------------------------
# Strategies (generators)
# ---------------------------------------------------------------------------


@composite
def sentiment_scores(draw):
    """Generate random sentiment scores in [-1.0, +1.0]."""
    return draw(floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False))


@composite
def source_detail_dicts(draw):
    """Generate random source detail dicts mapping source names to scores or None."""
    return draw(
        dictionaries(
            keys=text(min_size=1, max_size=20),
            values=one_of(
                floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False),
                none(),
            ),
            min_size=0,
            max_size=5,
        )
    )


# ---------------------------------------------------------------------------
# Property 7: Cache Round-Trip
# Feature: sentiment-analysis, Property 7: Cache Round-Trip
# Validates: Requirements 5.1, 5.2
# ---------------------------------------------------------------------------


class TestCacheRoundTrip:
    """Property 7: Cache Round-Trip.

    For any sentiment score and source details stored in the cache,
    retrieving within the configured TTL SHALL return the exact same
    score and details. Retrieving after TTL expiry SHALL return None.

    **Validates: Requirements 5.1, 5.2**
    """

    @given(
        score=sentiment_scores(),
        details=source_detail_dicts(),
        ttl=integers(min_value=1, max_value=3600),
    )
    @settings(max_examples=100)
    def test_get_within_ttl_returns_exact_values(self, score: float, details: dict, ttl: int):
        """Storing a score and details, then retrieving within TTL, returns exact values."""
        # Feature: sentiment-analysis, Property 7: Cache Round-Trip
        cache = SentimentCache(ttl_seconds=ttl)

        # Mock time.time() to control timestamps
        fake_time = 1000000.0
        with patch("time.time", return_value=fake_time):
            cache.set(score, details)

        # Retrieve within TTL (e.g., 1 second later, well within any TTL >= 1)
        with patch("time.time", return_value=fake_time + 0.5):
            result = cache.get()

        assert result is not None, "Cache should return data within TTL"
        retrieved_score, retrieved_details = result
        assert retrieved_score == score, (
            f"Expected score {score}, got {retrieved_score}"
        )
        assert retrieved_details == details, (
            f"Expected details {details}, got {retrieved_details}"
        )

    @given(
        score=sentiment_scores(),
        details=source_detail_dicts(),
        ttl=integers(min_value=1, max_value=3600),
    )
    @settings(max_examples=100)
    def test_get_after_ttl_returns_none(self, score: float, details: dict, ttl: int):
        """Storing a score and details, then retrieving after TTL expiry, returns None."""
        # Feature: sentiment-analysis, Property 7: Cache Round-Trip
        cache = SentimentCache(ttl_seconds=ttl)

        fake_time = 1000000.0
        with patch("time.time", return_value=fake_time):
            cache.set(score, details)

        # Retrieve after TTL has expired (ttl + 1 second past)
        with patch("time.time", return_value=fake_time + ttl + 1):
            result = cache.get()

        assert result is None, (
            f"Cache should return None after TTL expiry (ttl={ttl}s), got {result}"
        )


# ---------------------------------------------------------------------------
# Property 1: IG Contrarian Score Mapping
# Feature: sentiment-analysis, Property 1: IG Contrarian Score Mapping
# Validates: Requirements 1.2, 1.3, 1.4
# ---------------------------------------------------------------------------


class TestIGContrarianScoreMapping:
    """Property 1: IG Contrarian Score Mapping.

    For any IG client sentiment response with long_pct and short_pct
    (where long_pct + short_pct ≈ 100):
    - If long_pct > 70%, the score SHALL be negative and equal to -(long_pct - 50) / 50
    - If short_pct > 70%, the score SHALL be positive and equal to +(short_pct - 50) / 50
    - If both are in [30%, 70%], the score SHALL be exactly 0.0

    The resulting score SHALL always be in the range [-1.0, +1.0].

    **Validates: Requirements 1.2, 1.3, 1.4**
    """

    @given(
        long_pct=floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=100)
    def test_contrarian_formula_and_range(self, long_pct: float):
        """Verify contrarian formula produces correct score and stays in [-1.0, +1.0]."""
        # Feature: sentiment-analysis, Property 1: IG Contrarian Score Mapping
        short_pct = 100.0 - long_pct

        # Mock the IG client and API response
        mock_ig_client = MagicMock()
        mock_ig_client.base = "https://demo-api.ig.com/gateway/deal"
        mock_ig_client.verify_ssl = False

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "longPositionPercentage": long_pct,
            "shortPositionPercentage": short_pct,
        }
        mock_response.raise_for_status = MagicMock()
        mock_ig_client.s.get.return_value = mock_response
        mock_ig_client._hv.return_value = {"Version": "1"}

        source = IGClientSentimentSource(ig_client=mock_ig_client, market_id="GOLD")
        score = source.fetch_score()

        # Score must not be None (no error path triggered)
        assert score is not None, f"Expected a score, got None for long_pct={long_pct}"

        # Verify contrarian formula
        if long_pct > 70:
            expected = -(long_pct - 50) / 50
            assert abs(score - expected) < 1e-9, (
                f"long_pct={long_pct}: expected {expected}, got {score}"
            )
        elif short_pct > 70:
            expected = +(short_pct - 50) / 50
            assert abs(score - expected) < 1e-9, (
                f"short_pct={short_pct}: expected {expected}, got {score}"
            )
        else:
            assert score == 0.0, (
                f"long_pct={long_pct}, short_pct={short_pct}: expected 0.0, got {score}"
            )

        # Score must always be in [-1.0, +1.0]
        assert -1.0 <= score <= 1.0, (
            f"Score {score} out of range [-1.0, +1.0] for long_pct={long_pct}"
        )


# ---------------------------------------------------------------------------
# Property 4: Score Clamping Invariant
# Feature: sentiment-analysis, Property 4: Score Clamping Invariant
# Validates: Requirements 3.4
# ---------------------------------------------------------------------------


@composite
def extreme_scores_and_weights(draw):
    """Generate extreme source scores with arbitrary positive weights.

    Scores range from [-100.0, +100.0] to test clamping behavior.
    Weights can be highly unbalanced (e.g., one weight 100x larger than another).
    """
    num_sources = draw(integers(min_value=1, max_value=5))
    source_names = [f"source_{i}" for i in range(num_sources)]

    scores = {}
    weights = {}
    for name in source_names:
        # Extreme scores well outside [-1.0, +1.0]
        score = draw(
            one_of(
                floats(min_value=-100.0, max_value=100.0, allow_nan=False, allow_infinity=False),
                none(),
            )
        )
        scores[name] = score
        # Arbitrary positive weights — can be very unbalanced
        weights[name] = draw(
            floats(min_value=0.01, max_value=1000.0, allow_nan=False, allow_infinity=False)
        )

    return scores, weights


class TestScoreClampingInvariant:
    """Property 4: Score Clamping Invariant.

    For any combination of source scores and weights, the final aggregated
    Sentiment_Score SHALL always be in the range [-1.0, +1.0].

    **Validates: Requirements 3.4**
    """

    @given(data=extreme_scores_and_weights())
    @settings(max_examples=100)
    def test_aggregate_always_within_bounds(self, data: tuple):
        """Extreme source scores with arbitrary weights always produce a clamped result."""
        # Feature: sentiment-analysis, Property 4: Score Clamping Invariant
        scores, weights = data

        aggregator = WeightedAggregator(weights)
        result = aggregator.aggregate(scores)

        assert -1.0 <= result <= 1.0, (
            f"Aggregated score {result} is outside [-1.0, +1.0]. "
            f"Scores: {scores}, Weights: {weights}"
        )


# ---------------------------------------------------------------------------
# Property 2: News Score Computation
# Feature: sentiment-analysis, Property 2: News Score Computation
# Validates: Requirements 2.2, 2.3
# ---------------------------------------------------------------------------


@composite
def article_set(draw):
    """Generate a random set of news articles with timestamps and sentiment scores.

    Each article has:
    - time_published: Alpha Vantage format '%Y%m%dT%H%M%S'
    - overall_sentiment_score: float in [-1.0, +1.0]
    - hours_ago: how many hours before 'now' the article was published (for test control)
    """
    num_articles = draw(integers(min_value=0, max_value=15))
    articles = []
    for _ in range(num_articles):
        # hours_ago determines if article is inside or outside lookback window
        hours_ago = draw(floats(min_value=0.0, max_value=24.0, allow_nan=False, allow_infinity=False))
        score = draw(floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False))
        articles.append({"hours_ago": hours_ago, "score": score})
    return articles


class TestNewsScoreComputation:
    """Property 2: News Score Computation.

    For any set of news articles with timestamps and sentiment polarity scores:
    - Only articles published within the configured lookback window SHALL be included
    - If fewer than min_articles within window → score SHALL be None
    - Otherwise score SHALL equal the average of their overall_sentiment_scores

    **Validates: Requirements 2.2, 2.3**
    """

    @given(
        articles=article_set(),
        lookback_hours=integers(min_value=1, max_value=12),
        min_articles=integers(min_value=1, max_value=10),
    )
    @settings(max_examples=100)
    def test_lookback_filtering_and_min_articles(
        self, articles: list, lookback_hours: int, min_articles: int
    ):
        """Verify only articles within lookback window are included and min_articles threshold enforced."""
        # Feature: sentiment-analysis, Property 2: News Score Computation

        # Fixed reference time for deterministic testing
        now = datetime(2024, 6, 15, 12, 0, 0, tzinfo=timezone.utc)

        # Build the mock API feed from generated articles
        feed = []
        expected_qualifying_scores = []

        for article in articles:
            hours_ago = article["hours_ago"]
            score = article["score"]
            article_dt = now - timedelta(hours=hours_ago)
            time_published = article_dt.strftime("%Y%m%dT%H%M%S")

            feed.append({
                "time_published": time_published,
                "overall_sentiment_score": score,
            })

            # Article is within window if hours_ago < lookback_hours
            # (cutoff = now - lookback_hours, article qualifies if article_dt >= cutoff)
            if hours_ago <= lookback_hours:
                expected_qualifying_scores.append(score)

        # Create source with test parameters
        source = AlphaVantageNewsSource(
            api_key="test-key",
            lookback_hours=lookback_hours,
            min_articles=min_articles,
            timeout_seconds=15,
            max_requests_per_hour=100,  # high limit to avoid rate limiting
        )

        # Mock the requests.get call
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"feed": feed}

        with patch("strategy.sentiment_filter.requests.get", return_value=mock_response):
            with patch("strategy.sentiment_filter.datetime") as mock_dt:
                # Make datetime.now(timezone.utc) return our fixed time
                mock_dt.now.return_value = now
                # Keep strptime working normally
                mock_dt.strptime = datetime.strptime

                result = source.fetch_score()

        # Verify properties
        if len(expected_qualifying_scores) < min_articles:
            assert result is None, (
                f"Expected None (only {len(expected_qualifying_scores)} articles, "
                f"need {min_articles}), got {result}"
            )
        else:
            assert result is not None, (
                f"Expected a score ({len(expected_qualifying_scores)} articles >= "
                f"{min_articles}), got None"
            )
            expected_avg = sum(expected_qualifying_scores) / len(expected_qualifying_scores)
            expected_clamped = max(-1.0, min(1.0, expected_avg))
            assert abs(result - expected_clamped) < 1e-9, (
                f"Expected avg {expected_clamped}, got {result}"
            )

    @given(
        articles=article_set(),
        lookback_hours=integers(min_value=1, max_value=12),
        min_articles=integers(min_value=1, max_value=10),
    )
    @settings(max_examples=100)
    def test_result_within_valid_range(
        self, articles: list, lookback_hours: int, min_articles: int
    ):
        """Verify that when a score is returned, it is always in [-1.0, +1.0]."""
        # Feature: sentiment-analysis, Property 2: News Score Computation

        now = datetime(2024, 6, 15, 12, 0, 0, tzinfo=timezone.utc)

        feed = []
        for article in articles:
            hours_ago = article["hours_ago"]
            score = article["score"]
            article_dt = now - timedelta(hours=hours_ago)
            time_published = article_dt.strftime("%Y%m%dT%H%M%S")
            feed.append({
                "time_published": time_published,
                "overall_sentiment_score": score,
            })

        source = AlphaVantageNewsSource(
            api_key="test-key",
            lookback_hours=lookback_hours,
            min_articles=min_articles,
            timeout_seconds=15,
            max_requests_per_hour=100,
        )

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"feed": feed}

        with patch("strategy.sentiment_filter.requests.get", return_value=mock_response):
            with patch("strategy.sentiment_filter.datetime") as mock_dt:
                mock_dt.now.return_value = now
                mock_dt.strptime = datetime.strptime

                result = source.fetch_score()

        if result is not None:
            assert -1.0 <= result <= 1.0, (
                f"Score {result} out of valid range [-1.0, +1.0]"
            )


# ---------------------------------------------------------------------------
# Property 3: Weighted Aggregation
# Feature: sentiment-analysis, Property 3: Weighted Aggregation
# Validates: Requirements 3.1, 3.2, 3.3
# ---------------------------------------------------------------------------


@composite
def weight_configs(draw):
    """Generate random weight dicts with positive weights and matching score dicts."""
    # Generate 1-5 source names
    source_names = draw(
        lists(
            text(min_size=1, max_size=10, alphabet="abcdefghijklmnopqrstuvwxyz"),
            min_size=1,
            max_size=5,
            unique=True,
        )
    )
    # Generate positive weights for each source
    weights = {}
    for name in source_names:
        w = draw(floats(min_value=0.01, max_value=10.0, allow_nan=False, allow_infinity=False))
        weights[name] = w

    # Generate scores (float in [-1.0, +1.0] or None) for each source
    scores = {}
    for name in source_names:
        score = draw(
            one_of(
                floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False),
                none(),
            )
        )
        scores[name] = score

    return weights, scores


class TestWeightedAggregation:
    """Property 3: Weighted Aggregation.

    For any set of source scores (some possibly None) and configured weights,
    the aggregated score SHALL equal the weighted average of all non-None scores
    using their respective weights, re-normalized by the sum of active weights.
    When only one source returns a valid score, the aggregated score SHALL equal
    that source's score. When all sources are None, the aggregated score SHALL be 0.0.

    **Validates: Requirements 3.1, 3.2, 3.3**
    """

    @given(data=weight_configs())
    @settings(max_examples=100)
    def test_weighted_average_with_renormalization(self, data):
        """Result equals weighted average of non-None scores, re-normalized by active weights."""
        # Feature: sentiment-analysis, Property 3: Weighted Aggregation
        weights, scores = data
        aggregator = WeightedAggregator(weights)
        result = aggregator.aggregate(scores)

        # Compute expected value manually
        active_sources = {k: v for k, v in scores.items() if v is not None}
        if not active_sources:
            expected = 0.0
        else:
            weighted_sum = sum(scores[k] * weights[k] for k in active_sources)
            active_weight_sum = sum(weights[k] for k in active_sources)
            expected = weighted_sum / active_weight_sum
            expected = max(-1.0, min(1.0, expected))

        assert abs(result - expected) < 1e-9, (
            f"Expected {expected}, got {result}. "
            f"Weights={weights}, Scores={scores}"
        )

    @given(data=weight_configs())
    @settings(max_examples=100)
    def test_single_source_passthrough(self, data):
        """When only one source is non-None, result equals that source's score."""
        # Feature: sentiment-analysis, Property 3: Weighted Aggregation
        weights, scores = data

        # Filter to only cases with exactly one non-None score
        non_none = {k: v for k, v in scores.items() if v is not None}
        assume(len(non_none) == 1)

        aggregator = WeightedAggregator(weights)
        result = aggregator.aggregate(scores)

        # The single non-None score should be the result (clamped)
        single_score = list(non_none.values())[0]
        expected = max(-1.0, min(1.0, single_score))

        assert abs(result - expected) < 1e-9, (
            f"Single source passthrough failed: expected {expected}, got {result}. "
            f"Weights={weights}, Scores={scores}"
        )

    @given(data=weight_configs())
    @settings(max_examples=100)
    def test_all_none_returns_zero(self, data):
        """When all sources are None, result is 0.0."""
        # Feature: sentiment-analysis, Property 3: Weighted Aggregation
        weights, scores = data

        # Make all scores None
        all_none_scores = {k: None for k in scores}

        aggregator = WeightedAggregator(weights)
        result = aggregator.aggregate(all_none_scores)

        assert result == 0.0, (
            f"Expected 0.0 when all sources are None, got {result}. "
            f"Weights={weights}"
        )

    @given(data=weight_configs())
    @settings(max_examples=100)
    def test_result_always_in_range(self, data):
        """Result is always in [-1.0, +1.0] regardless of inputs."""
        # Feature: sentiment-analysis, Property 3: Weighted Aggregation
        weights, scores = data
        aggregator = WeightedAggregator(weights)
        result = aggregator.aggregate(scores)

        assert -1.0 <= result <= 1.0, (
            f"Result {result} out of range [-1.0, +1.0]. "
            f"Weights={weights}, Scores={scores}"
        )


# ---------------------------------------------------------------------------
# Property 5: Signal Confirmation Decision
# Feature: sentiment-analysis, Property 5: Signal Confirmation Decision
# Validates: Requirements 4.1, 4.2, 4.3, 4.4, 4.6
# ---------------------------------------------------------------------------


class TestSignalConfirmationDecision:
    """Property 5: Signal Confirmation Decision.

    For any signal with direction (BUY or SELL) and any Sentiment_Score
    in [-1.0, +1.0] and any threshold > 0:
    - A BUY signal is confirmed if and only if score >= -threshold
    - A SELL signal is confirmed if and only if score <= +threshold
    - When score is exactly 0.0, the signal SHALL always be confirmed regardless of direction

    **Validates: Requirements 4.1, 4.2, 4.3, 4.4, 4.6**
    """

    @given(
        direction=sampled_from(["BUY", "SELL"]),
        score=floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        threshold=floats(min_value=0.01, max_value=1.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=100)
    def test_buy_sell_confirmation_rules(self, direction: str, score: float, threshold: float):
        """Verify BUY confirmed iff score >= -threshold, SELL confirmed iff score <= +threshold."""
        # Feature: sentiment-analysis, Property 5: Signal Confirmation Decision
        config = {"enabled": True, "sentiment_threshold": threshold}
        sf = SentimentFilter(config=config)

        # Monkey-patch get_sentiment_score to return the given score
        sf.get_sentiment_score = lambda: (score, {"test": score})

        signal = {"side": direction}
        confirmed, metadata = sf.confirm_signal(signal)

        if direction == "BUY":
            expected_confirmed = score >= -threshold
        else:  # SELL
            expected_confirmed = score <= threshold

        assert confirmed == expected_confirmed, (
            f"direction={direction}, score={score}, threshold={threshold}: "
            f"expected confirmed={expected_confirmed}, got {confirmed}"
        )

    @given(
        direction=sampled_from(["BUY", "SELL"]),
        threshold=floats(min_value=0.01, max_value=1.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=100)
    def test_zero_score_always_confirms(self, direction: str, threshold: float):
        """When score is exactly 0.0, signal is always confirmed regardless of direction."""
        # Feature: sentiment-analysis, Property 5: Signal Confirmation Decision
        config = {"enabled": True, "sentiment_threshold": threshold}
        sf = SentimentFilter(config=config)

        # Monkey-patch get_sentiment_score to return exactly 0.0
        sf.get_sentiment_score = lambda: (0.0, {"test": 0.0})

        signal = {"side": direction}
        confirmed, metadata = sf.confirm_signal(signal)

        assert confirmed is True, (
            f"direction={direction}, score=0.0, threshold={threshold}: "
            f"expected confirmed=True, got {confirmed}"
        )


# ---------------------------------------------------------------------------
# Property 6: Interface Contract
# Feature: sentiment-analysis, Property 6: Interface Contract
# Validates: Requirements 4.5, 8.4, 8.5
# ---------------------------------------------------------------------------


@composite
def valid_signal_dicts(draw):
    """Generate random valid signal dicts with required keys."""
    side = draw(sampled_from(["BUY", "SELL"]))
    stop_pts = draw(floats(min_value=0.1, max_value=100.0, allow_nan=False, allow_infinity=False))
    tp_pts = draw(floats(min_value=0.1, max_value=100.0, allow_nan=False, allow_infinity=False))
    # Meta can be empty or have some random keys
    meta = draw(
        one_of(
            just({}),
            dictionaries(
                keys=text(min_size=1, max_size=10, alphabet="abcdefghijklmnopqrstuvwxyz"),
                values=text(min_size=0, max_size=20),
                min_size=1,
                max_size=3,
            ),
        )
    )
    return {"side": side, "stop_pts": stop_pts, "tp_pts": tp_pts, "meta": meta}


class TestInterfaceContract:
    """Property 6: Interface Contract.

    For any valid signal dict (containing keys side, stop_pts, tp_pts, meta),
    confirm_signal SHALL always return a tuple[bool, dict] where the dict
    contains at minimum the keys: sentiment_score, sentiment_threshold,
    sentiment_confirmed, sentiment_cache_hit, sentiment_sources, and
    sentiment_reason.

    **Validates: Requirements 4.5, 8.4, 8.5**
    """

    @given(signal=valid_signal_dicts())
    @settings(max_examples=100)
    def test_return_type_and_required_keys(self, signal: dict):
        """confirm_signal returns tuple[bool, dict] with all required metadata keys."""
        # Feature: sentiment-analysis, Property 6: Interface Contract
        config = {"enabled": True, "sentiment_threshold": 0.3}
        sf = SentimentFilter(config=config)

        result = sf.confirm_signal(signal)

        # 1. Return type is tuple with length 2
        assert isinstance(result, tuple), (
            f"Expected tuple, got {type(result)}"
        )
        assert len(result) == 2, (
            f"Expected tuple of length 2, got length {len(result)}"
        )

        confirmed, metadata = result

        # 2. First element is bool
        assert isinstance(confirmed, bool), (
            f"Expected first element to be bool, got {type(confirmed)}"
        )

        # 3. Second element is dict
        assert isinstance(metadata, dict), (
            f"Expected second element to be dict, got {type(metadata)}"
        )

        # 4. Dict contains all required keys
        required_keys = {
            "sentiment_score",
            "sentiment_threshold",
            "sentiment_confirmed",
            "sentiment_cache_hit",
            "sentiment_sources",
            "sentiment_reason",
        }
        missing_keys = required_keys - set(metadata.keys())
        assert not missing_keys, (
            f"Missing required metadata keys: {missing_keys}. "
            f"Got keys: {set(metadata.keys())}"
        )


# ---------------------------------------------------------------------------
# Property 8: Disabled Pass-Through
# Feature: sentiment-analysis, Property 8: Disabled Pass-Through
# Validates: Requirements 6.3
# ---------------------------------------------------------------------------


class TestDisabledPassThrough:
    """Property 8: Disabled Pass-Through.

    For any signal, when the filter is configured with `enabled: false`,
    `confirm_signal` SHALL return `(True, metadata)` where metadata indicates
    the filter is disabled, without invoking any sentiment source fetch.

    **Validates: Requirements 6.3**
    """

    @given(
        side=sampled_from(["BUY", "SELL"]),
    )
    @settings(max_examples=100)
    def test_disabled_filter_always_confirms(self, side: str):
        """When enabled=False, confirm_signal returns (True, metadata) for any signal."""
        # Feature: sentiment-analysis, Property 8: Disabled Pass-Through
        config = {"enabled": False}
        sf = SentimentFilter(config=config)

        signal = {"side": side}

        with patch.object(sf, "get_sentiment_score") as mock_get_score:
            confirmed, metadata = sf.confirm_signal(signal)

            # 1. Return is (True, metadata) — confirmed is always True
            assert confirmed is True, (
                f"Expected confirmed=True when filter is disabled, got {confirmed} "
                f"for side={side}"
            )

            # 2. metadata["sentiment_reason"] indicates filter is disabled
            assert "sentiment_reason" in metadata, (
                "metadata must contain 'sentiment_reason' key"
            )
            assert "disabled" in metadata["sentiment_reason"].lower(), (
                f"Expected 'disabled' in sentiment_reason, got: "
                f"{metadata['sentiment_reason']}"
            )

            # 3. No source fetch was invoked
            mock_get_score.assert_not_called(), (
                "get_sentiment_score should not be called when filter is disabled"
            )

    @given(
        side=sampled_from(["BUY", "SELL"]),
    )
    @settings(max_examples=100)
    def test_disabled_filter_score_is_none(self, side: str):
        """When enabled=False, metadata sentiment_score is None (no computation)."""
        # Feature: sentiment-analysis, Property 8: Disabled Pass-Through
        config = {"enabled": False}
        sf = SentimentFilter(config=config)

        signal = {"side": side}
        confirmed, metadata = sf.confirm_signal(signal)

        # Score should be None since no computation is performed
        assert metadata.get("sentiment_score") is None, (
            f"Expected sentiment_score=None when disabled, got "
            f"{metadata.get('sentiment_score')}"
        )
        # confirmed flag in metadata should also be True
        assert metadata.get("sentiment_confirmed") is True, (
            f"Expected sentiment_confirmed=True in metadata when disabled, got "
            f"{metadata.get('sentiment_confirmed')}"
        )

# ---------------------------------------------------------------------------
# Property 9: Fail-Open Under Exceptions
# Feature: sentiment-analysis, Property 9: Fail-Open Under Exceptions
# Validates: Requirements 7.1, 7.2
# ---------------------------------------------------------------------------


@composite
def random_signal_dicts(draw):
    """Generate random signal dicts with a side key."""
    side = draw(sampled_from(["BUY", "SELL"]))
    signal = {"side": side}
    # Optionally add other keys
    if draw(booleans()):
        signal["stop_pts"] = draw(floats(min_value=0.1, max_value=100.0, allow_nan=False, allow_infinity=False))
    if draw(booleans()):
        signal["tp_pts"] = draw(floats(min_value=0.1, max_value=200.0, allow_nan=False, allow_infinity=False))
    if draw(booleans()):
        signal["meta"] = {"some_key": "some_value"}
    return signal


@composite
def random_exceptions(draw):
    """Generate random exception instances from common exception types."""
    exc_type = draw(sampled_from([
        RuntimeError,
        ValueError,
        ConnectionError,
        TimeoutError,
        OSError,
        TypeError,
        KeyError,
        AttributeError,
        IOError,
        MemoryError,
    ]))
    message = draw(text(min_size=0, max_size=50))
    return exc_type(message)


class TestFailOpenUnderExceptions:
    """Property 9: Fail-Open Under Exceptions.

    For any signal, if the sentiment computation raises any exception,
    confirm_signal SHALL catch it and return (True, metadata) where
    metadata includes error details. The filter SHALL never raise an
    exception to the caller.

    **Validates: Requirements 7.1, 7.2**
    """

    @given(
        signal=random_signal_dicts(),
        exception=random_exceptions(),
    )
    @settings(max_examples=100)
    def test_confirm_signal_never_raises(self, signal: dict, exception: Exception):
        """Injecting random exceptions into get_sentiment_score never propagates to caller."""
        # Feature: sentiment-analysis, Property 9: Fail-Open Under Exceptions
        config = {"enabled": True, "sentiment_threshold": 0.3}
        sf = SentimentFilter(config=config)

        # Monkey-patch get_sentiment_score to raise the generated exception
        def raise_exception():
            raise exception

        sf.get_sentiment_score = raise_exception

        # confirm_signal must NEVER raise — it must always return a result
        try:
            result = sf.confirm_signal(signal)
        except Exception as e:
            # This should never happen — fail the test with details
            raise AssertionError(
                f"confirm_signal raised {type(e).__name__}({e}) instead of "
                f"catching it. Injected exception: {type(exception).__name__}({exception})"
            ) from e

        # Verify return structure: (True, metadata_dict)
        assert isinstance(result, tuple), (
            f"Expected tuple, got {type(result)}"
        )
        assert len(result) == 2, (
            f"Expected 2-tuple, got length {len(result)}"
        )

        confirmed, metadata = result

        # Must always confirm on error (fail-open)
        assert confirmed is True, (
            f"Expected confirmed=True (fail-open), got {confirmed}. "
            f"Exception was: {type(exception).__name__}({exception})"
        )

        # Metadata must be a dict
        assert isinstance(metadata, dict), (
            f"Expected metadata dict, got {type(metadata)}"
        )

        # Metadata must contain error details indicating an error occurred
        assert "sentiment_error" in metadata or "sentiment_reason" in metadata, (
            f"Metadata should contain error details, got: {metadata}"
        )

        # The reason should indicate a fail-open scenario
        reason = metadata.get("sentiment_reason", "")
        assert "fail-open" in reason or "error" in reason.lower(), (
            f"Expected fail-open/error indication in reason, got: {reason}"
        )

        # sentiment_confirmed should also be True in metadata
        assert metadata.get("sentiment_confirmed") is True, (
            f"Expected sentiment_confirmed=True in metadata, got "
            f"{metadata.get('sentiment_confirmed')}"
        )
