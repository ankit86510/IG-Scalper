# Implementation Plan: Sentiment Analysis Filter

## Overview

Implement a Gold market sentiment analysis filter that evaluates sentiment from IG client positioning and Alpha Vantage news, aggregates scores with configurable weights, and confirms or rejects trade signals based on sentiment alignment. The filter integrates into the existing pipeline after the ML Directional Filter, following the same `confirm_signal()` interface and fail-open philosophy.

## Tasks

- [x] 1. Create core interfaces and data models
  - [x] 1.1 Create `strategy/sentiment_filter.py` with `SentimentSource` protocol, `CacheEntry` dataclass, and `SentimentCache` class
    - Define `SentimentSource` Protocol with `name` property and `fetch_score() -> float | None` method
    - Implement `CacheEntry` dataclass with `score`, `source_details`, `timestamp`, `ttl_seconds` fields
    - Implement `SentimentCache` with `get()`, `set()`, `is_valid()` methods using `time.time()` for TTL
    - _Requirements: 5.1, 5.2_

  - [x] 1.2 Write property test for cache round-trip
    - **Property 7: Cache Round-Trip**
    - Generate random scores in [-1.0, +1.0] and source detail dicts, store in cache, verify retrieval within TTL returns exact values and after TTL returns None
    - **Validates: Requirements 5.1, 5.2**

- [x] 2. Implement sentiment sources
  - [x] 2.1 Implement `IGClientSentimentSource` in `strategy/sentiment_filter.py`
    - Accept `IGClient` instance and `market_id` (default "GOLD")
    - Call IG REST API `/clientsentiment/{market_id}` with 10s timeout
    - Apply contrarian logic: long_pct > 70% → `-(long_pct - 50) / 50`, short_pct > 70% → `+(short_pct - 50) / 50`, else 0.0
    - Return `None` on timeout/error (log at WARNING)
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5_

  - [x] 2.2 Write property test for IG contrarian score mapping
    - **Property 1: IG Contrarian Score Mapping**
    - Generate random `(long_pct, short_pct)` pairs where `long_pct + short_pct ≈ 100`, verify contrarian formula and output range [-1.0, +1.0]
    - **Validates: Requirements 1.2, 1.3, 1.4**

  - [x] 2.3 Implement `AlphaVantageNewsSource` in `strategy/sentiment_filter.py`
    - Accept `api_key`, `lookback_hours` (default 4), `min_articles` (default 3), `timeout_seconds` (default 15), `max_requests_per_hour` (default 5)
    - Fetch Gold-related news from Alpha Vantage News & Sentiment API
    - Filter articles to those within `lookback_hours` window
    - Return `None` if fewer than `min_articles` found, or on timeout/error
    - Compute weighted average sentiment polarity of qualifying articles
    - Support `${ALPHA_VANTAGE_KEY}` env var expansion for API key
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 5.5_

  - [x] 2.4 Write property test for news score computation
    - **Property 2: News Score Computation**
    - Generate random article sets with varying timestamps and polarity scores, verify lookback filtering, min_articles threshold, and weighted average calculation
    - **Validates: Requirements 2.2, 2.3**

- [x] 3. Implement weighted aggregation and signal confirmation
  - [x] 3.1 Implement `WeightedAggregator` in `strategy/sentiment_filter.py`
    - Accept `weights: dict[str, float]` configuration
    - Compute weighted average of non-None scores, re-normalized by sum of active weights
    - Return 0.0 when all sources are None
    - Clamp final result to [-1.0, +1.0]
    - _Requirements: 3.1, 3.2, 3.3, 3.4_

  - [x] 3.2 Write property test for weighted aggregation
    - **Property 3: Weighted Aggregation**
    - Generate random source scores (some None) and weight configs, verify weighted average with re-normalization, single-source passthrough, and all-None → 0.0
    - **Validates: Requirements 3.1, 3.2, 3.3**

  - [x] 3.3 Write property test for score clamping invariant
    - **Property 4: Score Clamping Invariant**
    - Generate extreme source scores and arbitrary weights, verify final score is always in [-1.0, +1.0]
    - **Validates: Requirements 3.4**

  - [x] 3.4 Implement signal confirmation logic in `SentimentFilter.confirm_signal()`
    - BUY confirmed if `score >= -threshold`; rejected if `score < -threshold`
    - SELL confirmed if `score <= +threshold`; rejected if `score > +threshold`
    - Score exactly 0.0 → always confirm
    - Return `tuple[bool, dict]` with sentiment metadata (score, threshold, confirmed, cache_hit, sources, reason)
    - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 4.6_

  - [x] 3.5 Write property test for signal confirmation decision
    - **Property 5: Signal Confirmation Decision**
    - Generate random `(direction, score, threshold)` triples, verify BUY/SELL confirmation rules and that score=0.0 always confirms
    - **Validates: Requirements 4.1, 4.2, 4.3, 4.4, 4.6**

- [x] 4. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 5. Implement the SentimentFilter orchestrator class
  - [x] 5.1 Implement `SentimentFilter.__init__()` and `get_sentiment_score()` in `strategy/sentiment_filter.py`
    - Read config from `sentiment_filter` section of settings_ai.yaml
    - Support keys: `enabled`, `sentiment_threshold`, `cache_ttl_seconds`, `sources`, `source_weights`
    - Initialize sources based on config (IG client, Alpha Vantage)
    - Use defaults for missing keys; support `${VAR}` env var expansion for API keys
    - Coordinate cache check → source fetch → aggregation flow
    - Respect rate limits per source (IG: 1/5min, Alpha Vantage: 5/hour)
    - _Requirements: 5.3, 5.4, 5.5, 6.1, 6.2, 6.4, 6.5_

  - [x] 5.2 Write property test for interface contract
    - **Property 6: Interface Contract**
    - Generate random valid signal dicts (with `side`, `stop_pts`, `tp_pts`, `meta` keys), call `confirm_signal`, verify return is `tuple[bool, dict]` with required metadata keys
    - **Validates: Requirements 4.5, 8.4, 8.5**

  - [x] 5.3 Write property test for disabled pass-through
    - **Property 8: Disabled Pass-Through**
    - Generate random signals with `enabled: false` config, verify `confirm_signal` returns `(True, metadata)` without invoking any source fetch
    - **Validates: Requirements 6.3**

  - [x] 5.4 Write property test for fail-open under exceptions
    - **Property 9: Fail-Open Under Exceptions**
    - Inject random exceptions into sentiment computation, verify `confirm_signal` always returns `(True, metadata)` with error details and never raises
    - **Validates: Requirements 7.1, 7.2**

- [x] 6. Implement fail-open error handling and logging
  - [x] 6.1 Add fail-open error handling to `SentimentFilter`
    - Wrap `confirm_signal` in try/except: any exception → return `(True, {error details})`, log at WARNING
    - When all sources fail → score = 0.0, confirm signal, log warning
    - Before first successful fetch → pass-through mode, log at DEBUG
    - Initialization failure → permanent pass-through mode, log at ERROR
    - _Requirements: 7.1, 7.2, 7.3, 7.4_

  - [x] 6.2 Add logging and observability to `SentimentFilter`
    - Confirmed signals: INFO with score, source scores, direction
    - Rejected signals: INFO with score, threshold, reason
    - Data refresh: DEBUG with source, score, cache TTL remaining
    - Include sentiment metadata in analytics decision log when `analytics.save_all_decisions` is true
    - _Requirements: 9.1, 9.2, 9.3, 9.4_

  - [x] 6.3 Write unit tests for error handling and edge cases
    - Test IG API timeout returns neutral (mock requests with timeout)
    - Test news API failure excludes source (mock with exception)
    - Test first call after startup passes through
    - Test initialization failure → permanent pass-through
    - Test config default values when keys missing
    - Test rate limiter respects per-source limits
    - _Requirements: 7.1, 7.2, 7.3, 7.4, 5.3, 5.4, 5.5_

- [x] 7. Pipeline integration and configuration
  - [x] 7.1 Add `sentiment_filter` configuration section to `config/settings_ai.yaml`
    - Add full config block with: `enabled`, `sentiment_threshold`, `cache_ttl_seconds`, `source_weights`, `sources` (ig_client, news sub-configs)
    - Use `${ALPHA_VANTAGE_KEY}` for API key references
    - _Requirements: 6.1, 6.2, 6.5_

  - [x] 7.2 Integrate `SentimentFilter` into `runners/run_ai_autonomous.py`
    - Import and instantiate `SentimentFilter` after `MLDirectionalFilter`
    - Pass `ig_client` instance and config to constructor
    - Call `sentiment_filter.confirm_signal()` after ML filter confirms, before position sizer
    - Skip sentiment filter when ML filter rejects
    - Append sentiment metadata to signal's `meta` dict
    - _Requirements: 8.1, 8.2, 8.3, 8.4, 8.5_

  - [x] 7.3 Write integration tests for pipeline ordering
    - Test: ML confirms → sentiment called → sizer called
    - Test: ML rejects → sentiment NOT called
    - Test: end-to-end with mocked APIs (full signal flow)
    - Test: analytics decision log includes sentiment metadata
    - _Requirements: 8.1, 8.2, 8.3, 8.5, 9.4_

- [x] 8. Final checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP
- Each task references specific requirements for traceability
- Checkpoints ensure incremental validation
- Property tests validate universal correctness properties from the design document
- Unit tests validate specific examples and edge cases
- All test files: `tests/test_sentiment_filter.py` (unit), `tests/test_sentiment_properties.py` (PBT), `tests/test_sentiment_integration.py` (integration)
- The filter follows the same `confirm_signal(signal, df) -> tuple[bool, dict]` interface as `MLDirectionalFilter`
- Fail-open philosophy: the filter never blocks trades due to its own errors

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1.1"] },
    { "id": 1, "tasks": ["1.2", "2.1", "2.3", "3.1"] },
    { "id": 2, "tasks": ["2.2", "2.4", "3.2", "3.3", "3.4"] },
    { "id": 3, "tasks": ["3.5", "5.1"] },
    { "id": 4, "tasks": ["5.2", "5.3", "5.4", "6.1", "6.2"] },
    { "id": 5, "tasks": ["6.3", "7.1"] },
    { "id": 6, "tasks": ["7.2"] },
    { "id": 7, "tasks": ["7.3"] }
  ]
}
```
