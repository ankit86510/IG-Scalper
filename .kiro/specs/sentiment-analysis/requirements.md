# Requirements Document

## Introduction

Gold Sentiment Analysis Filter for the IG Scalper trading bot. This module adds a sentiment-based signal filter that evaluates market sentiment from multiple sources (IG client positioning, news APIs, Fear & Greed Index) and confirms or rejects trade signals based on sentiment alignment. It integrates into the existing pipeline after the ML Directional Filter and before the Position Sizer, following the same fail-open philosophy used by other filters.

## Glossary

- **Sentiment_Filter**: The Python module that evaluates Gold-related market sentiment and confirms or rejects trade signals based on configurable thresholds
- **IG_Client_Sentiment**: The percentage of IG retail clients holding long vs short positions on Gold, retrieved via the IG REST API `/clientsentiment` endpoint
- **Contrarian_Signal**: A sentiment interpretation where extreme retail positioning (e.g., 80%+ long) suggests price will move in the opposite direction
- **Sentiment_Score**: A normalized floating-point value between -1.0 (extremely bearish) and +1.0 (extremely bullish) representing aggregated sentiment
- **News_Sentiment_Provider**: A component that fetches Gold-related news headlines and assigns a sentiment polarity score
- **Sentiment_Cache**: A time-based in-memory cache that stores the latest sentiment data to avoid redundant API calls within a configurable TTL window
- **Pipeline**: The sequential chain of filters and modules that process trade signals: Volatility Filter → Strategy → ML Filter → Sentiment Filter → Position Sizer → Order Execution
- **Fail_Open**: The design principle where errors, timeouts, or unavailable data cause the filter to pass signals through unmodified rather than blocking them
- **Sentiment_Threshold**: The minimum absolute Sentiment_Score required for the filter to confirm a signal in the aligned direction

## Requirements

### Requirement 1: Sentiment Data Retrieval from IG Client Positioning

**User Story:** As a trader, I want the bot to retrieve IG client sentiment data for Gold, so that it can detect extreme retail positioning as a contrarian indicator.

#### Acceptance Criteria

1. WHEN a sentiment update is requested, THE Sentiment_Filter SHALL retrieve the current percentage of IG clients holding long and short positions on Gold via the IG REST API `/clientsentiment` endpoint
2. WHEN the IG client sentiment response contains a long percentage above 70%, THE Sentiment_Filter SHALL interpret the sentiment as contrarian-bearish and assign a negative Sentiment_Score proportional to the long percentage
3. WHEN the IG client sentiment response contains a short percentage above 70%, THE Sentiment_Filter SHALL interpret the sentiment as contrarian-bullish and assign a positive Sentiment_Score proportional to the short percentage
4. WHEN the IG client sentiment long and short percentages are both between 30% and 70%, THE Sentiment_Filter SHALL assign a neutral Sentiment_Score of 0.0
5. IF the IG client sentiment API call fails or times out within 10 seconds, THEN THE Sentiment_Filter SHALL log the error and return a neutral Sentiment_Score of 0.0

### Requirement 2: News Sentiment Data Retrieval

**User Story:** As a trader, I want the bot to analyze Gold-related news headlines for sentiment, so that it can detect strong bullish or bearish market narratives.

#### Acceptance Criteria

1. WHEN a sentiment update is requested and a News_Sentiment_Provider is configured, THE Sentiment_Filter SHALL fetch the latest Gold-related news headlines from the configured provider (Alpha Vantage News Sentiment or NewsAPI)
2. WHEN news headlines are retrieved, THE Sentiment_Filter SHALL compute a weighted average sentiment polarity from the headlines published within the last 4 hours
3. WHEN fewer than 3 relevant headlines are available within the lookback window, THE Sentiment_Filter SHALL treat news sentiment as neutral (0.0) due to insufficient data
4. IF the news API call fails, returns an error, or times out within 15 seconds, THEN THE Sentiment_Filter SHALL log the error and exclude news sentiment from the aggregated score without blocking the signal

### Requirement 3: Sentiment Score Aggregation

**User Story:** As a trader, I want multiple sentiment sources combined into a single score, so that no single unreliable source dominates the decision.

#### Acceptance Criteria

1. THE Sentiment_Filter SHALL compute the final Sentiment_Score as a weighted average of all available source scores, using configurable weights defined in `config/settings_ai.yaml`
2. WHEN only one sentiment source returns valid data, THE Sentiment_Filter SHALL use that single source score as the final Sentiment_Score
3. WHEN all sentiment sources return neutral or fail, THE Sentiment_Filter SHALL assign a final Sentiment_Score of 0.0
4. THE Sentiment_Filter SHALL normalize the final Sentiment_Score to the range [-1.0, +1.0] by clamping values that exceed the bounds

### Requirement 4: Signal Confirmation Logic

**User Story:** As a trader, I want the sentiment filter to confirm or reject trade signals based on sentiment alignment, so that trades are only taken when sentiment supports the direction.

#### Acceptance Criteria

1. WHEN a BUY signal is received and the final Sentiment_Score is greater than or equal to the negative of the configured Sentiment_Threshold, THE Sentiment_Filter SHALL confirm the signal
2. WHEN a BUY signal is received and the final Sentiment_Score is less than the negative of the configured Sentiment_Threshold, THE Sentiment_Filter SHALL reject the signal and log the reason
3. WHEN a SELL signal is received and the final Sentiment_Score is less than or equal to the configured Sentiment_Threshold, THE Sentiment_Filter SHALL confirm the signal
4. WHEN a SELL signal is received and the final Sentiment_Score is greater than the configured Sentiment_Threshold, THE Sentiment_Filter SHALL reject the signal and log the reason
5. THE Sentiment_Filter SHALL return a tuple of (confirmed: bool, metadata: dict) consistent with the MLDirectionalFilter interface
6. THE Sentiment_Filter SHALL always resolve every received signal to either a confirmed or rejected state, defaulting to confirmed when the score is exactly neutral (0.0)

### Requirement 5: Caching and Rate Limit Compliance

**User Story:** As a trader, I want sentiment data cached between polls, so that the bot does not exceed API rate limits while polling every 2 minutes.

#### Acceptance Criteria

1. THE Sentiment_Cache SHALL store the most recent Sentiment_Score and source data with a configurable TTL (default 300 seconds)
2. WHEN a sentiment check is explicitly requested and the Sentiment_Cache contains data newer than the configured TTL, THE Sentiment_Filter SHALL return the cached score without making new API calls
3. THE Sentiment_Filter SHALL limit IG client sentiment API calls to a maximum of 1 request per 5 minutes to remain within IG API fair-use limits
4. WHEN using NewsAPI as a provider, THE Sentiment_Filter SHALL limit requests to a maximum of 4 per hour (96 per day) to remain within the free tier of 100 requests per day with safety margin
5. WHEN using Alpha Vantage News Sentiment as a provider, THE Sentiment_Filter SHALL limit requests to a maximum of 5 per hour to remain within free tier limits

### Requirement 6: Configuration via YAML

**User Story:** As a trader, I want to configure the sentiment filter through the existing YAML config file, so that I can tune thresholds and enable/disable the filter without code changes.

#### Acceptance Criteria

1. THE Sentiment_Filter SHALL read its configuration from the `sentiment_filter` section of `config/settings_ai.yaml`
2. THE Sentiment_Filter SHALL support the following configuration keys: `enabled` (bool), `sentiment_threshold` (float), `cache_ttl_seconds` (int), `sources` (list of provider configs), and `source_weights` (dict of source-name to weight)
3. WHEN the `enabled` key is set to false, THE Sentiment_Filter SHALL pass all signals through without making any sentiment-related API calls and without attaching sentiment metadata to the signal
4. WHEN a configuration key is missing, THE Sentiment_Filter SHALL use a documented default value
5. THE Sentiment_Filter SHALL support environment variable expansion via `${VAR}` syntax for API keys within its configuration section

### Requirement 7: Fail-Open Error Handling

**User Story:** As a trader, I want the sentiment filter to never block trading due to its own errors, so that connectivity issues or API outages do not prevent profitable trades.

#### Acceptance Criteria

1. IF the Sentiment_Filter encounters any unhandled exception during signal confirmation, THEN THE Sentiment_Filter SHALL log the error at WARNING level and return (confirmed=True, metadata with error details)
2. IF all configured sentiment sources are unavailable or return errors, THEN THE Sentiment_Filter SHALL pass the signal through with a neutral score and log a warning
3. WHILE the Sentiment_Filter has never successfully retrieved sentiment data since startup, THE Sentiment_Filter SHALL pass all signals through and log that sentiment data is pending
4. IF the Sentiment_Filter initialization fails due to missing dependencies or configuration errors, THEN THE Sentiment_Filter SHALL log the error and operate in pass-through mode for the lifetime of the process, preventing any subsequent attempts at normal sentiment filtering

### Requirement 8: Pipeline Integration

**User Story:** As a trader, I want the sentiment filter placed after the ML filter in the pipeline, so that only ML-confirmed signals are checked against sentiment.

#### Acceptance Criteria

1. THE Pipeline SHALL invoke the Sentiment_Filter after the MLDirectionalFilter and before the RiskPositionSizer
2. WHEN the MLDirectionalFilter rejects a signal, THE Pipeline SHALL skip the Sentiment_Filter for that signal
3. WHEN the MLDirectionalFilter confirms a signal, THE Pipeline SHALL invoke the Sentiment_Filter, and WHEN the Sentiment_Filter confirms, THE Pipeline SHALL proceed to the RiskPositionSizer
4. THE Sentiment_Filter SHALL accept the same signal dict format as MLDirectionalFilter.confirm_signal (containing keys: `side`, `stop_pts`, `tp_pts`, `meta`)
5. THE Sentiment_Filter SHALL include its metadata (score, sources used, cache_hit, confirmed) in the signal's `meta` dict for logging and dashboard display

### Requirement 9: Logging and Observability

**User Story:** As a trader, I want sentiment decisions logged clearly, so that I can audit why trades were taken or rejected.

#### Acceptance Criteria

1. WHEN the Sentiment_Filter confirms a signal, THE Sentiment_Filter SHALL log the decision at INFO level including the Sentiment_Score, contributing source scores, and the direction
2. WHEN the Sentiment_Filter rejects a signal, THE Sentiment_Filter SHALL log the decision at INFO level including the Sentiment_Score, the threshold, and the reason for rejection
3. THE Sentiment_Filter SHALL log each successful sentiment data refresh at DEBUG level including source, score, and cache TTL remaining
4. THE Sentiment_Filter SHALL include sentiment metadata in the analytics decision log when `analytics.save_all_decisions` is true in the configuration
