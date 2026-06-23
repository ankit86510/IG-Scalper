# Requirements Document

## Introduction

Multi-Timeframe Fair Value Gap (FVG) Analysis Cycle for the IG Scalper bot. This feature adds a repeating analysis cycle that scans multiple timeframes (60min → 15min → 5min) to detect Fair Value Gaps, determines directional bias from higher timeframes, and generates precise trade entry/exit signals on the 5min chart when alignment exists. The cycle runs on a configurable interval, respects TwelveData rate limits, and integrates with the existing strategy architecture.

## Glossary

- **FVG_Detector**: The module responsible for identifying Fair Value Gap patterns in OHLC price data
- **FVG**: A Fair Value Gap — a 3-candle pattern where the wick of candle 1 and candle 3 do not overlap, leaving a gap in price coverage
- **Bullish_FVG**: An FVG where candle 1 high is less than candle 3 low, indicating a gap up (buying imbalance)
- **Bearish_FVG**: An FVG where candle 1 low is greater than candle 3 high, indicating a gap down (selling imbalance)
- **Analysis_Cycle**: One complete execution of the multi-timeframe FVG scan across all configured timeframes
- **Cycle_Scheduler**: The component that triggers Analysis_Cycles at a configurable interval
- **Bias**: The directional expectation (bullish or bearish) derived from higher timeframe FVG analysis
- **FVG_Zone**: The price range between the unfilled boundaries of an FVG (entry zone for trades)
- **FVG_Fill**: When price returns to and trades through an FVG_Zone, partially or fully closing the gap
- **Signal_Generator**: The component that produces trade signals when 5min FVGs align with higher timeframe Bias
- **Rate_Budget_Manager**: The component that tracks and enforces TwelveData API request budgets across the Analysis_Cycle
- **Timeframe_Cascade**: The ordered sequence of timeframes analyzed from highest to lowest (60min → 15min → 5min)
- **FVG_Strategy**: The strategy class extending strategy.base.Strategy that orchestrates the multi-timeframe FVG analysis

## Requirements

### Requirement 1: FVG Detection on a Single Timeframe

**User Story:** As a trader, I want the system to detect Fair Value Gaps in OHLC candle data, so that I can identify price imbalances that may attract future price action.

#### Acceptance Criteria

1. WHEN a DataFrame of OHLC bars is provided, THE FVG_Detector SHALL scan all consecutive 3-candle windows from index 0 through the penultimate completed bar (iloc[-2]), excluding the last bar (which is still forming), and evaluate each window for FVG patterns
2. WHEN candle[i] high is less than candle[i+2] low in a 3-candle window, THE FVG_Detector SHALL classify the pattern as a Bullish_FVG with the zone defined as (candle[i] high, candle[i+2] low) and the formation timestamp set to the timestamp of candle[i+1] (the middle candle)
3. WHEN candle[i] low is greater than candle[i+2] high in a 3-candle window, THE FVG_Detector SHALL classify the pattern as a Bearish_FVG with the zone defined as (candle[i+2] high, candle[i] low) and the formation timestamp set to the timestamp of candle[i+1] (the middle candle)
4. THE FVG_Detector SHALL return a list of detected FVGs, each containing: type (bullish/bearish), zone upper boundary (float), zone lower boundary (float), formation timestamp (datetime), and source timeframe (string matching the input timeframe label)
5. WHEN fewer than 3 bars are available in the DataFrame, THE FVG_Detector SHALL return an empty list without raising an exception
6. IF the DataFrame contains rows with NaN or null values in any OHLC column, THEN THE FVG_Detector SHALL skip those rows during window evaluation without raising an exception
7. THE FVG_Detector SHALL produce FVG objects that survive a serialize-then-deserialize round-trip with all attributes (type, zone boundaries, formation timestamp, source timeframe) comparing equal before and after

### Requirement 2: FVG Freshness and Fill Tracking

**User Story:** As a trader, I want the system to track whether FVGs have been filled by subsequent price action, so that only actionable (unfilled) gaps are considered for trading.

#### Acceptance Criteria

1. WHEN the high of any subsequent bar is greater than or equal to the lower boundary of a Bullish_FVG zone, THE FVG_Detector SHALL update that FVG's fill status to partially filled and narrow the remaining zone to (bar high, zone upper boundary); IF the high of the bar is greater than or equal to the zone upper boundary, THEN THE FVG_Detector SHALL mark the FVG as fully filled
2. WHEN the low of any subsequent bar is less than or equal to the upper boundary of a Bearish_FVG zone, THE FVG_Detector SHALL update that FVG's fill status to partially filled and narrow the remaining zone to (zone lower boundary, bar low); IF the low of the bar is less than or equal to the zone lower boundary, THEN THE FVG_Detector SHALL mark the FVG as fully filled
3. THE FVG_Detector SHALL retain only unfilled and partially-filled FVGs for signal generation and SHALL remove fully-filled FVGs from the active set immediately upon detection of full fill
4. WHEN the number of bars elapsed on the FVG's source timeframe since the FVG formation bar (exclusive of the formation bar itself) exceeds the configurable maximum age (default: 50 bars), THE FVG_Detector SHALL expire and discard that FVG regardless of fill status
5. WHEN multiple subsequent bars are evaluated against an FVG, THE FVG_Detector SHALL process them in chronological order and update fill status incrementally per bar

### Requirement 3: Multi-Timeframe Analysis Cycle

**User Story:** As a trader, I want the system to analyze FVGs across multiple timeframes in a top-down cascade, so that I get directional bias from higher timeframes and precise entries from lower timeframes.

#### Acceptance Criteria

1. THE Analysis_Cycle SHALL process timeframes in descending order: 60min first, then 15min, then 5min, fetching the number of bars specified by the `fvg_strategy.lookback_candles` configuration (default: 200) for each timeframe
2. WHEN the 60min analysis produces one or more unfilled FVGs, THE Analysis_Cycle SHALL establish a Bias by comparing the count of unfilled Bullish_FVGs to unfilled Bearish_FVGs within the lookback window: bullish if Bullish_FVG count exceeds Bearish_FVG count, bearish if Bearish_FVG count exceeds Bullish_FVG count, neutral if counts are equal
3. WHEN the 60min Bias is established, THE Analysis_Cycle SHALL assign a Bias confidence score on a 0.0 to 1.0 scale, calculated as: abs(bullish_count - bearish_count) / (bullish_count + bearish_count)
4. WHEN the 15min analysis produces unfilled FVGs where the majority direction (bullish or bearish by count) matches the 60min Bias direction, THE Analysis_Cycle SHALL increase the Bias confidence by 0.2 (capped at 1.0)
5. WHEN the 15min analysis produces unfilled FVGs where the majority direction opposes the 60min Bias direction, THE Analysis_Cycle SHALL reduce the Bias confidence by 0.3 (floored at 0.0) and skip signal generation for that cycle
6. WHEN no unfilled FVGs exist on the 60min timeframe, THE Analysis_Cycle SHALL set Bias to neutral with confidence 0.0 and skip signal generation for that cycle
7. IF the data fetch for any timeframe returns an empty DataFrame, THEN THE Analysis_Cycle SHALL abort the current cycle, log the failed timeframe, and skip signal generation for that cycle
8. THE Analysis_Cycle SHALL fetch bar data for each timeframe using the existing multi_data_provider.get_bars method with the appropriate timeframe parameter and the configured symbol epic

### Requirement 4: Trade Signal Generation

**User Story:** As a trader, I want the system to generate entry and exit signals when a 5min FVG aligns with the higher timeframe bias, so that I can take high-probability trades.

#### Acceptance Criteria

1. WHEN one or more unfilled 5min Bullish_FVGs exist AND the Bias is bullish AND Bias confidence is at or above the configured min_bias_confidence threshold, THE Signal_Generator SHALL produce a BUY signal using the most recently formed Bullish_FVG, with entry at the FVG_Zone upper boundary (candle 3 low)
2. WHEN one or more unfilled 5min Bearish_FVGs exist AND the Bias is bearish AND Bias confidence is at or above the configured min_bias_confidence threshold, THE Signal_Generator SHALL produce a SELL signal using the most recently formed Bearish_FVG, with entry at the FVG_Zone lower boundary (candle 3 high)
3. WHEN a signal is produced, THE Signal_Generator SHALL set the stop loss beyond the opposite FVG boundary (below candle 1 high for bullish, above candle 1 low for bearish) plus a configurable buffer (default: 2 points)
4. WHEN a signal is produced, THE Signal_Generator SHALL set the take profit at the opposite side of the triggering FVG_Zone (lower boundary for BUY, upper boundary for SELL) or the nearest unfilled higher-timeframe FVG_Zone boundary acting as support/resistance, whichever yields a shorter distance from entry
5. IF the computed take profit distance is less than or equal to the stop loss distance, THEN THE Signal_Generator SHALL discard the signal and return None
6. THE Signal_Generator SHALL return signals in the standard format: {"side": "BUY"/"SELL", "stop_pts": float, "tp_pts": float, "meta": {...}} or None
7. WHEN a 5min FVG direction contradicts the higher timeframe Bias, THE Signal_Generator SHALL return None (no trade)
8. IF the Bias confidence is below the configured min_bias_confidence threshold, THEN THE Signal_Generator SHALL return None regardless of 5min FVG alignment
9. THE Signal_Generator SHALL include in the meta field: the source FVGs (all timeframes), Bias direction, Bias confidence score, and the triggering 5min FVG zone boundaries

### Requirement 5: Cycle Scheduling and Interval Configuration

**User Story:** As a trader, I want the analysis cycle to run automatically at a configurable interval, so that I can continuously monitor for FVG-based opportunities without manual intervention.

#### Acceptance Criteria

1. THE Cycle_Scheduler SHALL execute one Analysis_Cycle at a configurable interval defined in settings_ai.yaml under the key `fvg_strategy.cycle_interval_seconds` (default: 300 seconds)
2. WHEN the previous Analysis_Cycle has not completed before the next scheduled cycle, THE Cycle_Scheduler SHALL skip the pending cycle and log a warning
3. THE Cycle_Scheduler SHALL respect the KILL_SWITCH environment variable and stop scheduling when KILL_SWITCH equals "1"
4. THE Cycle_Scheduler SHALL respect the daily loss lockout from the risk module and pause cycles while lockout is active
5. WHEN the Cycle_Scheduler starts, THE Cycle_Scheduler SHALL log the configured interval and timeframe cascade in Europe/Rome timezone

### Requirement 6: TwelveData Rate Limit Compliance

**User Story:** As a system operator, I want the FVG analysis cycle to respect TwelveData API rate limits, so that the bot never exceeds 800 requests/day or 8 requests/minute.

#### Acceptance Criteria

1. THE Rate_Budget_Manager SHALL calculate the total requests needed per Analysis_Cycle as: number_of_timeframes × number_of_symbols (default: 3 timeframes × 1 symbol = 3 requests per cycle)
2. BEFORE fetching data for a timeframe, THE Rate_Budget_Manager SHALL verify that sufficient daily budget and per-minute budget remain by consulting the existing multi_data_provider budget status
3. IF the daily budget remaining is less than the requests needed for a full Analysis_Cycle, THEN THE Rate_Budget_Manager SHALL skip the cycle and log the budget status
4. IF the per-minute budget is exhausted, THEN THE Rate_Budget_Manager SHALL wait until the sliding window clears before proceeding (leveraging existing multi_data_provider rate limiting)
5. THE Rate_Budget_Manager SHALL use the existing cache in multi_data_provider to avoid redundant requests when bar data has not changed since the last fetch
6. THE Rate_Budget_Manager SHALL log the budget consumption after each Analysis_Cycle with daily_used, daily_remaining, and requests_this_cycle

### Requirement 7: Integration with Strategy Architecture

**User Story:** As a developer, I want the FVG multi-timeframe strategy to follow the existing strategy architecture, so that it can be used interchangeably with other strategies in the bot runners.

#### Acceptance Criteria

1. THE FVG_Strategy SHALL extend the strategy.base.Strategy abstract base class and implement the on_bar(df) method
2. WHEN on_bar(df) is called with a 5min DataFrame, THE FVG_Strategy SHALL trigger an internal Analysis_Cycle using the provided DataFrame for the 5min timeframe and fetching 60min and 15min data separately
3. THE FVG_Strategy SHALL accept configuration parameters via its constructor: cycle_interval_seconds, timeframe_cascade list, fvg_max_age_bars, stop_buffer_points, and min_bias_confidence
4. THE FVG_Strategy SHALL be configurable in settings_ai.yaml under a dedicated `fvg_strategy` section
5. WHEN on_bar(df) is called more frequently than the configured cycle interval, THE FVG_Strategy SHALL return the cached signal from the last completed Analysis_Cycle (or None if no signal exists)

### Requirement 8: Logging and Observability

**User Story:** As a system operator, I want comprehensive logging of FVG detection and analysis decisions, so that I can review the bot's reasoning and diagnose issues.

#### Acceptance Criteria

1. THE FVG_Strategy SHALL log each detected FVG with its type, zone boundaries, source timeframe, and formation timestamp using the core.logging_utils module
2. THE FVG_Strategy SHALL log the Bias determination at the end of each Analysis_Cycle including direction, confidence score, and contributing FVGs
3. WHEN a trade signal is generated, THE FVG_Strategy SHALL log the signal details including entry zone, stop distance, take profit distance, and alignment rationale
4. WHEN a cycle is skipped due to rate limits, lockout, or neutral bias, THE FVG_Strategy SHALL log the skip reason
5. THE FVG_Strategy SHALL log all timestamps in Europe/Rome timezone consistent with the existing logging configuration

### Requirement 9: Configuration Schema

**User Story:** As a developer, I want all FVG strategy parameters to be configurable via YAML, so that I can tune the strategy without code changes.

#### Acceptance Criteria

1. THE FVG_Strategy SHALL read its configuration from `config/settings_ai.yaml` under the `fvg_strategy` key
2. THE FVG_Strategy configuration SHALL include: cycle_interval_seconds (int, default 300), timeframes (list, default ["60min", "15min", "5min"]), fvg_max_age_bars (int, default 50), stop_buffer_points (float, default 2.0), min_bias_confidence (float, default 0.6), and lookback_candles (int, default 200)
3. IF a required configuration key is missing, THEN THE FVG_Strategy SHALL use the documented default value and log a warning
4. IF an invalid value is provided for a configuration key (negative interval, empty timeframe list), THEN THE FVG_Strategy SHALL raise a ValueError at initialization with a descriptive message
