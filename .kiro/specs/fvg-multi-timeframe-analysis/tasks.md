# Implementation Plan: FVG Multi-Timeframe Analysis

## Overview

Implement a multi-timeframe Fair Value Gap (FVG) analysis strategy for the IG Scalper bot. The implementation follows a bottom-up approach: build the pure detection/calculation modules first, then the orchestration layer, and finally wire everything together via the strategy class. Property-based tests validate correctness properties from the design document throughout.

## Tasks

- [x] 1. Set up FVG module structure and core data models
  - [x] 1.1 Create FVG dataclass and Bias dataclass in `strategy/fvg_detector.py`
    - Implement the `FVG` dataclass with fields: type, zone_upper, zone_lower, formation_ts, source_tf, fill_status, age_bars
    - Implement `to_dict()` and `from_dict()` serialization methods
    - Implement the `Bias` dataclass with fields: direction, confidence
    - _Requirements: 1.4, 1.7_

  - [x] 1.2 Write property test for FVG round-trip serialization
    - **Property 8: Round-Trip Serialization**
    - Generate random FVG objects with valid fields and verify `FVG.from_dict(f.to_dict()) == f`
    - **Validates: Requirements 1.7**

- [x] 2. Implement FVG detection logic
  - [x] 2.1 Implement `FVGDetector.detect()` method in `strategy/fvg_detector.py`
    - Scan all 3-candle windows through penultimate bar (iloc[-2])
    - Detect bullish FVGs: candle[i].high < candle[i+2].low
    - Detect bearish FVGs: candle[i].low > candle[i+2].high
    - Handle edge cases: < 3 bars returns empty list, NaN rows skipped
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 1.6_

  - [x] 2.2 Write property test for FVG zone validity
    - **Property 1: FVG Zone Validity**
    - Generate random OHLC DataFrames (100-500 rows, gold prices 3000-5000) and verify all detected FVGs have `zone_upper > zone_lower > 0`
    - **Validates: Requirements 1.2, 1.3**

  - [x] 2.3 Write property test for detection completeness
    - **Property 2: Detection Completeness**
    - Generate random OHLC DataFrames and verify `len(fvgs) <= len(df) - 2`
    - **Validates: Requirements 1.1**

  - [x] 2.4 Implement `FVGDetector.update_fill_status()` method
    - Process bars chronologically, update fill status (unfilled → partial → filled)
    - Narrow remaining zone on partial fill
    - Remove fully-filled FVGs from active set
    - Expire FVGs exceeding max_age_bars
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5_

  - [x] 2.5 Write property test for fill monotonicity
    - **Property 3: Fill Monotonicity**
    - Generate random FVG + sequence of bars, verify fill status only moves forward: unfilled → partial → filled
    - **Validates: Requirements 2.1, 2.2**

  - [x] 2.6 Write property test for age expiry
    - **Property 9: Age Expiry**
    - Generate FVGs with varying ages and verify no FVG with `age_bars > max_age` appears in active set
    - **Validates: Requirements 2.4, 2.5**

- [x] 3. Checkpoint - Ensure FVG detection tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 4. Implement Bias calculation
  - [x] 4.1 Implement `BiasCalculator` in `strategy/fvg_bias.py`
    - Implement `calculate_60min_bias()`: count unfilled bullish vs bearish, confidence = abs(bull - bear) / (bull + bear)
    - Implement `adjust_with_15min()`: +0.2 if 15min confirms, -0.3 if opposes (cap 1.0, floor 0.0)
    - Return neutral bias with confidence 0.0 when no unfilled FVGs exist
    - _Requirements: 3.2, 3.3, 3.4, 3.5, 3.6_

  - [x] 4.2 Write property test for bias confidence bounds
    - **Property 4: Bias Confidence Bounds**
    - Generate random lists of FVG objects, verify `0.0 <= confidence <= 1.0` for all bias calculations
    - **Validates: Requirements 3.3, 3.4, 3.5**

- [x] 5. Implement Signal generation
  - [x] 5.1 Implement `SignalGenerator` in `strategy/fvg_signal.py`
    - Select most recent unfilled 5min FVG matching bias direction
    - Calculate entry at FVG zone boundary
    - Calculate stop beyond opposite boundary + configurable buffer
    - Calculate TP at nearest HTF zone or opposite side of triggering zone
    - Discard signals where TP ≤ SL distance
    - Return standard signal format: {"side", "stop_pts", "tp_pts", "meta"}
    - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8, 4.9_

  - [x] 5.2 Write property test for signal-bias alignment
    - **Property 5: Signal-Bias Alignment**
    - Generate random FVGs + random bias, verify BUY only with bullish bias, SELL only with bearish bias, no signal with neutral
    - **Validates: Requirements 4.1, 4.2, 4.7**

  - [x] 5.3 Write property test for risk-reward sanity
    - **Property 6: Risk-Reward Sanity**
    - Generate random signal scenarios, verify all produced signals have `tp_pts > stop_pts > 0`
    - **Validates: Requirements 4.5**

- [x] 6. Checkpoint - Ensure bias and signal tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 7. Implement Cycle scheduling and strategy orchestration
  - [x] 7.1 Implement `CycleScheduler` in `strategy/fvg_scheduler.py`
    - Track interval timing with `should_run()` method
    - Check KILL_SWITCH env var
    - Check daily_lockout from core.risk module
    - Prevent overlapping cycles
    - Log configured interval and timeframe cascade in Europe/Rome timezone
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5_

  - [x] 7.2 Implement `FVGStrategy` in `strategy/fvg_strategy.py`
    - Extend `strategy.base.Strategy` ABC, implement `on_bar(df)`
    - Load configuration from `settings_ai.yaml` under `fvg_strategy` key with defaults
    - Validate config values at initialization (raise ValueError for invalid)
    - Orchestrate analysis cycle: fetch 60min/15min data via multi_data_provider, use 5min df from on_bar()
    - Cache signal between cycles, return cached or None when interval not elapsed
    - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.5, 9.1, 9.2, 9.3, 9.4_

  - [x] 7.3 Implement rate limit compliance in the analysis cycle
    - Verify daily and per-minute budget before fetching each timeframe
    - Skip cycle if insufficient daily budget remains
    - Leverage existing multi_data_provider caching and rate limiting
    - Log budget consumption after each cycle
    - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6_

  - [x] 7.4 Write property test for rate limit invariant
    - **Property 7: Rate Limit Invariant**
    - Simulate rapid on_bar() calls and verify total TwelveData API calls do not exceed 8 per 60s window or 800 per 24h
    - **Validates: Requirements 6.1, 6.2, 6.3**

- [x] 8. Implement logging and observability
  - [x] 8.1 Add comprehensive logging throughout FVG components
    - Log each detected FVG with type, zone, timeframe, timestamp via core.logging_utils
    - Log bias determination (direction, confidence, contributing FVGs)
    - Log signal details (entry zone, stop, TP, alignment rationale)
    - Log cycle skips with reason (rate limits, lockout, neutral bias)
    - All timestamps in Europe/Rome timezone
    - _Requirements: 8.1, 8.2, 8.3, 8.4, 8.5_

- [x] 9. Add FVG strategy configuration to settings_ai.yaml
  - [x] 9.1 Add `fvg_strategy` section to `config/settings_ai.yaml`
    - Add cycle_interval_seconds: 300
    - Add timeframes: ["60min", "15min", "5min"]
    - Add fvg_max_age_bars: 50
    - Add stop_buffer_points: 2.0
    - Add min_bias_confidence: 0.6
    - Add lookback_candles: 200
    - _Requirements: 9.1, 9.2_

- [-] 10. Integration wiring and final validation
  - [x] 10.1 Wire FVGStrategy into strategy module exports
    - Add import to `strategy/__init__.py`
    - Ensure FVGStrategy is discoverable by bot runners
    - _Requirements: 7.1_

  - [x] 10.2 Write integration tests for full analysis cycle
    - Mock DataProvider to return controlled DataFrames
    - Test complete 60min → 15min → 5min cascade produces correct signal
    - Test cache behavior (second on_bar within interval returns cached signal)
    - Test cycle skip on rate limit exhaustion
    - _Requirements: 3.1, 3.7, 3.8, 5.2, 6.3, 7.5_

- [-] 11. Final checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP
- Each task references specific requirements for traceability
- Checkpoints ensure incremental validation
- Property tests validate universal correctness properties from the design document
- Unit tests validate specific examples and edge cases
- The FVGDetector is pure/stateless — ideal for property-based testing with Hypothesis
- All data fetching uses existing `multi_data_provider.get_bars()` with built-in caching and rate limiting
- The strategy follows the existing `strategy.base.Strategy` ABC pattern (on_bar returns signal dict or None)

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1.1"] },
    { "id": 1, "tasks": ["1.2", "2.1"] },
    { "id": 2, "tasks": ["2.2", "2.3", "2.4"] },
    { "id": 3, "tasks": ["2.5", "2.6", "4.1"] },
    { "id": 4, "tasks": ["4.2", "5.1"] },
    { "id": 5, "tasks": ["5.2", "5.3", "7.1"] },
    { "id": 6, "tasks": ["7.2", "7.3", "9.1"] },
    { "id": 7, "tasks": ["7.4", "8.1"] },
    { "id": 8, "tasks": ["10.1", "10.2"] }
  ]
}
```
