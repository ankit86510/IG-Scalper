# Implementation Plan: ML Trading Improvements

## Overview

Bottom-up implementation: build the three independent modules (ML filter, volatility filter, position sizer) with their tests, then wire them into the existing runner loop. Each module is self-contained and independently testable before integration.

## Tasks

- [x] 1. Implement Volatility Regime Filter
  - [x] 1.1 Create `strategy/volatility_filter.py` with `VolatilityRegimeFilter` class
    - Implement `__init__` accepting config dict with keys: enabled, atr_period, lookback_bars, lower_percentile, upper_percentile
    - Implement `compute_atr_ratio(df)` returning ATR(period) / close for penultimate bar
    - Implement `update_history(atr_ratio)` appending to a bounded deque
    - Implement `compute_percentile(current_ratio)` returning rank within history as [0, 100]
    - Implement `allow_trading(df)` returning `(bool, dict)` with pass/block decision and metadata
    - When disabled or history < 20 entries, return `(True, ...)`
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 6.1, 6.2, 6.3, 6.4, 6.5, 7.1, 7.2, 7.3_

  - [x] 1.2 Write property tests for Volatility Filter (`tests/test_volatility_filter.py`)
    - **Property 8: ATR Ratio computation correctness**
    - **Property 9: Volatility history buffer is bounded**
    - **Property 10: Percentile rank correctness**
    - **Property 11: Insufficient volatility history allows trading**
    - **Property 12: Volatility gate blocks outside configured bounds**
    - **Property 13: Disabled volatility filter allows all**
    - **Validates: Requirements 5.1, 5.2, 5.3, 5.4, 6.1, 6.2, 7.3**

  - [x] 1.3 Write unit tests for Volatility Filter
    - Test config parsing with default values
    - Test logging output on block (high/low volatility messages)
    - Test edge case: all identical ATR ratios in history
    - _Requirements: 6.3, 6.4, 7.1, 7.2_

- [x] 2. Implement ML Directional Filter
  - [x] 2.1 Create `strategy/ml_filter.py` with `MLDirectionalFilter` class
    - Implement `__init__` accepting config dict with keys: enabled, probability_threshold, rolling_window_bars, retrain_interval_hours, model_path, model_type
    - Implement `extract_features(df)` returning (n_samples, 6) matrix: RSI(14), ATR_Ratio, SMA_Ratio, ret_1, ret_3, ret_5
    - Implement `generate_labels(df)` returning binary array (1 if next close > current close)
    - Implement `normalize(features)` using stored z-score mean/std from training
    - Implement `train(df)` fitting a logistic regression (or random forest) model, saving scaler stats
    - Implement `predict_probability(df)` returning P(bullish) in [0.0, 1.0] for penultimate bar
    - Implement `confirm_signal(signal, df)` returning `(bool, dict)` with confirmation decision and metadata
    - Implement `should_retrain()` and `retrain(df)` for periodic retraining
    - Implement model save/load via joblib at configured path
    - Set `is_enabled = False` when training data < 100 bars or model missing
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 2.1, 2.2, 2.3, 2.4, 2.5, 3.1, 3.2, 3.3, 3.4, 4.1, 4.2, 4.3_

  - [x] 2.2 Write property tests for ML Filter (`tests/test_ml_filter.py`) — Part 1
    - **Property 1: Feature extraction produces correct dimensions and valid ranges**
    - **Property 2: Label generation correctness**
    - **Property 3: Z-score normalization produces zero-mean unit-variance columns**
    - **Property 4: Insufficient data disables ML filter**
    - **Validates: Requirements 1.2, 1.3, 1.4, 1.5**

  - [x] 2.3 Write property tests for ML Filter — Part 2
    - **Property 5: Prediction probability bounded to [0, 1]**
    - **Property 6: ML signal confirmation follows threshold rule**
    - **Property 7: Disabled ML filter passes all signals**
    - **Validates: Requirements 2.1, 2.2, 2.3, 2.5, 4.3**

  - [x] 2.4 Write unit tests for ML Filter
    - Test model save/load round-trip with joblib
    - Test retrain trigger after configured interval elapses
    - Test log messages on rejection (direction, probability, threshold)
    - Test config parsing with default values
    - _Requirements: 1.6, 3.1, 3.2, 2.4, 4.1_

- [x] 3. Implement Risk-Per-Trade Position Sizer
  - [x] 3.1 Create `core/position_sizer.py` with `RiskPositionSizer` class
    - Implement `__init__` accepting config dict and ig_client reference
    - Config keys: risk_pct_per_trade, equity_refresh_interval_seconds, use_dynamic_sizing, max_size_multiple
    - Implement `refresh_equity()` fetching account balance from IG API with caching on failure
    - Implement `get_equity()` returning current cached equity
    - Implement `calculate_size(stop_distance, pip_value, min_size, size_step)` returning `(float | None, dict)`
    - Formula: size = floor((equity × risk_pct / 100) / (stop_distance × pip_value) / step) × step
    - Return None if size < min_size; cap at min_size × max_size_multiple
    - Handle division by zero (stop=0 or pip_value=0) by returning None
    - _Requirements: 8.1, 8.2, 8.3, 8.4, 8.5, 8.6, 9.1, 9.2, 9.3, 10.1, 10.2, 10.3, 10.4_

  - [x] 3.2 Write property tests for Position Sizer (`tests/test_position_sizer.py`)
    - **Property 14: Position size formula correctness**
    - **Property 15: Position size rounding**
    - **Property 16: Size below minimum rejects trade**
    - **Property 17: Size capped at maximum multiple**
    - **Property 18: Fallback to fixed sizing when dynamic disabled**
    - **Validates: Requirements 8.1, 8.5, 8.6, 10.3, 10.4**

  - [x] 3.3 Write unit tests for Position Sizer
    - Test equity refresh with mocked IG client (success + failure caching)
    - Test config parsing with default values
    - Test edge cases: very large stop, very small equity, zero pip_value
    - _Requirements: 9.1, 9.2, 9.3, 10.1_

- [x] 4. Checkpoint — Core modules complete
  - Ensure all tests pass, ask the user if questions arise.

- [x] 5. Add configuration entries to settings file
  - [x] 5.1 Add `ml_filter` and `volatility_filter` sections to `config/settings_ai.yaml`
    - Add `ml_filter:` block with: enabled, probability_threshold, rolling_window_bars, retrain_interval_hours, model_path, model_type
    - Add `volatility_filter:` block with: enabled, atr_period, lookback_bars, lower_percentile, upper_percentile
    - Add new keys to existing `risk:` block: risk_pct_per_trade, equity_refresh_interval_seconds, use_dynamic_sizing, max_size_multiple
    - _Requirements: 4.1, 4.2, 7.1, 7.2, 10.1, 10.2_

- [x] 6. Integrate filters and sizer into the trading loop
  - [x] 6.1 Wire all three components into `runners/run_ai_autonomous.py`
    - Add imports for MLDirectionalFilter, VolatilityRegimeFilter, RiskPositionSizer
    - Initialize all three components after strategy init using config sections
    - Train ML model on first epic's data at startup (if enabled)
    - Insert volatility filter check before `strategy.on_bar()` in epic loop
    - Insert ML filter confirmation after signal generation, before order
    - Replace fixed sizing with position sizer call (with fallback to `size_by_invested_capital`)
    - Add logging at each filter stage showing filter name, outcome, key metrics
    - Ensure no cooldown when position sizer rejects (skip trade, log reason, continue)
    - Add periodic ML retraining check in the main loop
    - _Requirements: 11.1, 11.2, 11.3, 11.4, 11.5_

  - [x] 6.2 Write integration tests (`tests/test_pipeline_integration.py`)
    - Test full pipeline order: volatility → on_bar → ML → sizing → order
    - Verify execution order via mock call sequence
    - Test each filter independently blocking the pipeline
    - Test no cooldown on position sizer rejection
    - Test disabled filters pass through correctly
    - _Requirements: 11.1, 11.2, 11.3, 11.4, 11.5_

- [x] 7. Final checkpoint — All modules integrated and tested
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP
- Each task references specific requirements for traceability
- Checkpoints ensure incremental validation
- Property tests validate universal correctness properties from the design document
- Unit tests validate specific examples and edge cases
- All modules use fail-open philosophy: errors → pass signals through
- The existing `size_by_invested_capital()` in `core/risk.py` is preserved as fallback

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1.1", "2.1", "3.1"] },
    { "id": 1, "tasks": ["1.2", "1.3", "2.2", "2.3", "2.4", "3.2", "3.3", "5.1"] },
    { "id": 2, "tasks": ["6.1"] },
    { "id": 3, "tasks": ["6.2"] }
  ]
}
```
