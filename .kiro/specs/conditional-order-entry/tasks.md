# Implementation Plan: Conditional Order Entry

## Overview

This plan implements the `ConditionalOrderManager` module that replaces immediate market order execution with strategic pending (working) orders placed at S/R-derived price levels. The implementation proceeds from configuration and data models, through core calculation logic, to IG API integration, polling/lifecycle management, and finally wiring into the existing signal pipeline.

## Tasks

- [x] 1. Configuration and data models
  - [x] 1.1 Add `conditional_orders` section to `config/settings_ai.yaml`
    - Add the `conditional_orders` block with keys: `enabled` (bool), `buffer_points` (float), `order_expiry_seconds` (int), `max_entry_distance_points` (float)
    - Use defaults: `enabled: true`, `buffer_points: 2.0`, `order_expiry_seconds: 300`, `max_entry_distance_points: 30.0`
    - _Requirements: 7.1, 7.4, 7.5, 7.6_

  - [x] 1.2 Create `broker/conditional_order_manager.py` with data models and class skeleton
    - Define `TrackedOrder` dataclass with fields: epic, deal_id, direction, entry_level, stop_distance, tp_distance, size, currency_code, placed_at, expiry_at, confidence, patterns, cancel_retry_count
    - Define `ConditionalOrderManager` class with `__init__` accepting ig_client, config, position_manager, trailing_manager, sr_detector, log
    - Implement configuration validation in `__init__`: reject if required keys missing or `order_expiry_seconds` outside [60, 86400], log error and set `self.enabled = False`
    - Initialize `tracked_orders: Dict[str, TrackedOrder]` and `active_signals: Dict[str, str]`
    - _Requirements: 7.1, 7.3, 3.6_

  - [x] 1.3 Write property test for configuration validation (Property 11)
    - **Property 11: Configuration Validation**
    - Test that missing required keys or out-of-range `order_expiry_seconds` disables conditional orders
    - Use Hypothesis strategies for arbitrary config dicts with missing/invalid values
    - **Validates: Requirements 3.6, 7.3**

- [x] 2. Entry level calculation
  - [x] 2.1 Implement `calculate_entry_level` method
    - For BUY: find nearest resistance above mid_price, add buffer_points
    - For SELL: find nearest support below mid_price, subtract buffer_points
    - Return `None` if no suitable S/R level exists
    - Select nearest by smallest absolute distance from mid_price among valid candidates
    - _Requirements: 1.1, 1.2, 1.5, 1.7_

  - [x] 2.2 Write property test for entry level calculation (Property 1)
    - **Property 1: Entry Level Calculation**
    - For arbitrary S/R levels, mid-price, direction, and buffer in [0.5, 50.0], verify the calculated entry equals nearest_resistance + buffer (BUY) or nearest_support - buffer (SELL)
    - **Validates: Requirements 1.1, 1.2, 1.7**

  - [x] 2.3 Implement max distance validation in `process_signal`
    - Reject signal if `|entry_level - mid_price| > max_entry_distance_points`
    - Log WARNING with epic, calculated entry level, current price, distance, and configured max
    - _Requirements: 1.6, 8.4_

  - [x] 2.4 Write property test for max distance rejection (Property 2)
    - **Property 2: Max Distance Rejection**
    - For arbitrary entry level and mid-price, verify rejection occurs iff distance exceeds configured max
    - **Validates: Requirements 1.6**

- [x] 3. Stop/TP calculation and order payload construction
  - [x] 3.1 Implement stop and take-profit calculation logic
    - Final stop = `max(ATR * stop_multiplier, min_stop_pts, market_min_stop)`
    - When `use_tp_limit` is true: TP = `final_stop * rr_take`
    - When `use_tp_limit` is false: TP = None
    - Recalculate TP after market rules enforcement to maintain R:R ratio
    - _Requirements: 2.2, 2.3, 2.4, 2.7, 2.9_

  - [x] 3.2 Write property test for stop/TP calculation (Property 3)
    - **Property 3: Stop and Take-Profit Calculation with Market Rules**
    - For arbitrary ATR, multiplier, min_stop_pts, market_min, and rr_take, verify final stop and TP
    - **Validates: Requirements 2.2, 2.3, 2.7, 2.9**

  - [x] 3.3 Implement `build_order_payload` method
    - Construct IG API payload with: `orderType: "STOP"`, `timeInForce: "GOOD_TILL_DATE"`, epic, direction, level, size, stopDistance, limitDistance (optional), currencyCode, goodTillDate, expiry, forceOpen, guaranteedStop
    - Compute `goodTillDate` as current UTC + `order_expiry_seconds` formatted as ISO 8601
    - _Requirements: 2.1, 2.5, 2.6, 3.1_

  - [x] 3.4 Write property test for expiry timestamp calculation (Property 4)
    - **Property 4: Expiry Timestamp Calculation**
    - For arbitrary UTC time and expiry_seconds in [60, 86400], verify goodTillDate string is correct
    - **Validates: Requirements 2.5, 3.1**

  - [x] 3.5 Write property test for order payload construction (Property 5)
    - **Property 5: Order Payload Construction**
    - For arbitrary valid signal parameters, verify payload contains correct orderType, timeInForce, and all fields mapped correctly
    - **Validates: Requirements 2.1, 2.6**

- [x] 4. Checkpoint — Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 5. IGClient extensions
  - [x] 5.1 Add `place_working_order` method to `broker/ig_client.py`
    - `POST /workingorders/otc` with Version 2 header
    - Accept: epic, direction, level, size, stop_distance, limit_distance (optional), good_till_date, currency_code, expiry
    - _Requirements: 2.1_

  - [x] 5.2 Add `get_working_orders` method to `broker/ig_client.py`
    - `GET /workingorders` with Version 2 header
    - Return full response JSON with working orders list
    - _Requirements: 3.4, 4.7_

  - [x] 5.3 Add `delete_working_order` method to `broker/ig_client.py`
    - `DELETE /workingorders/otc/{dealId}` with Version 2 header
    - Return response JSON
    - _Requirements: 4.1, 4.3_

- [x] 6. Order lifecycle and polling
  - [x] 6.1 Implement `process_signal` method (main entry point)
    - Check if position already open for epic (reject if so)
    - Check if pending order exists for same direction (skip duplicate)
    - Check if pending order exists for opposite direction (cancel first, then place new)
    - Calculate entry level, validate distance, build payload, place order
    - Handle fallback to market order when no S/R level found
    - Track placed order in `tracked_orders`
    - Log all actions per Requirements 8.1, 8.4, 8.5
    - _Requirements: 1.3, 1.4, 4.1, 4.2, 5.1, 5.2, 5.3, 8.1, 8.4, 8.5_

  - [x] 6.2 Write property test for signal direction handling (Property 6)
    - **Property 6: Signal Direction Handling**
    - For arbitrary epic with existing order direction D and new signal direction D', verify cancel iff D ≠ D' and keep iff D = D'
    - **Validates: Requirements 4.1, 4.2**

  - [x] 6.3 Write property test for one-order-per-epic invariant (Property 7)
    - **Property 7: One-Order-Per-Epic Invariant**
    - For arbitrary sequence of placements/cancellations, verify at most one tracked order per epic
    - Use stateful testing with Hypothesis RuleBasedStateMachine
    - **Validates: Requirements 5.1**

  - [x] 6.4 Implement `poll_orders` method
    - Call `ig_client.get_working_orders()` every polling cycle
    - Detect filled orders (status change), call `_handle_fill`
    - Detect expired/cancelled orders, remove from tracking, log per Requirement 8.3
    - Detect signal reversals by comparing active_signals to tracked order directions
    - Handle API errors gracefully (retain state, retry next cycle)
    - _Requirements: 3.3, 3.4, 3.5, 4.7, 8.2, 8.3_

  - [x] 6.5 Write property test for order lifecycle state machine (Property 8)
    - **Property 8: Order Lifecycle State Machine**
    - Verify that filled orders are removed from tracking and new signals rejected while position is open
    - **Validates: Requirements 5.2, 5.3, 5.4**

  - [x] 6.6 Write property test for expired order removal (Property 9)
    - **Property 9: Expired Order Removal**
    - For any tracked order reported as cancelled/expired by API, verify removal from internal tracking
    - **Validates: Requirements 3.3**

  - [x] 6.7 Implement `cancel_order` and `cancel_all_orders` methods
    - `cancel_order`: DELETE via ig_client, handle errors with retry count (max 3), log per Requirement 4.5/4.6
    - `cancel_all_orders`: iterate all tracked orders and cancel each (for kill switch / daily loss limit)
    - Implement 30-second timeout for kill switch bulk cancellation
    - _Requirements: 4.1, 4.3, 4.4, 4.5, 4.6, 8.3_

  - [x] 6.8 Write property test for cancellation retry logic (Property 10)
    - **Property 10: Cancellation Retry Logic**
    - For arbitrary cancel failure sequences, verify retry count increments and stops at 3
    - **Validates: Requirements 4.5, 4.6**

- [x] 7. Checkpoint — Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 8. Fill handoff and position management
  - [x] 8.1 Implement `_handle_fill` method
    - Extract fill price from `level` field and deal ID from `dealId` field of IG API response
    - Call `position_manager.add_position()` with fill price, deal_id, direction, size, stop_distance, tp_distance, confidence, patterns
    - When `use_trailing_stop` is true: call `trailing_manager.initialize()` with epic, deal_id, fill_price, direction, stop_distance, activation_pct, trailing_distance_pct
    - Remove order from `tracked_orders`
    - Log fill event per Requirement 8.2
    - _Requirements: 6.1, 6.2, 6.3, 6.4, 8.2_

  - [x] 8.2 Write property test for fill handoff correctness (Property 12)
    - **Property 12: Fill Handoff Correctness**
    - For arbitrary fill events, verify `add_position` is called with parameters matching original order signal data and fill price
    - **Validates: Requirements 6.1**

- [x] 9. Integration with signal pipeline
  - [x] 9.1 Wire `ConditionalOrderManager` into `runners/run_ai_autonomous.py`
    - Import `ConditionalOrderManager` from `broker.conditional_order_manager`
    - Instantiate after `PositionManager` and `TrailingStopManager` initialization
    - Read `conditional_orders` config section; if `enabled` is false or missing/invalid, skip instantiation
    - At the order placement point in signal processing: if conditional orders enabled, call `manager.process_signal(...)` instead of `ig.place_order(...)`
    - Add `manager.poll_orders()` call in the main loop (every 60 seconds)
    - Wire kill switch handling to call `manager.cancel_all_orders("kill_switch")`
    - Wire daily loss limit to call `manager.cancel_all_orders("daily_loss_limit")`
    - _Requirements: 7.2, 4.3, 4.4_

  - [x] 9.2 Implement API error handling in `process_signal`
    - Catch HTTP errors, timeouts, and rejection responses from IG API on order placement
    - Log error type and response details, skip signal, continue next cycle
    - _Requirements: 2.8_

  - [x] 9.3 Write unit tests for integration scenarios
    - Test fallback to market order when no S/R level found
    - Test `use_tp_limit=false` omits TP from payload
    - Test `use_trailing_stop=true/false` branching on fill
    - Test API error handling with mocked failures
    - Test kill switch cancellation flow
    - Test daily loss limit bulk cancellation
    - Test logging output correctness (level, fields)
    - _Requirements: 1.3, 1.4, 2.4, 2.8, 4.3, 4.4, 8.1–8.5_

- [x] 10. Final checkpoint — Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP
- Each task references specific requirements for traceability
- Checkpoints ensure incremental validation after core logic and after integration
- Property tests validate universal correctness properties from the design document using Hypothesis
- Unit tests validate specific examples, edge cases, and mocked API interactions
- The `ConditionalOrderManager` is toggled via `conditional_orders.enabled` — when false, existing market order logic is unchanged
- All 12 correctness properties from the design are covered by property test sub-tasks

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1.1", "1.2"] },
    { "id": 1, "tasks": ["1.3", "2.1", "5.1", "5.2", "5.3"] },
    { "id": 2, "tasks": ["2.2", "2.3", "3.1"] },
    { "id": 3, "tasks": ["2.4", "3.2", "3.3"] },
    { "id": 4, "tasks": ["3.4", "3.5", "6.1"] },
    { "id": 5, "tasks": ["6.2", "6.3", "6.4", "6.7"] },
    { "id": 6, "tasks": ["6.5", "6.6", "6.8", "8.1"] },
    { "id": 7, "tasks": ["8.2", "9.1"] },
    { "id": 8, "tasks": ["9.2", "9.3"] }
  ]
}
```
