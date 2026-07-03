# Requirements Document

## Introduction

The Conditional Order Entry feature replaces immediate market order execution with strategic pending (working) orders placed at S/R-derived price levels. When a signal is detected, the bot calculates an optimal entry level based on support/resistance analysis and places a stop entry order that only fills when price confirms the move by reaching and surpassing the order level. This eliminates stale entries from analysis delay and provides directional confirmation before committing capital.

## Glossary

- **Conditional_Order_Manager**: The component responsible for calculating entry levels, placing working orders via the IG API, monitoring their status, and cancelling them when conditions change or they expire.
- **Working_Order**: A pending order on the IG platform (type STOP or LIMIT) placed via `POST /workingorders/otc` that remains unfilled until price reaches the specified level.
- **Stop_Entry_Order**: A working order of type STOP that triggers when price moves through the specified level in the signal direction (above for BUY, below for SELL), confirming a breakout/breakdown.
- **Entry_Level**: The calculated price at which a stop entry order is placed, derived from the nearest S/R zone plus a configurable buffer.
- **Buffer**: A configurable distance (in points) added beyond the S/R level to confirm a genuine breakout rather than a touch-and-reject.
- **Order_Expiry**: The maximum duration a working order remains active before automatic cancellation if unfilled.
- **Signal_Reversal**: A condition where the bot detects a new signal in the opposite direction for the same epic, invalidating any pending order for that epic.
- **S/R_Zone**: A support or resistance price zone detected by the existing SupportResistanceDetector class.
- **Trailing_Stop_Manager**: The existing component that manages trailing stop adjustments on filled positions.
- **Position_Manager**: The existing component that tracks active positions per epic.
- **IG_Client**: The existing broker API client that communicates with the IG REST API.

## Requirements

### Requirement 1: Entry Level Calculation

**User Story:** As a trader, I want the bot to calculate optimal entry levels based on S/R zones, so that orders are placed at strategic price levels that confirm directional moves.

#### Acceptance Criteria

1. WHEN a BUY signal is detected, THE Conditional_Order_Manager SHALL calculate the entry level as the nearest resistance level above the current mid-price plus the configured buffer (0.5 to 50.0 points, default defined in settings YAML).
2. WHEN a SELL signal is detected, THE Conditional_Order_Manager SHALL calculate the entry level as the nearest support level below the current mid-price minus the configured buffer (0.5 to 50.0 points, default defined in settings YAML).
3. IF no resistance level exists above current price for a BUY signal, THEN THE Conditional_Order_Manager SHALL place a market order at current price with the same stop-loss and take-profit distances that would have applied to the conditional order.
4. IF no support level exists below current price for a SELL signal, THEN THE Conditional_Order_Manager SHALL place a market order at current price with the same stop-loss and take-profit distances that would have applied to the conditional order.
5. THE Conditional_Order_Manager SHALL use the S/R levels returned by the existing SupportResistanceDetector class without modifying S/R detection logic.
6. WHEN the distance between current mid-price and the calculated entry level exceeds the configured maximum distance (1.0 to 200.0 points, default defined in settings YAML), THE Conditional_Order_Manager SHALL reject the signal and log a message indicating the calculated distance and the configured maximum.
7. THE Conditional_Order_Manager SHALL select the nearest S/R level by choosing the resistance (for BUY) or support (for SELL) level with the smallest absolute distance from the current mid-price, where mid-price is defined as (bid + ask) / 2.

### Requirement 2: Working Order Placement

**User Story:** As a trader, I want the bot to place stop entry orders on the IG platform, so that trades only execute when price confirms the signal direction.

#### Acceptance Criteria

1. WHEN a valid entry level is calculated, THE Conditional_Order_Manager SHALL place a working order via `POST /workingorders/otc` with order type STOP and the calculated entry level as the order level.
2. WHEN placing a working order, THE Conditional_Order_Manager SHALL include a stop-loss distance calculated as ATR(ai_strategy.atr_period) multiplied by ai_strategy.stop_multiplier, floored at ai_strategy.min_stop_pts (points), using the same price data used for signal generation.
3. WHEN `execution.use_tp_limit` is true in configuration, THE Conditional_Order_Manager SHALL include a take-profit (limit) distance on the working order calculated as stop_distance multiplied by ai_strategy.rr_take.
4. WHEN `execution.use_tp_limit` is false in configuration, THE Conditional_Order_Manager SHALL omit the take-profit distance from the working order.
5. WHEN placing a working order, THE Conditional_Order_Manager SHALL set the `timeInForce` field to `GOOD_TILL_DATE` and the `goodTillDate` field to the current time plus `conditional_orders.order_expiry_seconds`, formatted as an ISO 8601 UTC string (yyyy-MM-dd'T'HH:mm:ss).
6. WHEN placing a working order, THE Conditional_Order_Manager SHALL set the `direction`, `epic`, `size`, and `currencyCode` fields using the values derived from the signal direction, the configured symbol epic, the risk-based position sizing calculation, and the instrument currency respectively.
7. WHEN the calculated stop distance or deal size is below the IG market dealing rules minimum, THE Conditional_Order_Manager SHALL adjust the stop distance upward to `minNormalStopOrLimitDistance` and the deal size upward to `minDealSize` before placing the order.
8. IF the IG API returns an HTTP error, a timeout, or a rejection response on order placement, THEN THE Conditional_Order_Manager SHALL log the error type and response details, skip the signal, and continue processing the next polling cycle without crashing.
9. WHEN the stop distance is adjusted upward by market rules enforcement, THE Conditional_Order_Manager SHALL recalculate the take-profit distance (if applicable) to maintain the configured ai_strategy.rr_take ratio relative to the enforced stop distance.

### Requirement 3: Order Expiry and Automatic Cancellation

**User Story:** As a trader, I want unfilled orders to automatically expire after a configurable duration, so that stale orders do not execute after conditions have changed.

#### Acceptance Criteria

1. WHEN placing a working order, THE Conditional_Order_Manager SHALL set the `goodTillDate` field to a timestamp equal to current time plus the configured `order_expiry_seconds` value, where `order_expiry_seconds` is an integer between 60 and 86400 (default: 300).
2. WHEN a working order reaches its `goodTillDate` without being filled, THE IG platform SHALL automatically cancel the order.
3. WHEN the polling cycle detects that a tracked working order has been cancelled or expired on the IG platform, THE Conditional_Order_Manager SHALL remove the order from its internal tracking state and log the expiry event including the order deal reference and epic.
4. THE Conditional_Order_Manager SHALL poll the status of all tracked working orders every 60 seconds to detect fills, expirations, and cancellations.
5. IF the IG REST API returns an error or is unreachable during a polling cycle, THEN THE Conditional_Order_Manager SHALL retain the order in its internal tracking state unchanged and retry on the next polling cycle.
6. IF the configured `order_expiry_seconds` value is outside the range 60–86400, THEN THE Conditional_Order_Manager SHALL reject the configuration at startup and log an error message indicating the invalid value.

### Requirement 4: Pending Order Monitoring and Conditional Cancellation

**User Story:** As a trader, I want the bot to cancel pending orders when conditions change, so that outdated orders do not fill on stale signals.

#### Acceptance Criteria

1. WHEN a new signal is detected for an epic that already has a pending working order in the opposite direction, THE Conditional_Order_Manager SHALL cancel the existing working order via `DELETE /workingorders/otc/{dealId}` and SHALL NOT place the new order until the cancellation response is received with a success status.
2. WHEN a new signal is detected for an epic that already has a pending working order in the same direction, THE Conditional_Order_Manager SHALL keep the existing order and skip placing a duplicate regardless of differences in price level or stop/limit distances.
3. WHEN the daily loss limit is reached, THE Conditional_Order_Manager SHALL cancel all pending working orders for all epics by iterating through orders returned by `GET /workingorders` and issuing individual `DELETE /workingorders/otc/{dealId}` requests for each.
4. WHEN the KILL_SWITCH environment variable equals "1", THE Conditional_Order_Manager SHALL cancel all pending working orders before shutting down and SHALL complete or abandon all cancellation attempts within 30 seconds of detecting the kill switch.
5. IF the IG API returns an error when attempting to cancel a working order, THEN THE Conditional_Order_Manager SHALL log the error at WARNING level and retry cancellation on the next polling cycle, up to a maximum of 3 consecutive retry attempts per order.
6. IF a working order cancellation has failed for 3 consecutive polling cycles, THEN THE Conditional_Order_Manager SHALL log the failure at ERROR level, skip further retries for that order, and continue processing remaining orders.
7. WHEN the Conditional_Order_Manager starts a polling cycle, THE Conditional_Order_Manager SHALL retrieve the current list of working orders via `GET /workingorders` and compare each order's epic and direction against active signals to determine required cancellations.

### Requirement 5: One Pending Order Per Epic

**User Story:** As a trader, I want at most one pending order per epic at any time, so that the bot does not accumulate conflicting or duplicate orders.

#### Acceptance Criteria

1. THE Conditional_Order_Manager SHALL maintain at most one active working order per epic at any time; if a new order placement is required for an epic that already has an active working order, the existing order SHALL be cancelled before the new order is placed.
2. WHEN a working order fills for an epic, THE Conditional_Order_Manager SHALL remove it from internal tracking and not place another working order for that epic until the Position_Manager reports no open position for that epic.
3. WHILE the Position_Manager holds an open position for an epic, THE Conditional_Order_Manager SHALL reject any new signal for that epic without placing a working order.
4. WHEN the Position_Manager reports that a position for an epic has been closed (via stop, take-profit, or trailing stop exit), THE Conditional_Order_Manager SHALL allow new working orders for that epic on the next signal.

### Requirement 6: Post-Fill Handoff to Trailing Stop

**User Story:** As a trader, I want filled conditional orders to be managed by the existing trailing stop logic, so that profit management works identically to market orders.

#### Acceptance Criteria

1. WHEN a working order is filled (detected via polling), THE Conditional_Order_Manager SHALL register the new position with the Position_Manager by calling `add_position` with the fill price, deal ID, direction, size, stop distance, take-profit distance, confidence, and patterns from the original signal.
2. WHEN `execution.use_trailing_stop` is true and a working order fills, THE Conditional_Order_Manager SHALL initialize the Trailing_Stop_Manager by calling `initialize` with the epic, deal ID, fill price, direction, stop distance, `execution.trailing_activation_pct`, and `execution.trailing_distance_pct` from configuration.
3. IF `execution.use_trailing_stop` is false and a working order fills, THEN THE Conditional_Order_Manager SHALL register the position with the Position_Manager only, relying on the order's attached stop-loss and take-profit levels for exit management.
4. WHEN a fill is detected, THE Conditional_Order_Manager SHALL extract the fill price from the `level` field and the deal ID from the `dealId` field of the IG API working order status response.

### Requirement 7: Configuration

**User Story:** As a trader, I want all conditional order parameters to be configurable in the YAML settings file, so that I can tune behaviour without code changes.

#### Acceptance Criteria

1. THE Conditional_Order_Manager SHALL read the following parameters from the `conditional_orders` section in `config/settings_ai.yaml`: `enabled` (boolean), `buffer_points` (numeric, range 0.5 to 50.0 points), `order_expiry_seconds` (integer, range 60 to 86400 seconds), `max_entry_distance_points` (numeric, range 1.0 to 200.0 points).
2. WHEN `conditional_orders.enabled` is false, THE Conditional_Order_Manager SHALL not be invoked and the bot SHALL place market orders using the existing logic.
3. IF any required parameter (`enabled`, `buffer_points`, `order_expiry_seconds`, `max_entry_distance_points`) is missing from the configuration file, THEN THE Conditional_Order_Manager SHALL log an error indicating which parameter is missing and disable conditional order functionality, falling back to market order execution.
4. THE Conditional_Order_Manager SHALL use the `buffer_points` value as the distance beyond the S/R level for entry calculation.
5. THE Conditional_Order_Manager SHALL use the `order_expiry_seconds` value to compute the `goodTillDate` timestamp by adding the configured seconds to the current time at order placement.
6. THE Conditional_Order_Manager SHALL use the `max_entry_distance_points` value to reject signals where the absolute distance between current price and calculated entry level exceeds this configured threshold.

### Requirement 8: Logging and Observability

**User Story:** As a trader, I want comprehensive logging of conditional order lifecycle events, so that I can review and debug order placement decisions.

#### Acceptance Criteria

1. WHEN a conditional order is placed, THE Conditional_Order_Manager SHALL log at INFO level the epic, direction, entry level, stop distance, expiry time, and buffer used.
2. WHEN a conditional order is filled, THE Conditional_Order_Manager SHALL log at INFO level the epic, fill price, deal ID, and time elapsed since placement in seconds.
3. WHEN a conditional order is cancelled (by expiry, signal reversal, or kill switch), THE Conditional_Order_Manager SHALL log at INFO level the epic, reason for cancellation (one of: "expired", "signal_reversal", "kill_switch", "daily_loss_limit"), and the unfilled entry level.
4. WHEN a conditional order is rejected due to max distance exceeded, THE Conditional_Order_Manager SHALL log at WARNING level the epic, calculated entry level, current price, the absolute distance in points, and the configured maximum distance.
5. WHEN the fallback to market order is used (no S/R level found), THE Conditional_Order_Manager SHALL log at INFO level the epic, direction, and reason for fallback (one of: "no_resistance_level", "no_support_level").
