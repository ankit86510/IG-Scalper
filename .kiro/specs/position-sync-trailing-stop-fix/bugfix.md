# Bugfix Requirements Document

## Introduction

The main trading loop in `runners/run_ai_autonomous.py` has three related bugs caused by infrequent execution of position sync and trailing stop monitoring. With a poll interval of ~120 seconds per loop iteration, `monitor_open_positions()` only runs every ~10 minutes (every 5th loop) and `sync_positions_from_broker()` only runs every ~20 minutes (every 10th loop). This causes trailing stops to never activate in time, broker-closed positions to remain in local state for extended periods, and the bot to skip new order placement on epics that are no longer open at the broker.

## Bug Analysis

### Current Behavior (Defect)

1.1 WHEN trailing stops are enabled and a position is open THEN the system only evaluates trailing stop updates every 5th loop iteration (~10 minutes), causing trailing stops to never activate or trail in a timely manner

1.2 WHEN IG auto-closes a position because SL/TP was hit at broker level THEN the system does not detect the closure for up to 10 loop iterations (~20 minutes) because `sync_positions_from_broker()` only runs every 10th loop

1.3 WHEN a position has been closed at the broker but `sync_positions_from_broker()` has not yet run THEN the system logs "⏭️ Skipping {epic} - position already open" and refuses to place new orders on that epic for up to 9 consecutive iterations (~18 minutes)

1.4 WHEN `sync_positions_from_broker()` executes THEN it runs AFTER the epic processing loop, meaning positions closed at broker are not removed before the skip check (`if epic in position_manager.positions`) evaluates

### Expected Behavior (Correct)

2.1 WHEN trailing stops are enabled and a position is open THEN the system SHALL evaluate trailing stop updates every loop iteration so that stops are trailed within one poll interval of price movement

2.2 WHEN IG auto-closes a position because SL/TP was hit at broker level THEN the system SHALL detect the closure within 1-2 loop iterations by running position sync every loop (or at minimum every 2nd loop)

2.3 WHEN a position has been closed at the broker THEN the system SHALL remove it from `position_manager.positions` before evaluating the epic skip check, freeing the epic for new signals within 1-2 loops

2.4 WHEN the main loop starts a new iteration THEN the system SHALL run `sync_positions_from_broker()` BEFORE the epic processing loop so that stale positions are cleared before order placement decisions are made

### Unchanged Behavior (Regression Prevention)

3.1 WHEN no positions are open THEN the system SHALL CONTINUE TO skip `monitor_open_positions()` without making unnecessary API calls

3.2 WHEN the daily P&L check and equity sync runs (account_summary) THEN the system SHALL CONTINUE TO execute at a reduced frequency (every 10th loop) to conserve API rate limits

3.3 WHEN a position is genuinely open at the broker THEN the system SHALL CONTINUE TO skip that epic and log "⏭️ Skipping {epic} - position already open"

3.4 WHEN the kill switch environment variable is set THEN the system SHALL CONTINUE TO break out of the main loop immediately

3.5 WHEN the poll interval is calculated based on active symbol count THEN the system SHALL CONTINUE TO respect TwelveData rate limits (800/day, 8/min)

3.6 WHEN `sync_positions_from_broker()` finds a position no longer at the broker THEN the system SHALL CONTINUE TO call `position_manager.remove_position(epic, reason="BROKER_CLOSED")` to log the closure to trade history
