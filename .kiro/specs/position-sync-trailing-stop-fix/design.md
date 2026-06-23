# Position Sync & Trailing Stop Frequency Bugfix Design

## Overview

The main trading loop in `runners/run_ai_autonomous.py` executes position sync and trailing stop monitoring too infrequently, causing up to ~18 minutes of stale state after a broker closes a position. The fix restructures the loop so that position sync runs every iteration (before epic processing) and trailing stop monitoring runs every iteration when positions exist, while keeping the equity/P&L check at reduced frequency to respect TwelveData rate limits.

## Glossary

- **Bug_Condition (C)**: The condition where `sync_positions_from_broker()` runs only every 10th loop and AFTER epic processing, combined with `monitor_open_positions()` running only every 5th loop
- **Property (P)**: Position sync runs every loop BEFORE epic processing; trailing stop monitoring runs every loop when positions exist
- **Preservation**: TwelveData rate limits (800/day, 8/min) continue to be respected; equity check stays at every-10th-loop frequency; kill switch, daily lockout, and skip-if-open logic remain unchanged
- **`sync_positions_from_broker()`**: Function that calls `ig.positions()` and removes locally-tracked positions that no longer exist at the broker
- **`monitor_open_positions()`**: Function that fetches current price via `aggregator.get_bars()` and updates trailing stops / checks TP for each open position
- **`poll_interval`**: Sleep duration between loop iterations, calculated as `max(60, (num_symbols * 86400) / 720)` to stay within TwelveData limits

## Bug Details

### Bug Condition

The bug manifests when the broker closes a position (SL/TP hit) but the bot continues to treat it as open for up to 9 more loop iterations (~18 minutes). The combination of infrequent sync AND sync running after epic processing means the bot skips valid trading opportunities on that epic.

**Formal Specification:**
```
FUNCTION isBugCondition(state)
  INPUT: state of type LoopIterationState
  OUTPUT: boolean

  RETURN (state.loop_count % 10 != 0 OR state.sync_runs_after_epic_processing)
         AND state.position_closed_at_broker
         AND epic IN position_manager.positions
END FUNCTION
```

### Examples

- Loop 1: Broker closes XAU/USD position via SL. Bot has `position_manager.positions["CS.D.CFDGOLD.CFD.IP"]` still present. Next sync is at loop 10 (~18 min later). Bot logs "⏭️ Skipping CS.D.CFDGOLD.CFD.IP - position already open" for 9 iterations.
- Loop 3: Trailing stop should activate (price moved 30% of stop distance in profit) but `monitor_open_positions()` won't run until loop 5. By loop 5, price may have reversed.
- Loop 10: `sync_positions_from_broker()` finally runs but AFTER the epic loop. The stale position was already skipped in the epic loop this iteration. Actual clearance happens at loop 11's epic processing — a full 11 iterations (~22 min) of dead time.
- Loop 7: Position is genuinely open at broker. `monitor_open_positions()` skipped (not a multiple of 5). Trailing stop doesn't trail despite favorable price movement.

## Expected Behavior

### Preservation Requirements

**Unchanged Behaviors:**
- TwelveData rate limits (800/day, 8/min) must still be respected — the poll_interval calculation and data aggregator's internal rate limiter must not be affected
- Equity/P&L check via `ig.account_summary()` stays at every-10th-loop frequency to conserve API budget
- Kill switch (`KILL_SWITCH` env var) continues to break the loop immediately
- Daily lockout continues to pause trading when daily loss exceeds threshold
- The "⏭️ Skipping {epic} - position already open" logic remains correct for positions genuinely open at the broker
- `position_manager.remove_position(epic, reason="BROKER_CLOSED")` continues to log closures to trade history
- Periodic reporting (every 300s) continues unchanged
- The `last_bar_time` deduplication logic for epic processing remains unchanged

**Scope:**
All inputs that do NOT involve the scheduling of `sync_positions_from_broker()` or `monitor_open_positions()` should be completely unaffected by this fix. This includes:
- Order placement logic
- AI signal generation and confidence thresholds
- Position sizing and market rules enforcement
- Trailing stop initialization on new positions
- Sleep/poll interval calculation

## Hypothesized Root Cause

Based on the bug description, the most likely issues are:

1. **Incorrect Loop Frequency for Position Sync**: `sync_positions_from_broker()` is gated by `if loop_count % 10 == 0`, meaning it only runs every ~20 minutes (with ~120s poll interval). This was likely set conservatively to avoid API rate limits, but IG's positions endpoint is separate from TwelveData and has generous limits.

2. **Incorrect Execution Order**: `sync_positions_from_broker()` runs inside the `loop_count % 10` block AFTER the epic processing `for epic in epics:` loop. Even when sync does run, stale positions have already been skipped in that iteration.

3. **Incorrect Loop Frequency for Monitoring**: `monitor_open_positions()` is gated by `if loop_count % 5 == 0`, meaning trailing stops only update every ~10 minutes. For a scalping bot with ATR-based stops, this is far too infrequent.

4. **Conflation of API Rate Concerns**: The code appears to conflate TwelveData rate limits (which govern price data fetching) with IG API rate limits (which govern position/account queries). The IG positions API is much more generous and can be called every loop without issue.

## Correctness Properties

Property 1: Bug Condition - Position Sync Runs Before Epic Processing Every Loop

_For any_ loop iteration where a position has been closed at the broker (isBugCondition returns true), the fixed main loop SHALL call `sync_positions_from_broker()` before the epic processing loop, removing stale positions from `position_manager.positions` so they are not skipped.

**Validates: Requirements 2.2, 2.3, 2.4**

Property 2: Preservation - TwelveData Rate Limits Unchanged

_For any_ loop iteration where no positions are closed at the broker (isBugCondition returns false), the fixed code SHALL produce the same TwelveData API call pattern as the original code, preserving the poll_interval calculation and per-minute/per-day rate limit compliance.

**Validates: Requirements 3.2, 3.5**

## Fix Implementation

### Changes Required

Assuming our root cause analysis is correct:

**File**: `runners/run_ai_autonomous.py`

**Function**: `main()` — the main `while True` trading loop

**Specific Changes**:

1. **Move `sync_positions_from_broker()` to run every loop, before epic processing**: Remove it from the `if loop_count % 10 == 0` block and place it immediately after the kill switch check, before the epic loop. The IG positions API is separate from TwelveData and can handle one call per ~120s loop without issue.

2. **Move `monitor_open_positions()` to run every loop (when positions exist)**: Remove the `if loop_count % 5 == 0` gate. The function already has a guard `if not position_manager.positions: return` so it's a no-op when no positions are open. The `aggregator.get_bars()` call inside it uses cached data from the same bar period, so it doesn't add TwelveData API calls.

3. **Keep equity check at reduced frequency**: The `ig.account_summary()` call and daily P&L calculation remain inside `if loop_count % 10 == 0` since this is the only part that needs rate-limiting (and is separate from position sync).

4. **Restructure loop order**: The new order within the loop body becomes:
   - Kill switch check
   - `sync_positions_from_broker()` (every loop)
   - `monitor_open_positions()` (every loop, no-op if no positions)
   - Equity/P&L update (every 10th loop)
   - Periodic reporting (every 300s)
   - Daily lockout check
   - Epic processing loop
   - Sleep

5. **No change to poll_interval**: The sleep duration remains `max(60, (num_symbols * 86400) / 720)` — this is what governs TwelveData rate limits, not the frequency of IG API calls.

## Testing Strategy

### Validation Approach

The testing strategy follows a two-phase approach: first, surface counterexamples that demonstrate the bug on unfixed code, then verify the fix works correctly and preserves existing behavior.

### Exploratory Bug Condition Checking

**Goal**: Surface counterexamples that demonstrate the bug BEFORE implementing the fix. Confirm or refute the root cause analysis. If we refute, we will need to re-hypothesize.

**Test Plan**: Write tests that mock `ig.positions()` to return an empty positions list (simulating broker closure) and verify that `position_manager.positions` retains stale entries across multiple loop iterations. Run on UNFIXED code to observe the staleness window.

**Test Cases**:
1. **Stale Position Persists**: Mock broker returning no positions, verify `position_manager.positions` still contains the epic after 9 loop iterations (will fail on unfixed code — position stays stale)
2. **Sync Runs After Epic Processing**: Instrument loop to verify `sync_positions_from_broker()` is called after the `for epic in epics:` block (will confirm bug on unfixed code)
3. **Monitor Skipped on Non-5th Loops**: Verify `monitor_open_positions()` is not called on loop iterations 1, 2, 3, 4 (will confirm infrequent monitoring on unfixed code)
4. **Trailing Stop Not Updated for 10 Minutes**: With mock price data showing favorable movement, verify trailing stop doesn't activate between monitoring intervals (will fail on unfixed code)

**Expected Counterexamples**:
- Position remains in `position_manager.positions` for 9+ iterations after broker closure
- Possible causes: `loop_count % 10` gate, sync running after epic processing

### Fix Checking

**Goal**: Verify that for all inputs where the bug condition holds, the fixed function produces the expected behavior.

**Pseudocode:**
```
FOR ALL state WHERE isBugCondition(state) DO
  result := run_fixed_loop_iteration(state)
  ASSERT position_removed_before_epic_processing(result)
  ASSERT trailing_stop_updated_this_iteration(result)
END FOR
```

### Preservation Checking

**Goal**: Verify that for all inputs where the bug condition does NOT hold, the fixed function produces the same result as the original function.

**Pseudocode:**
```
FOR ALL state WHERE NOT isBugCondition(state) DO
  ASSERT twelvedata_api_calls(fixed_loop) == twelvedata_api_calls(original_loop)
  ASSERT poll_interval(fixed_loop) == poll_interval(original_loop)
  ASSERT equity_check_frequency(fixed_loop) == equity_check_frequency(original_loop)
END FOR
```

**Testing Approach**: Property-based testing is recommended for preservation checking because:
- It generates many loop iteration states (varying loop_count, position states, market data)
- It catches edge cases where the refactored code might accidentally change behavior
- It provides strong guarantees that TwelveData rate limits are never violated

**Test Plan**: Observe behavior on UNFIXED code first for non-position-sync operations (equity checks, poll interval, epic processing), then write property-based tests capturing that behavior.

**Test Cases**:
1. **Equity Check Frequency Preserved**: Verify `ig.account_summary()` is still called only every 10th loop after the fix
2. **Poll Interval Unchanged**: Verify `time.sleep()` is called with the same value regardless of fix
3. **Skip Logic Correct for Open Positions**: Verify that when a position IS genuinely open at broker, the epic is still skipped
4. **Kill Switch Still Immediate**: Verify kill switch breaks loop on the same iteration it's detected

### Unit Tests

- Test that `sync_positions_from_broker()` removes stale positions correctly
- Test that `monitor_open_positions()` is a no-op when `position_manager.positions` is empty
- Test loop iteration ordering: sync → monitor → equity → epics
- Test that trailing stop updates propagate to broker on every monitored iteration

### Property-Based Tests

- Generate random sequences of loop iterations with varying `loop_count` values and verify sync always runs before epic processing
- Generate random position states (open/closed at broker) and verify stale positions are cleared within 1 iteration
- Generate random configurations of `num_symbols` and verify `poll_interval` formula is unchanged
- Generate random sequences and verify `account_summary()` call count equals `floor(total_loops / 10)`

### Integration Tests

- Test full loop with mocked IG client: broker closes position, verify next iteration clears it and allows new signal
- Test trailing stop activation timing: with mock price feed, verify stop trails within one loop of price improvement
- Test that a 10-symbol configuration still respects TwelveData 800/day limit over simulated 24h of loop iterations
