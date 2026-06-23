# Implementation Plan

## Overview

Fix the main trading loop in `runners/run_ai_autonomous.py` so that `sync_positions_from_broker()` runs every loop BEFORE epic processing and `monitor_open_positions()` runs every loop when positions exist, while preserving TwelveData rate limits and equity check frequency.

## Tasks

- [x] 1. Write bug condition exploration test
  - **Property 1: Bug Condition** - Stale Position Persists After Broker Closure
  - **CRITICAL**: This test MUST FAIL on unfixed code - failure confirms the bug exists
  - **DO NOT attempt to fix the test or the code when it fails**
  - **NOTE**: This test encodes the expected behavior - it will validate the fix when it passes after implementation
  - **GOAL**: Surface counterexamples that demonstrate the bug exists
  - **Scoped PBT Approach**: Scope the property to concrete failing cases: broker closes a position (returns empty positions list), then verify the main loop clears it from `position_manager.positions` before epic processing on the NEXT iteration
  - Create test file `tests/test_position_sync_bug_condition.py`
  - Mock `ig.positions()` to return empty list (simulating broker closure of a position that exists in `position_manager.positions`)
  - Mock `data_aggregator.get_bars()`, `ig.account_summary()`, and sleep to avoid real API calls
  - Property: For any loop iteration where `isBugCondition(state)` holds (position closed at broker AND loop_count % 10 != 0 OR sync runs after epic processing), assert that `sync_positions_from_broker()` is called BEFORE the epic processing loop AND that stale positions are removed from `position_manager.positions` before the skip check evaluates
  - Use Hypothesis to generate random `loop_count` values (1-9, 11-19, etc.) and verify sync still runs
  - Run test on UNFIXED code
  - **EXPECTED OUTCOME**: Test FAILS because sync only runs every 10th loop and after epic processing
  - Document counterexamples: e.g., "At loop_count=3, position_manager.positions still contains 'CS.D.CFDGOLD.CFD.IP' despite broker returning empty positions list. sync_positions_from_broker() was not called."
  - Mark task complete when test is written, run, and failure is documented
  - _Requirements: 1.1, 1.2, 1.3, 1.4_

- [x] 2. Write preservation property tests (BEFORE implementing fix)
  - **Property 2: Preservation** - Rate Limits and Equity Check Frequency Unchanged
  - **IMPORTANT**: Follow observation-first methodology
  - Observe on UNFIXED code: `ig.account_summary()` is called exactly once every 10 loops
  - Observe on UNFIXED code: `time.sleep()` is called with `max(60, (num_symbols * 86400) / 720)` regardless of loop count
  - Observe on UNFIXED code: kill switch (`KILL_SWITCH=1`) breaks the loop immediately on the iteration it's detected
  - Observe on UNFIXED code: when a position IS genuinely open at broker, the epic is skipped with "⏭️ Skipping" log
  - Create test file `tests/test_position_sync_preservation.py`
  - Write property-based tests using Hypothesis:
    - **Property 2a**: For all sequences of N loop iterations (N drawn from integers 1..50), `ig.account_summary()` call count equals `floor(N / 10)` — equity check frequency preserved
    - **Property 2b**: For all `num_symbols` values (integers 1..20), `time.sleep()` is called with exactly `max(60, (num_symbols * 86400) / 720)` — poll interval unchanged
    - **Property 2c**: For any loop iteration where kill switch is set, the loop breaks immediately without processing epics
    - **Property 2d**: For any epic that IS in broker positions AND in `position_manager.positions`, the epic processing skips it (preservation of skip-if-genuinely-open logic)
  - Verify all tests PASS on UNFIXED code
  - **EXPECTED OUTCOME**: Tests PASS (confirms baseline behavior to preserve)
  - Mark task complete when tests are written, run, and passing on unfixed code
  - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5_

- [x] 3. Fix for stale position sync and infrequent trailing stop monitoring

  - [x] 3.1 Restructure main loop in `runners/run_ai_autonomous.py`
    - Move `sync_positions_from_broker(ig, position_manager, log)` out of the `if loop_count % 10 == 0` block
    - Place it immediately after the kill switch check, BEFORE the epic processing `for epic in epics:` loop
    - It must run every loop iteration unconditionally
    - _Bug_Condition: isBugCondition(state) where state.loop_count % 10 != 0 OR state.sync_runs_after_epic_processing_
    - _Expected_Behavior: sync_positions_from_broker() runs every iteration before epic processing, clearing stale positions within 1 loop_
    - _Preservation: TwelveData rate limits unaffected (sync uses IG API, not TwelveData)_
    - _Requirements: 2.2, 2.3, 2.4_

  - [x] 3.2 Move `monitor_open_positions()` to run every loop
    - Remove the `if loop_count % 5 == 0` gate around `monitor_open_positions()`
    - Place the call after `sync_positions_from_broker()` and before the equity check
    - The function already guards with `if not position_manager.positions: return` so it's a no-op when no positions exist
    - _Bug_Condition: monitor_open_positions only runs every 5th loop (~10 min), trailing stops never activate in time_
    - _Expected_Behavior: monitor_open_positions() runs every loop when positions exist, trailing stops update within one poll interval_
    - _Preservation: No additional TwelveData calls — get_bars() uses cached data within same bar period; no-op when no positions_
    - _Requirements: 2.1, 3.1_

  - [x] 3.3 Keep equity/P&L check at reduced frequency
    - Ensure `ig.account_summary()` and daily P&L calculation remain inside `if loop_count % 10 == 0`
    - Remove `sync_positions_from_broker()` from this block (moved to every-loop)
    - _Preservation: Equity check frequency preserved at every 10th loop per Requirements 3.2_
    - _Requirements: 3.2_

  - [x] 3.4 Verify new loop order is correct
    - Final loop body order must be: kill switch → sync_positions_from_broker → monitor_open_positions → equity/P&L (every 10th) → periodic reporting → daily lockout check → epic processing → sleep
    - Verify no logic dependencies are broken by the reorder
    - _Requirements: 2.4_

  - [x] 3.5 Verify bug condition exploration test now passes
    - **Property 1: Expected Behavior** - Stale Position Cleared Before Epic Processing
    - **IMPORTANT**: Re-run the SAME test from task 1 - do NOT write a new test
    - The test from task 1 encodes the expected behavior (sync runs every loop before epics)
    - When this test passes, it confirms the expected behavior is satisfied
    - Run bug condition exploration test from step 1
    - **EXPECTED OUTCOME**: Test PASSES (confirms bug is fixed — stale positions cleared within 1 iteration)
    - _Requirements: 2.2, 2.3, 2.4_

  - [x] 3.6 Verify preservation tests still pass
    - **Property 2: Preservation** - Rate Limits and Equity Check Frequency Unchanged
    - **IMPORTANT**: Re-run the SAME tests from task 2 - do NOT write new tests
    - Run preservation property tests from step 2
    - **EXPECTED OUTCOME**: Tests PASS (confirms no regressions — equity check still every 10th loop, poll interval unchanged, kill switch still immediate, skip-if-open still works)
    - Confirm all property tests still pass after fix (no regressions)

- [x] 4. Checkpoint - Ensure all tests pass
  - Run full test suite: `python -m pytest tests/test_position_sync_bug_condition.py tests/test_position_sync_preservation.py -v`
  - Verify Property 1 (bug condition) passes on fixed code
  - Verify Property 2 (preservation) passes on fixed code
  - Verify no other existing tests are broken by the loop restructuring
  - Ensure all tests pass, ask the user if questions arise.

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1", "2"] },
    { "id": 1, "tasks": ["3.1", "3.2", "3.3"] },
    { "id": 2, "tasks": ["3.4"] },
    { "id": 3, "tasks": ["3.5", "3.6"] },
    { "id": 4, "tasks": ["4"] }
  ]
}
```

## Notes

- The IG positions API is separate from TwelveData and can handle one call per ~120s loop without rate limit concerns
- `monitor_open_positions()` uses `aggregator.get_bars()` which leverages cached data within the same bar period, so calling it every loop does NOT add TwelveData API calls
- The `poll_interval` formula `max(60, (num_symbols * 86400) / 720)` remains the sole mechanism for TwelveData rate limiting
- Property-based tests use Hypothesis library (already present in the project based on `.hypothesis/` directory)
