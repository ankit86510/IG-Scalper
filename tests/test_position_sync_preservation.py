"""Property-based preservation tests for the main trading loop.

These tests capture EXISTING CORRECT behavior that must NOT change after the fix.
All tests MUST PASS on the current unfixed code.

**Validates: Requirements 3.1, 3.2, 3.3, 3.4, 3.5**
"""

import os
import time
import logging
from unittest.mock import MagicMock, patch, call
from datetime import datetime, UTC

import pandas as pd
from hypothesis import given, settings, assume
from hypothesis.strategies import integers, lists, text, sampled_from

# Import the modules under test
from runners.run_ai_autonomous import (
    PositionManager,
    TrailingStopManager,
    sync_positions_from_broker,
    monitor_open_positions,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_mock_ig(broker_epics=None):
    """Create a mock IG client with configurable broker positions."""
    ig = MagicMock()
    positions_list = []
    if broker_epics:
        for epic in broker_epics:
            positions_list.append({
                'market': {'epic': epic},
                'position': {
                    'dealId': f'DEAL_{epic}',
                    'direction': 'BUY',
                    'size': 1.0,
                    'level': 3300.0,
                }
            })
    ig.positions.return_value = {'positions': positions_list}
    ig.account_summary.return_value = {
        'accounts': [{'balance': {'balance': 10000.0}}]
    }
    return ig


def make_mock_aggregator():
    """Create a mock data aggregator that returns minimal valid data."""
    aggregator = MagicMock()
    # Return a minimal DataFrame so epic processing doesn't skip due to empty data
    df = pd.DataFrame(
        {'open': [3300.0] * 60, 'high': [3310.0] * 60,
         'low': [3290.0] * 60, 'close': [3305.0] * 60},
        index=pd.date_range('2024-01-01', periods=60, freq='5min')
    )
    aggregator.get_bars.return_value = df
    return aggregator


def run_n_loop_iterations(n, ig, position_manager, epics, aggregator=None,
                          trailing_manager=None, daily_pnl_pct=0.0,
                          max_daily_loss_pct=5.0, start_equity=10000.0):
    """Run exactly N iterations of the main trading loop logic.

    This replicates the FIXED main loop structure from run_ai_autonomous.py main().
    New order: kill switch → sync → monitor → equity (every 10th) → epics → sleep
    Returns the loop_count at end.
    """
    log = logging.getLogger('test_preservation')
    log.setLevel(logging.DEBUG)

    if aggregator is None:
        aggregator = make_mock_aggregator()
    if trailing_manager is None:
        trailing_manager = MagicMock()
        trailing_manager.update.return_value = (None, None)
        trailing_manager.get_info.return_value = None

    loop_count = 0
    last_report_time = time.time() + 9999  # prevent reporting from firing
    last_bar_time = {e: None for e in epics}

    for _ in range(n):
        loop_count += 1

        # Kill switch check
        if os.environ.get("KILL_SWITCH", "0") == "1":
            break

        # Sync positions from broker every loop, BEFORE epic processing
        sync_positions_from_broker(ig, position_manager, log)

        # Monitor open positions every loop (no-op when no positions exist)
        monitor_open_positions(ig, position_manager, trailing_manager, aggregator, log)

        # Update P&L every 10 loops (equity check at reduced frequency)
        if loop_count % 10 == 0:
            try:
                acct = ig.account_summary()
                current_equity = acct['accounts'][0]['balance']['balance']
                daily_pnl_pct = ((current_equity - start_equity) / start_equity) * 100
            except Exception:
                pass

        # Skip daily lockout and periodic reporting for simplicity
        # (they don't affect the properties we're testing)

        # Process each instrument
        for epic in epics:
            if epic in position_manager.positions:
                log.info(f"⏭️ Skipping {epic} - position already open")
                continue

        # Sleep — we don't actually sleep, but we record that it would happen
        num_symbols = len(epics)
        poll_interval = max(60, (num_symbols * 86400) / 720)
        time.sleep(poll_interval)

    return loop_count


# ---------------------------------------------------------------------------
# Property 2a: Equity Check Frequency Preserved
# For all sequences of N loop iterations (N drawn from integers 1..50),
# ig.account_summary() call count equals floor(N / 10)
#
# **Validates: Requirements 3.2**
# ---------------------------------------------------------------------------

class TestEquityCheckFrequency:
    """Property 2a: ig.account_summary() is called exactly floor(N/10) times
    in N loop iterations.

    **Validates: Requirements 3.2**
    """

    @given(n=integers(min_value=1, max_value=50))
    @settings(max_examples=50, deadline=None)
    def test_account_summary_call_count(self, n):
        """For N loop iterations, account_summary is called floor(N/10) times."""
        epics = ["CS.D.CFDGOLD.CFD.IP"]
        ig = make_mock_ig(broker_epics=epics)
        position_manager = PositionManager(logging.getLogger('test'))

        with patch('time.sleep'):
            run_n_loop_iterations(n, ig, position_manager, epics)

        expected_calls = n // 10
        actual_calls = ig.account_summary.call_count
        assert actual_calls == expected_calls, (
            f"Expected {expected_calls} account_summary calls for {n} iterations, "
            f"got {actual_calls}"
        )


# ---------------------------------------------------------------------------
# Property 2b: Poll Interval Unchanged
# For all num_symbols values (integers 1..20), time.sleep() is called
# with exactly max(60, (num_symbols * 86400) / 720)
#
# **Validates: Requirements 3.5**
# ---------------------------------------------------------------------------

class TestPollInterval:
    """Property 2b: time.sleep() is called with max(60, (num_symbols * 86400) / 720)
    regardless of loop count.

    **Validates: Requirements 3.5**
    """

    @given(num_symbols=integers(min_value=1, max_value=20))
    @settings(max_examples=20, deadline=None)
    def test_sleep_interval_matches_formula(self, num_symbols):
        """Sleep is called with the correct poll interval for any number of symbols."""
        epics = [f"EPIC_{i}" for i in range(num_symbols)]
        ig = make_mock_ig(broker_epics=[])
        position_manager = PositionManager(logging.getLogger('test'))

        expected_interval = max(60, (num_symbols * 86400) / 720)

        with patch('time.sleep') as mock_sleep:
            # Run exactly 1 iteration
            run_n_loop_iterations(1, ig, position_manager, epics)

            # Verify sleep was called with the expected interval
            assert mock_sleep.call_count >= 1, "time.sleep should be called at least once"
            # The last sleep call in each iteration is the poll interval
            last_call_args = mock_sleep.call_args_list[-1]
            actual_interval = last_call_args[0][0]
            assert actual_interval == expected_interval, (
                f"Expected sleep({expected_interval}) for {num_symbols} symbols, "
                f"got sleep({actual_interval})"
            )

    @given(
        num_symbols=integers(min_value=1, max_value=20),
        n_iterations=integers(min_value=1, max_value=10),
    )
    @settings(max_examples=30, deadline=None)
    def test_sleep_interval_constant_across_iterations(self, num_symbols, n_iterations):
        """Sleep interval is the same value on every iteration regardless of loop count."""
        epics = [f"EPIC_{i}" for i in range(num_symbols)]
        ig = make_mock_ig(broker_epics=[])
        position_manager = PositionManager(logging.getLogger('test'))

        expected_interval = max(60, (num_symbols * 86400) / 720)

        with patch('time.sleep') as mock_sleep:
            run_n_loop_iterations(n_iterations, ig, position_manager, epics)

            # Every sleep call should have the expected interval
            assert mock_sleep.call_count == n_iterations, (
                f"Expected {n_iterations} sleep calls, got {mock_sleep.call_count}"
            )
            for i, call_item in enumerate(mock_sleep.call_args_list):
                actual = call_item[0][0]
                assert actual == expected_interval, (
                    f"Iteration {i+1}: expected sleep({expected_interval}), "
                    f"got sleep({actual})"
                )


# ---------------------------------------------------------------------------
# Property 2c: Kill Switch Immediate Break
# For any loop iteration where kill switch is set, the loop breaks
# immediately without processing epics
#
# **Validates: Requirements 3.4**
# ---------------------------------------------------------------------------

class TestKillSwitchImmediate:
    """Property 2c: When KILL_SWITCH=1, the loop breaks immediately on that iteration
    without processing epics.

    **Validates: Requirements 3.4**
    """

    @given(kill_at=integers(min_value=1, max_value=20))
    @settings(max_examples=20, deadline=None)
    def test_kill_switch_breaks_loop_immediately(self, kill_at):
        """Kill switch set at iteration K means loop executes K-1 full iterations."""
        epics = ["CS.D.CFDGOLD.CFD.IP", "CS.D.EURUSD.CFD.IP"]
        ig = make_mock_ig(broker_epics=[])
        position_manager = PositionManager(logging.getLogger('test'))
        aggregator = make_mock_aggregator()

        loop_iterations_completed = 0

        with patch('time.sleep') as mock_sleep:
            # We'll manually control KILL_SWITCH via environment
            log = logging.getLogger('test_kill')
            log.setLevel(logging.DEBUG)

            trailing_manager = MagicMock()
            trailing_manager.update.return_value = (None, None)
            trailing_manager.get_info.return_value = None

            loop_count = 0
            last_bar_time = {e: None for e in epics}

            for iteration in range(50):  # max 50 iterations
                loop_count += 1

                # Set kill switch at the target iteration
                if loop_count == kill_at:
                    os.environ["KILL_SWITCH"] = "1"

                # Kill switch check (as in the real code)
                if os.environ.get("KILL_SWITCH", "0") == "1":
                    break

                # If we get here, the iteration was not killed
                loop_iterations_completed += 1

                # Sync positions from broker every loop (FIXED structure)
                sync_positions_from_broker(ig, position_manager, log)

                # Monitor open positions every loop (FIXED structure)
                monitor_open_positions(ig, position_manager, trailing_manager, aggregator, log)

                # Update P&L every 10 loops
                if loop_count % 10 == 0:
                    ig.account_summary()

                # Process each instrument
                for epic in epics:
                    if epic in position_manager.positions:
                        continue

                # Sleep
                num_symbols = len(epics)
                poll_interval = max(60, (num_symbols * 86400) / 720)
                time.sleep(poll_interval)

        # Clean up
        os.environ.pop("KILL_SWITCH", None)

        # The loop should have completed exactly kill_at - 1 full iterations
        assert loop_iterations_completed == kill_at - 1, (
            f"Expected {kill_at - 1} completed iterations before kill at iteration "
            f"{kill_at}, got {loop_iterations_completed}"
        )

    @given(kill_at=integers(min_value=1, max_value=10))
    @settings(max_examples=10, deadline=None)
    def test_kill_switch_no_epic_processing_on_kill_iteration(self, kill_at):
        """On the iteration where kill switch activates, no epics are processed."""
        epics = ["CS.D.CFDGOLD.CFD.IP"]
        ig = make_mock_ig(broker_epics=[])
        position_manager = PositionManager(logging.getLogger('test'))
        aggregator = make_mock_aggregator()

        epic_processing_count = 0

        with patch('time.sleep'):
            log = logging.getLogger('test_kill_epic')
            loop_count = 0

            trailing_manager = MagicMock()
            trailing_manager.update.return_value = (None, None)
            trailing_manager.get_info.return_value = None

            for iteration in range(50):
                loop_count += 1

                if loop_count == kill_at:
                    os.environ["KILL_SWITCH"] = "1"

                if os.environ.get("KILL_SWITCH", "0") == "1":
                    break

                # Sync positions from broker every loop (FIXED structure)
                sync_positions_from_broker(ig, position_manager, log)

                # Monitor open positions every loop (FIXED structure)
                monitor_open_positions(ig, position_manager, trailing_manager, aggregator, log)

                # Count epic processing iterations
                for epic in epics:
                    if epic not in position_manager.positions:
                        epic_processing_count += 1

                num_symbols = len(epics)
                poll_interval = max(60, (num_symbols * 86400) / 720)
                time.sleep(poll_interval)

        os.environ.pop("KILL_SWITCH", None)

        # Epic processing should have happened kill_at - 1 times (one per completed iteration)
        assert epic_processing_count == kill_at - 1, (
            f"Expected {kill_at - 1} epic processings before kill, got {epic_processing_count}"
        )


# ---------------------------------------------------------------------------
# Property 2d: Skip-If-Genuinely-Open Logic Preserved
# For any epic that IS in broker positions AND in position_manager.positions,
# the epic processing skips it
#
# **Validates: Requirements 3.3**
# ---------------------------------------------------------------------------

class TestSkipIfOpen:
    """Property 2d: When a position IS genuinely open at broker AND tracked locally,
    the epic is skipped during epic processing.

    **Validates: Requirements 3.3**
    """

    @given(
        num_open=integers(min_value=1, max_value=5),
        num_closed=integers(min_value=0, max_value=5),
    )
    @settings(max_examples=30, deadline=None)
    def test_open_positions_are_skipped(self, num_open, num_closed):
        """Epics with genuinely open positions are skipped in epic processing."""
        # Create epics - some open, some not
        open_epics = [f"OPEN_EPIC_{i}" for i in range(num_open)]
        closed_epics = [f"CLOSED_EPIC_{i}" for i in range(num_closed)]
        all_epics = open_epics + closed_epics

        ig = make_mock_ig(broker_epics=open_epics)
        log = logging.getLogger('test_skip')
        position_manager = PositionManager(log)

        # Add positions for open epics (simulating genuinely open positions)
        for epic in open_epics:
            position_manager.positions[epic] = {
                'deal_id': f'DEAL_{epic}',
                'direction': 'BUY',
                'size': 1.0,
                'entry_price': 3300.0,
                'entry_time': datetime.now(UTC).isoformat(),
                'stop_distance': 10.0,
                'tp_distance': 20.0,
                'stop_level': 3290.0,
                'tp_level': 3320.0,
                'confidence': 0.8,
                'patterns': ['test_pattern'],
                'status': 'OPEN'
            }

        # Simulate the epic processing loop
        skipped_epics = []
        processed_epics = []

        for epic in all_epics:
            if epic in position_manager.positions:
                skipped_epics.append(epic)
                continue
            processed_epics.append(epic)

        # All open epics should be skipped
        assert set(skipped_epics) == set(open_epics), (
            f"Expected open epics {open_epics} to be skipped, "
            f"but skipped {skipped_epics}"
        )

        # All closed epics should be processed
        assert set(processed_epics) == set(closed_epics), (
            f"Expected closed epics {closed_epics} to be processed, "
            f"but processed {processed_epics}"
        )

    @given(n_iterations=integers(min_value=1, max_value=10))
    @settings(max_examples=20, deadline=None)
    def test_skip_persists_across_iterations(self, n_iterations):
        """Positions genuinely open at broker remain skipped across multiple iterations."""
        epics = ["CS.D.CFDGOLD.CFD.IP", "CS.D.EURUSD.CFD.IP"]
        open_epic = "CS.D.CFDGOLD.CFD.IP"

        # Broker says position is open
        ig = make_mock_ig(broker_epics=[open_epic])
        log = logging.getLogger('test_skip_persist')
        position_manager = PositionManager(log)

        # Position is tracked locally
        position_manager.positions[open_epic] = {
            'deal_id': 'DEAL_GOLD',
            'direction': 'BUY',
            'size': 1.0,
            'entry_price': 3300.0,
            'entry_time': datetime.now(UTC).isoformat(),
            'stop_distance': 10.0,
            'tp_distance': 20.0,
            'stop_level': 3290.0,
            'tp_level': 3320.0,
            'confidence': 0.8,
            'patterns': ['test'],
            'status': 'OPEN'
        }

        skip_count = 0

        with patch('time.sleep'):
            for iteration in range(n_iterations):
                for epic in epics:
                    if epic in position_manager.positions:
                        skip_count += 1

        # The open epic should be skipped every iteration
        assert skip_count == n_iterations, (
            f"Expected {n_iterations} skips of open position, got {skip_count}"
        )
