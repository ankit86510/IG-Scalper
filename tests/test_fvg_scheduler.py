"""
Unit tests for CycleScheduler.

Tests cover:
- Interval timing logic
- KILL_SWITCH environment variable check
- Daily lockout integration
- Overlapping cycle prevention
- Startup logging with Europe/Rome timezone
"""

import os
import time
from unittest.mock import patch

import pytest

from strategy.fvg_scheduler import CycleScheduler


class TestCycleSchedulerShouldRun:
    """Tests for CycleScheduler.should_run() method."""

    def test_should_run_on_first_call(self):
        """First call should always run (no previous cycle time)."""
        scheduler = CycleScheduler(interval_seconds=300)
        assert scheduler.should_run() is True

    def test_should_not_run_before_interval_elapses(self):
        """Should not run if interval hasn't elapsed since last cycle."""
        scheduler = CycleScheduler(interval_seconds=300)
        scheduler.mark_cycle_start()
        scheduler.mark_cycle_complete()
        # Immediately after a cycle, interval hasn't elapsed
        assert scheduler.should_run() is False

    def test_should_run_after_interval_elapses(self):
        """Should run after interval has elapsed."""
        scheduler = CycleScheduler(interval_seconds=1)
        scheduler.mark_cycle_start()
        scheduler.mark_cycle_complete()
        # Simulate time passing
        scheduler._last_cycle_time = time.time() - 2
        assert scheduler.should_run() is True

    def test_kill_switch_active_blocks_run(self):
        """KILL_SWITCH=1 should prevent cycle from running."""
        scheduler = CycleScheduler(interval_seconds=300)
        with patch.dict(os.environ, {"KILL_SWITCH": "1"}):
            assert scheduler.should_run() is False

    def test_kill_switch_inactive_allows_run(self):
        """KILL_SWITCH not set or != '1' should allow cycle."""
        scheduler = CycleScheduler(interval_seconds=300)
        with patch.dict(os.environ, {"KILL_SWITCH": "0"}):
            assert scheduler.should_run() is True

    def test_kill_switch_not_set_allows_run(self):
        """No KILL_SWITCH env var should allow cycle."""
        scheduler = CycleScheduler(interval_seconds=300)
        with patch.dict(os.environ, {}, clear=True):
            # Remove KILL_SWITCH if present
            os.environ.pop("KILL_SWITCH", None)
            assert scheduler.should_run() is True

    def test_daily_lockout_active_blocks_run(self):
        """Active daily lockout should prevent cycle from running."""
        lockout_fn = lambda: True
        scheduler = CycleScheduler(interval_seconds=300, lockout_checker=lockout_fn)
        assert scheduler.should_run() is False

    def test_daily_lockout_inactive_allows_run(self):
        """Inactive daily lockout should allow cycle."""
        lockout_fn = lambda: False
        scheduler = CycleScheduler(interval_seconds=300, lockout_checker=lockout_fn)
        assert scheduler.should_run() is True

    def test_no_lockout_checker_allows_run(self):
        """No lockout checker provided should allow cycle."""
        scheduler = CycleScheduler(interval_seconds=300, lockout_checker=None)
        assert scheduler.should_run() is True

    def test_overlapping_cycle_blocks_run(self):
        """Should not run if previous cycle is still running."""
        scheduler = CycleScheduler(interval_seconds=1)
        scheduler.mark_cycle_start()
        # Don't call mark_cycle_complete — cycle still running
        # Force elapsed time
        scheduler._last_cycle_time = time.time() - 10
        assert scheduler.should_run() is False

    def test_check_order_kill_switch_before_lockout(self):
        """Kill switch should be checked before lockout (both active)."""
        # If kill switch is active, lockout callable should not be called
        call_count = {"n": 0}

        def lockout_fn():
            call_count["n"] += 1
            return True

        scheduler = CycleScheduler(interval_seconds=300, lockout_checker=lockout_fn)
        with patch.dict(os.environ, {"KILL_SWITCH": "1"}):
            result = scheduler.should_run()
        assert result is False
        assert call_count["n"] == 0  # Lockout not checked


class TestCycleSchedulerMarkers:
    """Tests for mark_cycle_start() and mark_cycle_complete() methods."""

    def test_mark_cycle_start_sets_running(self):
        """mark_cycle_start should set _cycle_running to True."""
        scheduler = CycleScheduler(interval_seconds=300)
        scheduler.mark_cycle_start()
        assert scheduler._cycle_running is True

    def test_mark_cycle_start_updates_last_cycle_time(self):
        """mark_cycle_start should update _last_cycle_time to current time."""
        scheduler = CycleScheduler(interval_seconds=300)
        before = time.time()
        scheduler.mark_cycle_start()
        after = time.time()
        assert before <= scheduler._last_cycle_time <= after

    def test_mark_cycle_complete_clears_running(self):
        """mark_cycle_complete should set _cycle_running to False."""
        scheduler = CycleScheduler(interval_seconds=300)
        scheduler.mark_cycle_start()
        scheduler.mark_cycle_complete()
        assert scheduler._cycle_running is False

    def test_full_cycle_lifecycle(self):
        """Complete start → complete cycle allows next run after interval."""
        scheduler = CycleScheduler(interval_seconds=1)
        # First run
        assert scheduler.should_run() is True
        scheduler.mark_cycle_start()
        assert scheduler._cycle_running is True
        scheduler.mark_cycle_complete()
        assert scheduler._cycle_running is False
        # Immediately after — interval not elapsed
        assert scheduler.should_run() is False
        # Simulate time passing
        scheduler._last_cycle_time = time.time() - 2
        assert scheduler.should_run() is True


class TestCycleSchedulerInit:
    """Tests for CycleScheduler initialization and startup logging."""

    def test_default_timeframes(self):
        """Default timeframes should be 60min -> 15min -> 5min."""
        scheduler = CycleScheduler(interval_seconds=300)
        assert scheduler._timeframes == ["60min", "15min", "5min"]

    def test_custom_timeframes(self):
        """Custom timeframes should be stored."""
        tfs = ["1h", "30min", "5min"]
        scheduler = CycleScheduler(interval_seconds=60, timeframes=tfs)
        assert scheduler._timeframes == tfs

    def test_interval_stored(self):
        """Interval should be stored as attribute."""
        scheduler = CycleScheduler(interval_seconds=120)
        assert scheduler.interval == 120

    def test_initial_state(self):
        """Initial state: not running, last cycle time is 0."""
        scheduler = CycleScheduler(interval_seconds=300)
        assert scheduler._cycle_running is False
        assert scheduler._last_cycle_time == 0

    def test_startup_log_message(self, caplog):
        """Scheduler should log config at startup in Europe/Rome timezone."""
        import logging

        with caplog.at_level(logging.INFO, logger="ig-scalper"):
            scheduler = CycleScheduler(
                interval_seconds=300,
                timeframes=["60min", "15min", "5min"],
            )

        assert any("CycleScheduler initialized" in msg for msg in caplog.messages)
        assert any("interval=300s" in msg for msg in caplog.messages)
        assert any("60min -> 15min -> 5min" in msg for msg in caplog.messages)


class TestCycleSchedulerWithDailyLockout:
    """Tests integrating CycleScheduler with core.risk.daily_lockout."""

    def test_integration_with_daily_lockout_function(self):
        """CycleScheduler works with core.risk.daily_lockout as checker."""
        from core.risk import daily_lockout

        # Simulate daily P&L of -6% with max allowed -5%
        lockout_fn = lambda: daily_lockout(-6.0, 5.0)
        scheduler = CycleScheduler(interval_seconds=300, lockout_checker=lockout_fn)
        assert scheduler.should_run() is False

    def test_integration_lockout_not_triggered(self):
        """CycleScheduler allows run when lockout not triggered."""
        from core.risk import daily_lockout

        # Simulate daily P&L of -2% with max allowed -5%
        lockout_fn = lambda: daily_lockout(-2.0, 5.0)
        scheduler = CycleScheduler(interval_seconds=300, lockout_checker=lockout_fn)
        assert scheduler.should_run() is True
