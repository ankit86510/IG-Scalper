"""
Cycle Scheduler for FVG Multi-Timeframe Strategy.

Manages cycle timing, kill switch, and daily lockout checks.
"""

import logging
import os
import time
from datetime import datetime
from typing import Callable, List, Optional
from zoneinfo import ZoneInfo

from core.risk import daily_lockout

logger = logging.getLogger("ig-scalper")

ROME_TZ = ZoneInfo("Europe/Rome")


class CycleScheduler:
    """Manages cycle timing, kill switch, and lockout checks."""

    def __init__(
        self,
        interval_seconds: int,
        timeframes: Optional[List[str]] = None,
        lockout_checker: Optional[Callable[[], bool]] = None,
    ):
        """
        Args:
            interval_seconds: Seconds between analysis cycles.
            timeframes: List of timeframes in the cascade (for startup logging).
            lockout_checker: Optional callable returning True if daily lockout is active.
                             If None, lockout check is skipped.
        """
        self.interval = interval_seconds
        self._last_cycle_time: float = 0
        self._cycle_running: bool = False
        self._timeframes = timeframes or ["60min", "15min", "5min"]
        self._lockout_checker = lockout_checker

        # Log startup configuration in Europe/Rome timezone
        now_rome = datetime.now(tz=ROME_TZ)
        logger.info(
            f"CycleScheduler initialized at {now_rome.strftime('%Y-%m-%d %H:%M:%S %Z')} | "
            f"interval={self.interval}s | "
            f"timeframes={' -> '.join(self._timeframes)}"
        )

    def should_run(self) -> bool:
        """
        Returns True if a new cycle should execute.

        Checks:
        - Interval elapsed since last cycle
        - KILL_SWITCH != "1"
        - Daily lockout is not active
        - Previous cycle is not still running
        """
        # Check kill switch
        if os.environ.get("KILL_SWITCH") == "1":
            now_rome = datetime.now(tz=ROME_TZ)
            logger.warning(
                f"Cycle skipped at {now_rome.strftime('%Y-%m-%d %H:%M:%S %Z')}: "
                f"KILL_SWITCH is active"
            )
            return False

        # Check daily lockout
        if self._lockout_checker is not None:
            if self._lockout_checker():
                now_rome = datetime.now(tz=ROME_TZ)
                logger.warning(
                    f"Cycle skipped at {now_rome.strftime('%Y-%m-%d %H:%M:%S %Z')}: "
                    f"daily loss lockout is active"
                )
                return False

        # Check overlapping cycle
        if self._cycle_running:
            now_rome = datetime.now(tz=ROME_TZ)
            logger.warning(
                f"Cycle skipped at {now_rome.strftime('%Y-%m-%d %H:%M:%S %Z')}: "
                f"previous cycle still running"
            )
            return False

        # Check interval elapsed
        now = time.time()
        elapsed = now - self._last_cycle_time
        if elapsed < self.interval:
            return False

        return True

    def mark_cycle_start(self) -> None:
        """Mark that a cycle has started. Call before running analysis."""
        self._cycle_running = True
        self._last_cycle_time = time.time()

    def mark_cycle_complete(self) -> None:
        """Mark that a cycle has completed. Call after analysis finishes."""
        self._cycle_running = False
