"""
Rate Budget Manager for FVG Multi-Timeframe Strategy.

Tracks and enforces TwelveData API request budgets across analysis cycles.
Wraps the existing multi_data_provider rate limiting infrastructure with
cycle-level budget verification and consumption logging.
"""

import logging
from typing import Dict, Optional

logger = logging.getLogger("ig-scalper")


class RateBudgetManager:
    """Manages rate limit compliance for FVG analysis cycles.

    Provides pre-cycle budget verification and post-cycle consumption
    logging. Leverages the TwelveDataProvider's built-in per-minute
    rate limiting and caching to avoid redundant requests.

    The manager does NOT enforce limits itself — it queries the existing
    provider's budget status and decides whether a cycle should proceed.
    """

    def __init__(
        self,
        data_provider,
        num_timeframes: int = 3,
        num_symbols: int = 1,
    ):
        """Initialize RateBudgetManager.

        Args:
            data_provider: The SmartDataAggregator or provider that wraps
                TwelveDataProvider. Must expose a TwelveData provider
                internally for budget queries.
            num_timeframes: Number of timeframes fetched per cycle (default 3).
            num_symbols: Number of symbols per cycle (default 1).
        """
        self._data_provider = data_provider
        self._num_timeframes = num_timeframes
        self._num_symbols = num_symbols
        self._requests_this_cycle: int = 0

    @property
    def requests_per_cycle(self) -> int:
        """Total requests needed for one full analysis cycle.

        Req 6.1: number_of_timeframes × number_of_symbols
        """
        return self._num_timeframes * self._num_symbols

    def _get_twelvedata_provider(self):
        """Extract the TwelveDataProvider from the data_provider.

        Handles both SmartDataAggregator (which has a list of providers)
        and direct TwelveDataProvider instances.
        """
        # Direct TwelveDataProvider
        if hasattr(self._data_provider, "get_budget_status"):
            return self._data_provider

        # SmartDataAggregator wraps providers as (name, instance) tuples
        if hasattr(self._data_provider, "providers"):
            for name, provider in self._data_provider.providers:
                if name == "TwelveData" and hasattr(provider, "get_budget_status"):
                    return provider

        return None

    def get_budget_status(self) -> Optional[Dict]:
        """Query current budget status from the TwelveData provider.

        Returns:
            Dict with daily_used, daily_remaining, daily_limit,
            minute_used, minute_limit, etc. or None if provider
            unavailable.
        """
        provider = self._get_twelvedata_provider()
        if provider is None:
            return None
        return provider.get_budget_status()

    def has_sufficient_daily_budget(self) -> bool:
        """Check if enough daily budget remains for a full cycle.

        Req 6.2, 6.3: Verify daily budget remaining >= requests per cycle.

        Returns:
            True if sufficient budget exists, False otherwise.
        """
        budget = self.get_budget_status()
        if budget is None:
            # No TwelveData provider — assume OK (fallback providers used)
            logger.warning(
                "RateBudgetManager: No TwelveData provider found, "
                "cannot verify daily budget — proceeding anyway"
            )
            return True

        daily_remaining = budget.get("daily_remaining", 0)
        needed = self.requests_per_cycle

        if daily_remaining < needed:
            logger.warning(
                f"RateBudgetManager: Insufficient daily budget — "
                f"remaining={daily_remaining}, needed={needed}, "
                f"daily_used={budget.get('daily_used', '?')}, "
                f"daily_limit={budget.get('daily_limit', '?')}"
            )
            return False

        return True

    def has_sufficient_minute_budget(self) -> bool:
        """Check if per-minute budget has capacity.

        Req 6.4: If per-minute budget is exhausted, the existing
        multi_data_provider will sleep/wait internally. This check
        provides visibility but does NOT block — the actual enforcement
        is in TwelveDataProvider.get_bars().

        Returns:
            True if minute budget has capacity, False if exhausted
            (provider will handle the wait internally).
        """
        budget = self.get_budget_status()
        if budget is None:
            return True

        minute_used = budget.get("minute_used", 0)
        minute_limit = budget.get("minute_limit", 7)

        # Provider handles waiting internally, but we log it
        if minute_used >= minute_limit:
            logger.info(
                f"RateBudgetManager: Per-minute budget exhausted — "
                f"used={minute_used}/{minute_limit}. "
                f"TwelveDataProvider will wait for sliding window to clear."
            )
            return False

        return True

    def should_proceed_with_cycle(self) -> bool:
        """Pre-cycle gate: verify sufficient budget exists.

        Req 6.2: Before fetching, verify daily and per-minute budget.
        Req 6.3: If daily remaining < requests needed → skip cycle.
        Req 6.4: Per-minute exhaustion is handled by provider (wait).

        Returns:
            True if the cycle should proceed, False to skip.
        """
        # Daily budget is a hard gate — skip if insufficient
        if not self.has_sufficient_daily_budget():
            return False

        # Per-minute is informational — provider handles the wait
        # Just log the status for visibility
        self.has_sufficient_minute_budget()

        return True

    def record_cycle_start(self) -> None:
        """Reset per-cycle request counter at cycle start."""
        self._requests_this_cycle = 0

    def record_request(self) -> None:
        """Record that a request was made during the current cycle."""
        self._requests_this_cycle += 1

    def log_cycle_consumption(self) -> None:
        """Log budget consumption after a completed cycle.

        Req 6.6: Log daily_used, daily_remaining, requests_this_cycle.
        """
        budget = self.get_budget_status()

        if budget is None:
            logger.info(
                f"RateBudgetManager: Cycle complete — "
                f"requests_this_cycle={self._requests_this_cycle} "
                f"(TwelveData budget status unavailable)"
            )
            return

        logger.info(
            f"RateBudgetManager: Cycle complete — "
            f"requests_this_cycle={self._requests_this_cycle}, "
            f"daily_used={budget.get('daily_used', '?')}, "
            f"daily_remaining={budget.get('daily_remaining', '?')}, "
            f"minute_used={budget.get('minute_used', '?')}/{budget.get('minute_limit', '?')}"
        )
