"""
Risk-Per-Trade Position Sizer

Calculates deal size based on account equity, configured risk percentage,
actual stop distance, and instrument pip value.

Formula: size = floor((equity × risk_pct / 100) / (stop_distance × pip_value) / step) × step
"""

import logging
import math
import time

log = logging.getLogger("ig-scalper")


class RiskPositionSizer:
    """Equity-based position sizing: size = (equity × risk%) / (stop × pip_value)."""

    def __init__(self, config: dict, ig_client):
        """
        Initialise the position sizer.

        Args:
            config: dict with keys:
                risk_pct_per_trade: float (default 2.0)
                equity_refresh_interval_seconds: int (default 300)
                use_dynamic_sizing: bool (default True)
                max_size_multiple: int (default 50)
            ig_client: IGClient instance for fetching account equity
        """
        self.ig_client = ig_client
        self.risk_pct = config.get("risk_pct_per_trade", 2.0)
        self.refresh_interval = config.get("equity_refresh_interval_seconds", 300)
        self.use_dynamic_sizing = config.get("use_dynamic_sizing", True)
        self.max_size_multiple = config.get("max_size_multiple", 50)

        # State
        self._cached_equity: float = config.get("account_equity", 10000.0)
        self._last_refresh_time: float = 0.0  # force refresh on first call

    def refresh_equity(self) -> float:
        """Fetch account balance from IG API. Cache on failure.

        Uses the account balance field (not available-to-deal) as the
        equity basis for calculation (Requirement 9.3).

        Returns:
            Current equity value (cached if API call fails).
        """
        try:
            response = self.ig_client.account_summary()
            # IG API returns {"accounts": [{"balance": {"balance": ...}}]}
            accounts = response.get("accounts", [])
            if accounts:
                balance = accounts[0].get("balance", {}).get("balance")
                if balance is not None:
                    self._cached_equity = float(balance)
                    self._last_refresh_time = time.time()
                    log.debug(f"Equity refreshed: {self._cached_equity:.2f}")
                    return self._cached_equity

            log.warning("IG account summary returned no valid balance, using cached equity")
        except Exception as e:
            log.warning(f"Failed to refresh equity from IG API: {e}. Using cached value: {self._cached_equity:.2f}")

        return self._cached_equity

    def get_equity(self) -> float:
        """Return current cached equity, refreshing if interval has elapsed."""
        now = time.time()
        if now - self._last_refresh_time >= self.refresh_interval:
            self.refresh_equity()
        return self._cached_equity

    def calculate_size(
        self,
        stop_distance: float,
        pip_value: float,
        min_size: float,
        size_step: float,
    ) -> tuple[float | None, dict]:
        """
        Calculate position size based on risk-per-trade formula.

        Args:
            stop_distance: Distance in points between entry and stop loss
            pip_value: Monetary value per point of movement for one unit
            min_size: Minimum deal size for the instrument
            size_step: Valid size increment (e.g. 0.1)

        Returns:
            Tuple of (size or None if rejected, metadata dict).
            metadata includes: equity, raw_size, capped_size, reason.
            None means risk budget insufficient for minimum size or invalid inputs.
        """
        equity = self.get_equity()

        metadata = {
            "equity": equity,
            "risk_pct": self.risk_pct,
            "stop_distance": stop_distance,
            "pip_value": pip_value,
            "raw_size": 0.0,
            "capped_size": None,
            "reason": "",
        }

        # Division by zero guard
        if stop_distance <= 0 or pip_value <= 0:
            metadata["reason"] = (
                f"Invalid inputs: stop_distance={stop_distance}, pip_value={pip_value}"
            )
            log.error(
                f"Position sizer: division by zero prevented — "
                f"stop_distance={stop_distance}, pip_value={pip_value}"
            )
            return None, metadata

        # Ensure size_step is positive
        if size_step <= 0:
            size_step = 0.1

        # Core formula: size = floor((equity × risk_pct / 100) / (stop × pip_value) / step) × step
        risk_amount = equity * self.risk_pct / 100.0
        raw_size = risk_amount / (stop_distance * pip_value)
        metadata["raw_size"] = raw_size

        # Round down to nearest valid increment
        sized = math.floor(raw_size / size_step) * size_step

        # Check minimum size
        if sized < min_size:
            metadata["reason"] = (
                f"Risk budget insufficient: calculated size {sized:.4f} < min_size {min_size}"
            )
            log.info(
                f"Position sizer REJECTED: size {sized:.4f} below minimum {min_size} "
                f"(equity={equity:.2f}, risk={self.risk_pct}%, stop={stop_distance})"
            )
            return None, metadata

        # Cap at max_size_multiple × min_size
        max_size = min_size * self.max_size_multiple
        if sized > max_size:
            sized = max_size
            metadata["capped_size"] = max_size
            metadata["reason"] = f"Size capped at {max_size} (max_size_multiple={self.max_size_multiple})"
            log.info(
                f"Position sizer: size capped from {raw_size:.4f} to {max_size} "
                f"(max {self.max_size_multiple}× min_size)"
            )
        else:
            metadata["capped_size"] = sized
            metadata["reason"] = "ok"

        return sized, metadata
