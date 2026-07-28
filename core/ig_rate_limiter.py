"""
IG REST API Rate Limiter

Tracks all non-trading API requests against IG's limits and proactively
prevents hitting them. Persists state to disk so restarts don't lose count.

IG Demo limits:
  - Non-trading requests: 30/minute (shared: /markets, /prices, /positions, etc.)
  - Historical price data points: 10,000/week (resets every Monday 00:00 UTC)
  - Trading requests: 60/minute (separate budget: /positions/otc, /workingorders/otc)

This module budgets conservatively:
  - Non-trading: max 25/minute (83% of 30 — leaves headroom)
  - Historical prices: max 8,000/week (80% of 10,000 — safety margin)
  - Daily price budget: 8,000 / 5 trading days = 1,600/day max

Usage:
    from core.ig_rate_limiter import IGRateLimiter
    limiter = IGRateLimiter()

    if limiter.can_request_prices():
        limiter.record_price_request(data_points=200)
        # ... make request ...

    if limiter.can_request_non_trading():
        limiter.record_non_trading_request()
        # ... make request ...
"""

import json
import logging
import os
import threading
import time
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

STATE_FILE = "data/ig_rate_limiter_state.json"

# --- Configurable limits ---
# Non-trading (per minute)
NON_TRADING_PER_MINUTE = 25  # Conservative: 25 of 30 allowed

# Historical prices (weekly)
WEEKLY_PRICE_POINTS_LIMIT = 8000  # Conservative: 8000 of 10,000
DAILY_PRICE_POINTS_LIMIT = 1600   # 8000 / 5 trading days

# Per-minute cap on /prices requests specifically (avoid bursts)
PRICES_PER_MINUTE = 6  # Max 6 /prices calls per minute


class IGRateLimiter:
    """Centralized rate limiter for all IG REST API calls."""

    def __init__(self, state_file: str = STATE_FILE):
        self._state_file = state_file
        self._lock = threading.Lock()

        # Non-trading sliding window (timestamps of last 60s)
        self._non_trading_times: list = []

        # /prices sliding window (timestamps of last 60s)
        self._price_request_times: list = []

        # Weekly/daily price point counters
        self._weekly_points = 0
        self._daily_points = 0
        self._week_start = self._get_week_start()
        self._day_start = self._get_day_start()

        # 403 backoff
        self._prices_blocked_until = 0.0
        self._non_trading_blocked_until = 0.0

        # Load persisted state
        self._load_state()

    # ─── PUBLIC API ──────────────────────────────────────────────────────────

    def can_request_prices(self, data_points: int = 200) -> bool:
        """Check if a /prices request is allowed.

        Checks:
          1. Not in 403 backoff period
          2. Per-minute /prices cap not exceeded
          3. Daily price point budget not exceeded
          4. Weekly price point budget not exceeded
          5. General non-trading per-minute cap not exceeded
        """
        with self._lock:
            self._rotate_periods()
            now = time.time()

            if now < self._prices_blocked_until:
                return False

            if len(self._price_request_times) >= PRICES_PER_MINUTE:
                return False

            if self._daily_points + data_points > DAILY_PRICE_POINTS_LIMIT:
                return False

            if self._weekly_points + data_points > WEEKLY_PRICE_POINTS_LIMIT:
                return False

            if len(self._non_trading_times) >= NON_TRADING_PER_MINUTE:
                return False

            return True

    def record_price_request(self, data_points: int = 200):
        """Record a successful /prices request."""
        with self._lock:
            now = time.time()
            self._price_request_times.append(now)
            self._non_trading_times.append(now)
            self._daily_points += data_points
            self._weekly_points += data_points
            self._save_state()

    def record_price_403(self):
        """Record a 403 on /prices — back off for 6 hours."""
        with self._lock:
            self._prices_blocked_until = time.time() + 6 * 3600
            logger.warning(
                f"[RateLimiter] /prices 403 — blocked until "
                f"{datetime.fromtimestamp(self._prices_blocked_until, tz=timezone.utc).strftime('%H:%M UTC')}"
            )
            self._save_state()

    def can_request_non_trading(self) -> bool:
        """Check if a non-trading request (/markets, /positions, etc.) is allowed."""
        with self._lock:
            self._rotate_periods()
            now = time.time()

            if now < self._non_trading_blocked_until:
                return False

            if len(self._non_trading_times) >= NON_TRADING_PER_MINUTE:
                return False

            return True

    def record_non_trading_request(self):
        """Record a non-trading request."""
        with self._lock:
            self._non_trading_times.append(time.time())

    def record_non_trading_403(self):
        """Record a 403 on non-trading endpoint — back off 2 minutes."""
        with self._lock:
            self._non_trading_blocked_until = time.time() + 120
            logger.warning("[RateLimiter] Non-trading 403 — backing off 2 minutes")

    def get_status(self) -> dict:
        """Return current rate limiter status for logging."""
        with self._lock:
            self._rotate_periods()
            now = time.time()
            return {
                "non_trading_last_min": len(self._non_trading_times),
                "non_trading_limit": NON_TRADING_PER_MINUTE,
                "prices_last_min": len(self._price_request_times),
                "prices_limit": PRICES_PER_MINUTE,
                "daily_points_used": self._daily_points,
                "daily_points_limit": DAILY_PRICE_POINTS_LIMIT,
                "weekly_points_used": self._weekly_points,
                "weekly_points_limit": WEEKLY_PRICE_POINTS_LIMIT,
                "prices_blocked": now < self._prices_blocked_until,
                "prices_blocked_remaining_s": max(0, self._prices_blocked_until - now),
            }

    # ─── INTERNAL ────────────────────────────────────────────────────────────

    def _rotate_periods(self):
        """Rotate daily/weekly counters if period has changed. Trim sliding windows."""
        now = time.time()

        # Trim per-minute sliding windows
        cutoff = now - 60
        self._non_trading_times = [t for t in self._non_trading_times if t > cutoff]
        self._price_request_times = [t for t in self._price_request_times if t > cutoff]

        # Check daily reset
        current_day = self._get_day_start()
        if current_day != self._day_start:
            logger.info(
                f"[RateLimiter] Daily reset: {self._daily_points} points used yesterday"
            )
            self._day_start = current_day
            self._daily_points = 0

        # Check weekly reset (Monday 00:00 UTC)
        current_week = self._get_week_start()
        if current_week != self._week_start:
            logger.info(
                f"[RateLimiter] Weekly reset: {self._weekly_points} points used last week"
            )
            self._week_start = current_week
            self._weekly_points = 0
            # Also clear any 403 backoff on weekly reset
            self._prices_blocked_until = 0.0

    @staticmethod
    def _get_week_start() -> str:
        """Get current ISO week start (Monday) as string for comparison."""
        now = datetime.now(timezone.utc)
        # Monday of current week
        monday = now.date() - __import__('datetime').timedelta(days=now.weekday())
        return monday.isoformat()

    @staticmethod
    def _get_day_start() -> str:
        """Get current day as string for comparison."""
        return datetime.now(timezone.utc).strftime("%Y-%m-%d")

    def _save_state(self):
        """Persist counters to disk (survives restarts)."""
        try:
            state = {
                "weekly_points": self._weekly_points,
                "daily_points": self._daily_points,
                "week_start": self._week_start,
                "day_start": self._day_start,
                "prices_blocked_until": self._prices_blocked_until,
                "saved_at": time.time(),
            }
            os.makedirs(os.path.dirname(self._state_file), exist_ok=True)
            with open(self._state_file, 'w') as f:
                json.dump(state, f)
        except Exception as e:
            logger.debug(f"[RateLimiter] Failed to save state: {e}")

    def _load_state(self):
        """Load persisted state from disk."""
        try:
            with open(self._state_file, 'r') as f:
                state = json.load(f)

            saved_week = state.get("week_start", "")
            saved_day = state.get("day_start", "")

            # Only restore if same week
            if saved_week == self._week_start:
                self._weekly_points = state.get("weekly_points", 0)
            else:
                logger.info("[RateLimiter] New week — starting fresh")

            # Only restore daily if same day
            if saved_day == self._day_start:
                self._daily_points = state.get("daily_points", 0)

            # Restore backoff
            blocked = state.get("prices_blocked_until", 0.0)
            if blocked > time.time():
                self._prices_blocked_until = blocked

            logger.info(
                f"[RateLimiter] Loaded state: weekly={self._weekly_points}/{WEEKLY_PRICE_POINTS_LIMIT}, "
                f"daily={self._daily_points}/{DAILY_PRICE_POINTS_LIMIT}"
            )

        except (FileNotFoundError, json.JSONDecodeError):
            logger.info("[RateLimiter] No saved state — starting fresh")
        except Exception as e:
            logger.warning(f"[RateLimiter] Failed to load state: {e}")
