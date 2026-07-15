"""Conditional Order Manager — manages working order lifecycle."""

import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Dict, Optional, Tuple

import requests


@dataclass
class TrackedOrder:
    """Internal state for a tracked working order."""

    epic: str
    deal_id: str
    direction: str  # "BUY" or "SELL"
    entry_level: float
    stop_distance: float
    tp_distance: Optional[float]
    size: float
    currency_code: str
    placed_at: datetime  # UTC
    expiry_at: datetime  # UTC
    confidence: float
    patterns: list
    cancel_retry_count: int = 0  # Tracks consecutive cancellation failures


# Required config keys for conditional orders
_REQUIRED_KEYS = ("enabled", "buffer_points", "order_expiry_seconds", "max_entry_distance_points")


class ConditionalOrderManager:
    """Manages conditional (working) order lifecycle."""

    def __init__(self, ig_client, config: dict, position_manager,
                 trailing_manager, sr_detector, log):
        self.ig_client = ig_client
        self.config = config
        self.position_manager = position_manager
        self.trailing_manager = trailing_manager
        self.sr_detector = sr_detector
        self.log = log

        # Validate configuration
        self.enabled = self._validate_config(config)

        # Internal tracking state
        self.tracked_orders: Dict[str, TrackedOrder] = {}  # key = epic
        self.active_signals: Dict[str, str] = {}  # key = epic, value = direction

        # Sync existing working orders from IG on startup
        if self.enabled:
            self._sync_existing_orders()

    def process_signal(self, epic: str, direction: str, mid_price: float,
                       sr_levels: dict, stop_pts: float, tp_pts: float,
                       size: float, currency_code: str, confidence: float,
                       patterns: list, atr_value: float) -> dict:
        """
        Main entry point. Calculates entry level and places working order.
        Returns: {"action": "placed"|"rejected"|"fallback"|"skipped", "details": {...}}
        """
        # 1. Check if conditional orders are enabled
        if not self.enabled:
            return {"action": "skipped", "details": {"reason": "disabled"}}

        # 2. Check if position already open for this epic (Req 5.3)
        if epic in self.position_manager.positions:
            return {"action": "skipped", "details": {"reason": "position_open"}}

        # 3. Check for existing pending order — duplicate or reversal (Req 4.1, 4.2)
        if epic in self.tracked_orders:
            existing_order = self.tracked_orders[epic]
            if existing_order.direction == direction:
                # Same direction → keep existing, skip duplicate (Req 4.2)
                return {"action": "skipped", "details": {"reason": "duplicate_order"}}
            else:
                # Opposite direction → cancel existing first, then place new (Req 4.1)
                self.cancel_order(epic, "signal_reversal")

        # 4. Calculate entry level from S/R zones
        entry_level = self.calculate_entry_level(direction, mid_price, sr_levels)

        if entry_level is None:
            # No suitable S/R level found — fall back to market order (Req 1.3, 1.4)
            fallback_reason = (
                "no_resistance_level" if direction == "BUY" else "no_support_level"
            )
            # Log per Req 8.5
            self.log.info(
                f"Conditional order fallback: epic={epic}, direction={direction}, "
                f"reason={fallback_reason}"
            )
            # Place market order as fallback
            try:
                self.ig_client.place_order(
                    epic=epic,
                    direction=direction,
                    size=size,
                    currency_code=currency_code,
                    stop_distance=stop_pts,
                    limit_distance=tp_pts if tp_pts else None,
                )
            except requests.exceptions.Timeout as e:
                self.log.error(
                    f"Fallback market order timeout: epic={epic}, "
                    f"error_type=timeout, details={e}"
                )
            except requests.exceptions.HTTPError as e:
                status_code = e.response.status_code if e.response is not None else None
                response_text = e.response.text if e.response is not None else ""
                self.log.error(
                    f"Fallback market order HTTP error: epic={epic}, "
                    f"error_type=http_error, status_code={status_code}, "
                    f"response={response_text}"
                )
            except requests.exceptions.ConnectionError as e:
                self.log.error(
                    f"Fallback market order connection error: epic={epic}, "
                    f"error_type=connection_error, details={e}"
                )
            except Exception as e:
                self.log.error(
                    f"Fallback market order failed: epic={epic}, "
                    f"error_type={type(e).__name__}, details={e}"
                )
            return {"action": "fallback", "details": {"reason": fallback_reason,
                                                      "epic": epic,
                                                      "direction": direction}}

        # 5. Validate max entry distance (Req 1.6, 8.4)
        max_distance = self.config["conditional_orders"]["max_entry_distance_points"]
        distance = abs(entry_level - mid_price)

        if distance > max_distance:
            self.log.warning(
                f"Conditional order rejected: epic={epic}, "
                f"entry_level={entry_level}, current_price={mid_price}, "
                f"distance={distance:.2f}, max_distance={max_distance}"
            )
            return {
                "action": "rejected",
                "details": {
                    "reason": "max_distance_exceeded",
                    "epic": epic,
                    "entry_level": entry_level,
                    "current_price": mid_price,
                    "distance": distance,
                    "max_distance": max_distance,
                },
            }

        # 6. Compute expiry, build payload, place working order
        expiry_timestamp = self.compute_expiry_timestamp()
        buffer = self.config["conditional_orders"]["buffer_points"]

        payload = self.build_order_payload(
            epic=epic,
            direction=direction,
            entry_level=entry_level,
            size=size,
            stop_distance=stop_pts,
            tp_distance=tp_pts if tp_pts else None,
            currency_code=currency_code,
            expiry_timestamp=expiry_timestamp,
        )

        try:
            response = self.ig_client.place_working_order(
                epic=epic,
                direction=direction,
                level=entry_level,
                size=size,
                stop_distance=stop_pts,
                limit_distance=tp_pts if tp_pts else None,
                good_till_date=expiry_timestamp,
                currency_code=currency_code,
            )
        except requests.exceptions.Timeout as e:
            self.log.error(
                f"Working order placement timeout: epic={epic}, "
                f"error_type=timeout, details={e}"
            )
            return {
                "action": "rejected",
                "details": {
                    "reason": "api_error",
                    "error_type": "timeout",
                    "epic": epic,
                    "error": str(e),
                },
            }
        except requests.exceptions.HTTPError as e:
            status_code = e.response.status_code if e.response is not None else None
            response_text = e.response.text if e.response is not None else ""
            self.log.error(
                f"Working order placement HTTP error: epic={epic}, "
                f"error_type=http_error, status_code={status_code}, "
                f"response={response_text}"
            )
            return {
                "action": "rejected",
                "details": {
                    "reason": "api_error",
                    "error_type": "http_error",
                    "epic": epic,
                    "status_code": status_code,
                    "error": str(e),
                    "response": response_text,
                },
            }
        except requests.exceptions.ConnectionError as e:
            self.log.error(
                f"Working order placement connection error: epic={epic}, "
                f"error_type=connection_error, details={e}"
            )
            return {
                "action": "rejected",
                "details": {
                    "reason": "api_error",
                    "error_type": "connection_error",
                    "epic": epic,
                    "error": str(e),
                },
            }
        except Exception as e:
            self.log.error(
                f"Working order placement failed: epic={epic}, "
                f"error_type={type(e).__name__}, details={e}"
            )
            return {
                "action": "rejected",
                "details": {
                    "reason": "api_error",
                    "error_type": type(e).__name__,
                    "epic": epic,
                    "error": str(e),
                },
            }

        # Check for IG API rejection response (HTTP 200 but order rejected)
        deal_status = response.get("dealStatus")
        if deal_status == "REJECTED":
            reject_reason = response.get("reason", "unknown")
            self.log.error(
                f"Working order placement rejected by IG: epic={epic}, "
                f"error_type=rejection, reason={reject_reason}, "
                f"response={response}"
            )
            return {
                "action": "rejected",
                "details": {
                    "reason": "api_error",
                    "error_type": "rejection",
                    "epic": epic,
                    "reject_reason": reject_reason,
                    "response": response,
                },
            }

        # 7. Track the placed order
        # 7. Confirm the deal to get the actual dealId (IG returns dealReference first)
        deal_ref = response.get("dealReference", "")
        deal_id = deal_ref  # fallback if confirm fails
        try:
            confirm = self.ig_client.confirm_deal(deal_ref)
            deal_id = confirm.get("dealId", deal_ref)
        except Exception as e:
            self.log.warning(
                f"Could not confirm deal reference {deal_ref}: {e} — using dealReference as fallback"
            )

        now_utc = datetime.now(timezone.utc)
        expiry_seconds = self.config["conditional_orders"]["order_expiry_seconds"]
        expiry_at = now_utc + timedelta(seconds=expiry_seconds)

        tracked = TrackedOrder(
            epic=epic,
            deal_id=deal_id,
            direction=direction,
            entry_level=entry_level,
            stop_distance=stop_pts,
            tp_distance=tp_pts if tp_pts else None,
            size=size,
            currency_code=currency_code,
            placed_at=now_utc,
            expiry_at=expiry_at,
            confidence=confidence,
            patterns=patterns,
        )
        self.tracked_orders[epic] = tracked
        self.active_signals[epic] = direction

        # 8. Log per Req 8.1
        self.log.info(
            f"Conditional order placed: epic={epic}, direction={direction}, "
            f"entry_level={entry_level}, stop_distance={stop_pts}, "
            f"expiry_time={expiry_timestamp}, buffer={buffer}"
        )

        return {
            "action": "placed",
            "details": {
                "epic": epic,
                "direction": direction,
                "entry_level": entry_level,
                "stop_distance": stop_pts,
                "tp_distance": tp_pts if tp_pts else None,
                "expiry_time": expiry_timestamp,
                "deal_reference": deal_ref,
            },
        }

    def calculate_entry_level(self, direction: str, mid_price: float,
                              sr_levels: dict) -> Optional[float]:
        """
        Selects nearest S/R level and applies buffer.
        Returns entry level or None if no suitable level found.

        For BUY: nearest resistance above mid_price + buffer_points
        For SELL: nearest support below mid_price - buffer_points
        """
        buffer = self.config["conditional_orders"]["buffer_points"]

        if direction == "BUY":
            resistance_levels = sr_levels.get("resistance", [])
            # Filter resistance levels that are above mid_price
            candidates = [lvl for lvl in resistance_levels if lvl > mid_price]
            if not candidates:
                return None
            # Pick the nearest (smallest absolute distance from mid_price)
            nearest = min(candidates, key=lambda lvl: abs(lvl - mid_price))
            return nearest + buffer

        elif direction == "SELL":
            support_levels = sr_levels.get("support", [])
            # Filter support levels that are below mid_price
            candidates = [lvl for lvl in support_levels if lvl < mid_price]
            if not candidates:
                return None
            # Pick the nearest (smallest absolute distance from mid_price)
            nearest = min(candidates, key=lambda lvl: abs(lvl - mid_price))
            return nearest - buffer

        return None

    def calculate_stop_tp(self, atr_value: float, market_min_stop: float = 0.0) -> Tuple[float, Optional[float]]:
        """Calculate stop distance and optional take-profit distance.

        Logic:
          1. raw_stop = atr_value * stop_multiplier
          2. final_stop = max(raw_stop, min_stop_pts, market_min_stop)
          3. if use_tp_limit: tp = final_stop * rr_take
          4. else: tp = None

        The TP is calculated AFTER market rules enforcement (step 2),
        so the R:R ratio is always maintained relative to the enforced stop.

        Returns: (final_stop: float, tp_distance: Optional[float])
        """
        ai_strategy = self.config.get("ai_strategy", {})
        execution = self.config.get("execution", {})

        stop_multiplier = ai_strategy.get("stop_multiplier", 2.0)
        min_stop_pts = ai_strategy.get("min_stop_pts", 5.0)
        rr_take = ai_strategy.get("rr_take", 2.0)
        use_tp_limit = execution.get("use_tp_limit", False)

        # Step 1: raw stop from ATR
        raw_stop = atr_value * stop_multiplier

        # Step 2: enforce floor at min_stop_pts and market minimum
        final_stop = max(raw_stop, min_stop_pts, market_min_stop)

        # Step 3 & 4: TP calculation (after market rules enforcement)
        tp_distance: Optional[float] = None
        if use_tp_limit:
            tp_distance = final_stop * rr_take

        return (final_stop, tp_distance)

    def compute_expiry_timestamp(self) -> str:
        """Compute goodTillDate as current UTC + order_expiry_seconds.

        Format: yyyy/MM/dd HH:mm:ss (IG API required format for working orders).
        IG interprets goodTillDate as UTC.
        """
        expiry_seconds = self.config["conditional_orders"]["order_expiry_seconds"]
        now_utc = datetime.now(timezone.utc)
        expiry_dt = now_utc + timedelta(seconds=expiry_seconds)
        return expiry_dt.strftime("%Y/%m/%d %H:%M:%S")

    def build_order_payload(self, epic: str, direction: str, entry_level: float,
                            size: float, stop_distance: float,
                            tp_distance: Optional[float], currency_code: str,
                            expiry_timestamp: str) -> dict:
        """Constructs the IG API working order payload.

        Builds the payload for POST /workingorders/otc with order type STOP.

        Args:
            epic: Market epic identifier.
            direction: "BUY" or "SELL".
            entry_level: The price level at which the stop order triggers.
            size: Deal size.
            stop_distance: Stop-loss distance in points.
            tp_distance: Take-profit (limit) distance in points, or None to omit.
            currency_code: Instrument currency (e.g. "USD").
            expiry_timestamp: ISO 8601 UTC string for goodTillDate.

        Returns:
            dict: IG API order payload ready for submission.
        """
        payload = {
            "epic": epic,
            "direction": direction,
            "type": "STOP",
            "timeInForce": "GOOD_TILL_DATE",
            "goodTillDate": expiry_timestamp,
            "level": entry_level,
            "size": size,
            "stopDistance": stop_distance,
            "currencyCode": currency_code,
            "expiry": "-",
            "forceOpen": True,
            "guaranteedStop": False,
        }

        # Only include limitDistance if take-profit is specified
        if tp_distance is not None:
            payload["limitDistance"] = tp_distance

        return payload

    def poll_orders(self) -> None:
        """Poll IG API for tracked order status.

        Handles fills, expirations, cancellations, and signal reversals.
        One polling cycle — caller is responsible for calling every 60s.

        Logic:
          1. GET /workingorders — on error, retain state, return (Req 3.5)
          2. Build dict of current IG working orders by deal_id
          3. For each tracked order:
             a. NOT in IG list → filled (position exists) or expired/cancelled
             b. IN IG list → check for signal reversal
          4. Clean up tracking for removed orders

        Requirements: 3.3, 3.4, 3.5, 4.7, 8.2, 8.3
        """
        # Step 1: Fetch working orders from IG — retain state on error (Req 3.5)
        try:
            response = self.ig_client.get_working_orders()
        except Exception as e:
            self.log.error(
                f"Poll orders API error: {e} — retaining state, retry next cycle"
            )
            return

        # Step 2: Build lookup of IG working orders by deal_id
        ig_orders = {}
        for order_entry in response.get("workingOrders", []):
            order_data = order_entry.get("workingOrderData", {})
            deal_id = order_data.get("dealId", "")
            if deal_id:
                ig_orders[deal_id] = order_data

        # Step 3: Check each tracked order against IG state
        epics_to_remove = []

        for epic, tracked in list(self.tracked_orders.items()):
            if tracked.deal_id not in ig_orders:
                # Order no longer on IG — either filled or expired/cancelled
                if epic in self.position_manager.positions:
                    # Position exists → treat as FILLED → call _handle_fill
                    self._handle_fill(epic, tracked)
                else:
                    # No position → treat as EXPIRED/CANCELLED (Req 3.3, 8.3)
                    self.log.info(
                        f"Conditional order expired/cancelled: epic={epic}, "
                        f"reason=expired, unfilled_entry_level={tracked.entry_level}"
                    )
                epics_to_remove.append(epic)
            else:
                # Order still pending on IG — check for signal reversal (Req 4.7)
                current_signal = self.active_signals.get(epic)
                if current_signal is not None and current_signal != tracked.direction:
                    # Signal has reversed — cancel the stale order
                    self.cancel_order(epic, "signal_reversal")

        # Step 4: Clean up tracking for filled/expired orders
        for epic in epics_to_remove:
            if epic in self.tracked_orders:
                del self.tracked_orders[epic]
            if epic in self.active_signals:
                del self.active_signals[epic]

    def _handle_fill(self, epic: str, tracked: "TrackedOrder") -> None:
        """Handle a filled working order — register with PositionManager.

        Extracts fill info and hands off to position management.
        Logs fill event per Req 8.2.

        Args:
            epic: The market epic.
            tracked: The TrackedOrder that was filled.
        """
        from datetime import datetime, timezone

        now_utc = datetime.now(timezone.utc)
        elapsed_seconds = (now_utc - tracked.placed_at).total_seconds()

        # Register position with PositionManager (Req 6.1)
        self.position_manager.add_position(
            epic=epic,
            deal_id=tracked.deal_id,
            direction=tracked.direction,
            size=tracked.size,
            entry_price=tracked.entry_level,
            stop=tracked.stop_distance,
            tp=tracked.tp_distance if tracked.tp_distance is not None else 0,
            confidence=tracked.confidence,
            patterns=tracked.patterns,
        )

        # Initialize trailing stop if configured (Req 6.2, 6.3)
        execution_config = self.config.get("execution", {})
        if execution_config.get("use_trailing_stop", False):
            self.trailing_manager.initialize(
                epic=epic,
                deal_id=tracked.deal_id,
                entry_price=tracked.entry_level,
                direction=tracked.direction,
                stop_distance=tracked.stop_distance,
                activation_pct=execution_config.get("trailing_activation_pct", 0.5),
                trailing_distance_pct=execution_config.get("trailing_distance_pct", 0.5),
            )

        # Log fill event per Req 8.2
        self.log.info(
            f"Conditional order filled: epic={epic}, fill_price={tracked.entry_level}, "
            f"deal_id={tracked.deal_id}, elapsed_seconds={elapsed_seconds:.0f}"
        )

    def cancel_order(self, epic: str, reason: str) -> bool:
        """Cancel a tracked working order for the given epic.

        Implements retry logic per Req 4.5/4.6:
        - If cancel_retry_count >= 3: log ERROR, remove from tracking (abandoned), return False
        - On API error: increment cancel_retry_count, log WARNING, return False
        - On success: log INFO (Req 8.3), remove from tracking, return True

        Args:
            epic: The market epic identifier.
            reason: Reason for cancellation (e.g. "signal_reversal", "expired",
                    "kill_switch", "daily_loss_limit").

        Returns:
            True on successful cancellation, False on failure or abandoned.
        """
        if epic not in self.tracked_orders:
            return False

        tracked = self.tracked_orders[epic]

        # Req 4.6: After 3 consecutive failures, log ERROR, skip further retries
        if tracked.cancel_retry_count >= 3:
            self.log.error(
                f"Cancel order abandoned after 3 consecutive failures: epic={epic}, "
                f"reason={reason}, deal_id={tracked.deal_id}"
            )
            # Remove from tracking (abandoned)
            del self.tracked_orders[epic]
            if epic in self.active_signals:
                del self.active_signals[epic]
            return False

        try:
            self.ig_client.delete_working_order(tracked.deal_id)
            # Req 8.3: Log INFO on cancellation with epic, reason, unfilled entry level
            self.log.info(
                f"Conditional order cancelled: epic={epic}, reason={reason}, "
                f"unfilled_entry_level={tracked.entry_level}"
            )
            del self.tracked_orders[epic]
            if epic in self.active_signals:
                del self.active_signals[epic]
            return True
        except Exception as e:
            # Req 4.5: On cancel error, log WARNING and retry next poll cycle
            tracked.cancel_retry_count += 1
            self.log.warning(
                f"Cancel order failed: epic={epic}, reason={reason}, "
                f"error={e}, retry_count={tracked.cancel_retry_count}"
            )
            return False

    def cancel_all_orders(self, reason: str) -> int:
        """Cancel all tracked working orders (kill switch / daily loss limit).

        Iterates all tracked orders and cancels each. Per Req 4.3, 4.4.
        Implements 30-second timeout for kill switch bulk cancellation (Req 4.4).

        Args:
            reason: Reason for bulk cancellation (e.g. "kill_switch", "daily_loss_limit").

        Returns:
            Count of successfully cancelled orders.
        """
        epics_to_cancel = list(self.tracked_orders.keys())
        cancelled_count = 0
        start_time = time.monotonic()

        for epic in epics_to_cancel:
            # Req 4.4: Kill switch must complete within 30 seconds
            if reason == "kill_switch":
                elapsed = time.monotonic() - start_time
                if elapsed > 30.0:
                    self.log.error(
                        f"Kill switch cancellation timeout: elapsed={elapsed:.1f}s > 30s, "
                        f"cancelled={cancelled_count}/{len(epics_to_cancel)} orders"
                    )
                    break

            if self.cancel_order(epic, reason):
                cancelled_count += 1

        return cancelled_count

    def _validate_config(self, config: dict) -> bool:
        """Validate conditional_orders config section.

        Returns True if config is valid, False otherwise (disables feature).
        """
        co_config = config.get("conditional_orders", {})

        # Check all required keys are present
        for key in _REQUIRED_KEYS:
            if key not in co_config:
                self.log.error(
                    f"Conditional orders disabled: missing required parameter '{key}'"
                )
                return False

        # Validate order_expiry_seconds range [60, 86400]
        expiry = co_config["order_expiry_seconds"]
        if not (60 <= expiry <= 86400):
            self.log.error(
                f"Conditional orders disabled: order_expiry_seconds={expiry} "
                f"is outside valid range [60, 86400]"
            )
            return False

        return True

    def _sync_existing_orders(self) -> None:
        """Sync working orders already on IG into internal tracking state.

        Called on startup to pick up orders from previous sessions that
        are still pending. This prevents:
        - Duplicate order placement for the same epic
        - Orphaned orders that never get monitored for fill/expiry
        """
        try:
            response = self.ig_client.get_working_orders()
            ig_orders = response.get("workingOrders", [])

            if not ig_orders:
                self.log.info("📋 Working order sync: no existing orders on IG")
                return

            synced = 0
            for order_entry in ig_orders:
                order_data = order_entry.get("workingOrderData", {})
                market_data = order_entry.get("marketData", {})

                epic = order_data.get("epic", "")
                deal_id = order_data.get("dealId", "")
                direction = order_data.get("direction", "")
                level = order_data.get("level") or order_data.get("orderLevel")
                size = order_data.get("size") or order_data.get("orderSize")
                currency_code = order_data.get("currencyCode", "USD")
                good_till_date = order_data.get("goodTillDate", "")

                if not epic or not deal_id or not direction:
                    continue

                # Parse goodTillDate to get expiry
                expiry_at = datetime.now(timezone.utc) + timedelta(
                    seconds=self.config["conditional_orders"]["order_expiry_seconds"]
                )
                if good_till_date:
                    try:
                        # IG format: "2026/07/15 12:26:27"
                        import pytz
                        expiry_at = datetime.strptime(
                            good_till_date, "%Y/%m/%d %H:%M:%S"
                        ).replace(tzinfo=pytz.timezone("Europe/Rome")).astimezone(timezone.utc)
                    except Exception:
                        pass

                # Create TrackedOrder from IG data
                tracked = TrackedOrder(
                    epic=epic,
                    deal_id=deal_id,
                    direction=direction,
                    entry_level=float(level) if level is not None else 0.0,
                    stop_distance=float(order_data.get("stopDistance") or 0) or 5.0,
                    tp_distance=float(order_data.get("limitDistance") or 0) or None,
                    size=float(size) if size is not None else 1.0,
                    currency_code=currency_code,
                    placed_at=datetime.now(timezone.utc),  # Approximate
                    expiry_at=expiry_at,
                    confidence=0.0,  # Unknown from previous session
                    patterns=[],
                )

                self.tracked_orders[epic] = tracked
                self.active_signals[epic] = direction
                synced += 1

                self.log.info(
                    f"📋 Synced existing order: epic={epic}, direction={direction}, "
                    f"level={level}, deal_id={deal_id}"
                )

            self.log.info(f"📋 Working order sync complete: {synced} orders imported")

        except Exception as e:
            self.log.warning(f"⚠ Working order sync failed (non-fatal): {e}")
