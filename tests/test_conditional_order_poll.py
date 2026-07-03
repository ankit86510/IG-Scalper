"""Unit tests for ConditionalOrderManager.poll_orders method.

Tests task 6.4: Poll working orders, detect fills/expirations/reversals.
Validates: Requirements 3.3, 3.4, 3.5, 4.7, 8.2, 8.3
"""

import logging
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest

from broker.conditional_order_manager import ConditionalOrderManager, TrackedOrder


def _valid_config():
    """Return a valid conditional_orders config dict."""
    return {
        "conditional_orders": {
            "enabled": True,
            "buffer_points": 2.0,
            "order_expiry_seconds": 300,
            "max_entry_distance_points": 30.0,
        },
        "execution": {
            "use_trailing_stop": False,
        },
    }


def _make_manager(config=None, log=None):
    """Create a ConditionalOrderManager with mocked dependencies."""
    if config is None:
        config = _valid_config()
    ig_client = MagicMock()
    position_manager = MagicMock()
    position_manager.positions = {}
    trailing_manager = MagicMock()
    sr_detector = MagicMock()
    if log is None:
        log = logging.getLogger("test_poll_orders")
    return ConditionalOrderManager(
        ig_client=ig_client,
        config=config,
        position_manager=position_manager,
        trailing_manager=trailing_manager,
        sr_detector=sr_detector,
        log=log,
    )


def _make_tracked_order(epic="CS.D.CFEGOLD.CEB.IP", deal_id="DEAL123",
                        direction="BUY", entry_level=2012.0):
    """Create a TrackedOrder for testing."""
    now = datetime.now(timezone.utc)
    return TrackedOrder(
        epic=epic,
        deal_id=deal_id,
        direction=direction,
        entry_level=entry_level,
        stop_distance=5.0,
        tp_distance=10.0,
        size=1.0,
        currency_code="USD",
        placed_at=now - timedelta(seconds=60),
        expiry_at=now + timedelta(seconds=240),
        confidence=0.8,
        patterns=["engulfing"],
    )


class TestPollOrdersAPIError:
    """poll_orders retains state on API error (Req 3.5)."""

    def test_api_error_retains_tracked_orders(self):
        """On API error, tracked_orders and active_signals are unchanged."""
        manager = _make_manager()
        tracked = _make_tracked_order()
        manager.tracked_orders["CS.D.CFEGOLD.CEB.IP"] = tracked
        manager.active_signals["CS.D.CFEGOLD.CEB.IP"] = "BUY"

        manager.ig_client.get_working_orders.side_effect = Exception("Connection timeout")

        manager.poll_orders()

        # State unchanged
        assert "CS.D.CFEGOLD.CEB.IP" in manager.tracked_orders
        assert "CS.D.CFEGOLD.CEB.IP" in manager.active_signals

    def test_api_error_logs_error(self, caplog):
        """On API error, an error is logged."""
        logger = logging.getLogger("test_api_error_log")
        manager = _make_manager(log=logger)
        manager.tracked_orders["EPIC1"] = _make_tracked_order(epic="EPIC1")
        manager.ig_client.get_working_orders.side_effect = RuntimeError("timeout")

        with caplog.at_level(logging.ERROR, logger="test_api_error_log"):
            manager.poll_orders()

        error_records = [r for r in caplog.records if r.levelno == logging.ERROR]
        assert len(error_records) >= 1
        assert "timeout" in error_records[0].message


class TestPollOrdersExpiredCancelled:
    """poll_orders detects expired/cancelled orders and removes from tracking (Req 3.3, 8.3)."""

    def test_order_not_in_ig_no_position_is_expired(self):
        """Order not in IG list and no position → treated as expired, removed."""
        manager = _make_manager()
        tracked = _make_tracked_order(deal_id="DEAL_GONE")
        manager.tracked_orders["CS.D.CFEGOLD.CEB.IP"] = tracked
        manager.active_signals["CS.D.CFEGOLD.CEB.IP"] = "BUY"

        # IG returns empty working orders
        manager.ig_client.get_working_orders.return_value = {"workingOrders": []}
        # No position exists
        manager.position_manager.positions = {}

        manager.poll_orders()

        # Should be removed from tracking
        assert "CS.D.CFEGOLD.CEB.IP" not in manager.tracked_orders
        assert "CS.D.CFEGOLD.CEB.IP" not in manager.active_signals

    def test_expired_order_logged(self, caplog):
        """Expired order logs INFO with epic, reason, unfilled_entry_level (Req 8.3)."""
        logger = logging.getLogger("test_expired_log")
        manager = _make_manager(log=logger)
        tracked = _make_tracked_order(entry_level=2015.5)
        manager.tracked_orders["CS.D.CFEGOLD.CEB.IP"] = tracked
        manager.active_signals["CS.D.CFEGOLD.CEB.IP"] = "BUY"

        manager.ig_client.get_working_orders.return_value = {"workingOrders": []}
        manager.position_manager.positions = {}

        with caplog.at_level(logging.INFO, logger="test_expired_log"):
            manager.poll_orders()

        info_records = [r for r in caplog.records if r.levelno == logging.INFO]
        assert len(info_records) >= 1
        msg = info_records[0].message
        assert "CS.D.CFEGOLD.CEB.IP" in msg
        assert "expired" in msg
        assert "2015.5" in msg


class TestPollOrdersFilled:
    """poll_orders detects filled orders and calls _handle_fill (Req 8.2)."""

    def test_order_not_in_ig_with_position_is_filled(self):
        """Order not in IG list but position exists → treated as filled."""
        manager = _make_manager()
        tracked = _make_tracked_order(deal_id="DEAL_FILLED")
        manager.tracked_orders["CS.D.CFEGOLD.CEB.IP"] = tracked
        manager.active_signals["CS.D.CFEGOLD.CEB.IP"] = "BUY"

        # IG returns empty working orders
        manager.ig_client.get_working_orders.return_value = {"workingOrders": []}
        # Position DOES exist for this epic
        manager.position_manager.positions = {"CS.D.CFEGOLD.CEB.IP": {"deal_id": "DEAL_FILLED"}}

        manager.poll_orders()

        # add_position should have been called
        manager.position_manager.add_position.assert_called_once()
        call_kwargs = manager.position_manager.add_position.call_args
        assert call_kwargs.kwargs["epic"] == "CS.D.CFEGOLD.CEB.IP"
        assert call_kwargs.kwargs["deal_id"] == "DEAL_FILLED"
        assert call_kwargs.kwargs["direction"] == "BUY"
        assert call_kwargs.kwargs["entry_price"] == 2012.0
        assert call_kwargs.kwargs["stop"] == 5.0
        assert call_kwargs.kwargs["tp"] == 10.0

        # Order should be removed from tracking after fill
        assert "CS.D.CFEGOLD.CEB.IP" not in manager.tracked_orders
        assert "CS.D.CFEGOLD.CEB.IP" not in manager.active_signals

    def test_fill_with_trailing_stop_enabled(self):
        """When use_trailing_stop=True, trailing_manager.initialize is called on fill."""
        config = _valid_config()
        config["execution"] = {
            "use_trailing_stop": True,
            "trailing_activation_pct": 0.6,
            "trailing_distance_pct": 0.4,
        }
        manager = _make_manager(config=config)
        tracked = _make_tracked_order(deal_id="DEAL_TRAIL")
        manager.tracked_orders["CS.D.CFEGOLD.CEB.IP"] = tracked
        manager.active_signals["CS.D.CFEGOLD.CEB.IP"] = "BUY"

        manager.ig_client.get_working_orders.return_value = {"workingOrders": []}
        manager.position_manager.positions = {"CS.D.CFEGOLD.CEB.IP": {"deal_id": "DEAL_TRAIL"}}

        manager.poll_orders()

        # trailing_manager.initialize should have been called
        manager.trailing_manager.initialize.assert_called_once()
        call_kwargs = manager.trailing_manager.initialize.call_args.kwargs
        assert call_kwargs["epic"] == "CS.D.CFEGOLD.CEB.IP"
        assert call_kwargs["deal_id"] == "DEAL_TRAIL"
        assert call_kwargs["fill_price"] == 2012.0
        assert call_kwargs["direction"] == "BUY"
        assert call_kwargs["activation_pct"] == 0.6
        assert call_kwargs["trailing_distance_pct"] == 0.4

    def test_fill_logged(self, caplog):
        """Fill event logs INFO with epic, fill_price, deal_id, elapsed_seconds (Req 8.2)."""
        logger = logging.getLogger("test_fill_log")
        manager = _make_manager(log=logger)
        tracked = _make_tracked_order(deal_id="DEAL_LOG", entry_level=2020.0)
        manager.tracked_orders["CS.D.CFEGOLD.CEB.IP"] = tracked
        manager.active_signals["CS.D.CFEGOLD.CEB.IP"] = "BUY"

        manager.ig_client.get_working_orders.return_value = {"workingOrders": []}
        manager.position_manager.positions = {"CS.D.CFEGOLD.CEB.IP": {"deal_id": "DEAL_LOG"}}

        with caplog.at_level(logging.INFO, logger="test_fill_log"):
            manager.poll_orders()

        info_records = [r for r in caplog.records if r.levelno == logging.INFO]
        assert len(info_records) >= 1
        msg = info_records[0].message
        assert "filled" in msg
        assert "CS.D.CFEGOLD.CEB.IP" in msg
        assert "DEAL_LOG" in msg
        assert "2020.0" in msg


class TestPollOrdersSignalReversal:
    """poll_orders detects signal reversals and cancels orders (Req 4.7)."""

    def test_signal_reversal_cancels_order(self):
        """Order still on IG but active_signal direction differs → cancel."""
        manager = _make_manager()
        tracked = _make_tracked_order(direction="BUY", deal_id="DEAL_REV")
        manager.tracked_orders["CS.D.CFEGOLD.CEB.IP"] = tracked
        # Signal has reversed to SELL
        manager.active_signals["CS.D.CFEGOLD.CEB.IP"] = "SELL"

        # IG still has the order
        manager.ig_client.get_working_orders.return_value = {
            "workingOrders": [
                {"workingOrderData": {"dealId": "DEAL_REV", "epic": "CS.D.CFEGOLD.CEB.IP", "direction": "BUY"}}
            ]
        }

        manager.poll_orders()

        # delete_working_order should have been called
        manager.ig_client.delete_working_order.assert_called_once_with("DEAL_REV")

    def test_same_direction_no_cancel(self):
        """Order still on IG and active_signal matches → no cancel."""
        manager = _make_manager()
        tracked = _make_tracked_order(direction="BUY", deal_id="DEAL_KEEP")
        manager.tracked_orders["CS.D.CFEGOLD.CEB.IP"] = tracked
        manager.active_signals["CS.D.CFEGOLD.CEB.IP"] = "BUY"

        manager.ig_client.get_working_orders.return_value = {
            "workingOrders": [
                {"workingOrderData": {"dealId": "DEAL_KEEP", "epic": "CS.D.CFEGOLD.CEB.IP", "direction": "BUY"}}
            ]
        }

        manager.poll_orders()

        # No cancellation
        manager.ig_client.delete_working_order.assert_not_called()
        # Order still tracked
        assert "CS.D.CFEGOLD.CEB.IP" in manager.tracked_orders


class TestPollOrdersMultipleOrders:
    """poll_orders handles multiple tracked orders correctly."""

    def test_mixed_outcomes(self):
        """Multiple orders: one filled, one expired, one still pending."""
        manager = _make_manager()

        # Order 1: filled (not in IG, position exists)
        tracked1 = _make_tracked_order(epic="EPIC_FILL", deal_id="D1", direction="BUY")
        manager.tracked_orders["EPIC_FILL"] = tracked1
        manager.active_signals["EPIC_FILL"] = "BUY"

        # Order 2: expired (not in IG, no position)
        tracked2 = _make_tracked_order(epic="EPIC_EXP", deal_id="D2", direction="SELL")
        manager.tracked_orders["EPIC_EXP"] = tracked2
        manager.active_signals["EPIC_EXP"] = "SELL"

        # Order 3: still pending (in IG, same direction)
        tracked3 = _make_tracked_order(epic="EPIC_PEND", deal_id="D3", direction="BUY")
        manager.tracked_orders["EPIC_PEND"] = tracked3
        manager.active_signals["EPIC_PEND"] = "BUY"

        # IG only has order 3
        manager.ig_client.get_working_orders.return_value = {
            "workingOrders": [
                {"workingOrderData": {"dealId": "D3", "epic": "EPIC_PEND", "direction": "BUY"}}
            ]
        }
        # Position exists for EPIC_FILL only
        manager.position_manager.positions = {"EPIC_FILL": {"deal_id": "D1"}}

        manager.poll_orders()

        # EPIC_FILL: filled → removed from tracking, add_position called
        assert "EPIC_FILL" not in manager.tracked_orders
        manager.position_manager.add_position.assert_called_once()

        # EPIC_EXP: expired → removed from tracking
        assert "EPIC_EXP" not in manager.tracked_orders

        # EPIC_PEND: still pending → still tracked
        assert "EPIC_PEND" in manager.tracked_orders

    def test_no_tracked_orders_no_error(self):
        """Polling with no tracked orders does nothing gracefully."""
        manager = _make_manager()
        manager.ig_client.get_working_orders.return_value = {"workingOrders": []}

        # Should not raise
        manager.poll_orders()

        assert len(manager.tracked_orders) == 0
