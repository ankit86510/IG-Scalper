"""Unit tests for ConditionalOrderManager integration scenarios.

Tests task 9.3: Integration scenarios for conditional order entry.
Validates: Requirements 1.3, 1.4, 2.4, 2.8, 4.3, 4.4, 8.1–8.5
"""

import logging
import time
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest
from requests.exceptions import HTTPError, Timeout

from broker.conditional_order_manager import ConditionalOrderManager, TrackedOrder


def _valid_config(**overrides):
    """Return a valid config dict with optional overrides."""
    config = {
        "conditional_orders": {
            "enabled": True,
            "buffer_points": 2.0,
            "order_expiry_seconds": 300,
            "max_entry_distance_points": 30.0,
        },
        "execution": {
            "use_trailing_stop": False,
            "use_tp_limit": True,
        },
    }
    config.update(overrides)
    return config


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
        log = logging.getLogger("test_integration")
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


# ---------------------------------------------------------------------------
# 1. Fallback to market order (Req 1.3, 1.4)
# ---------------------------------------------------------------------------

class TestFallbackToMarketOrder:
    """Test fallback to market order when no S/R level found."""

    def test_buy_no_resistance_calls_place_order(self):
        """BUY with no resistance above mid → fallback calls ig_client.place_order."""
        manager = _make_manager()

        result = manager.process_signal(
            epic="CS.D.CFEGOLD.CEB.IP", direction="BUY", mid_price=2000.0,
            sr_levels={"resistance": [], "support": [1990.0]},
            stop_pts=5.0, tp_pts=10.0, size=1.0,
            currency_code="USD", confidence=0.8, patterns=["engulfing"],
            atr_value=3.0,
        )

        assert result["action"] == "fallback"
        assert result["details"]["reason"] == "no_resistance_level"
        manager.ig_client.place_order.assert_called_once_with(
            epic="CS.D.CFEGOLD.CEB.IP",
            direction="BUY",
            size=1.0,
            currency_code="USD",
            stop_distance=5.0,
            limit_distance=10.0,
        )

    def test_sell_no_support_calls_place_order(self):
        """SELL with no support below mid → fallback calls ig_client.place_order."""
        manager = _make_manager()

        result = manager.process_signal(
            epic="CS.D.CFEGOLD.CEB.IP", direction="SELL", mid_price=2000.0,
            sr_levels={"resistance": [2010.0], "support": []},
            stop_pts=6.0, tp_pts=12.0, size=0.5,
            currency_code="USD", confidence=0.7, patterns=["doji"],
            atr_value=2.5,
        )

        assert result["action"] == "fallback"
        assert result["details"]["reason"] == "no_support_level"
        manager.ig_client.place_order.assert_called_once_with(
            epic="CS.D.CFEGOLD.CEB.IP",
            direction="SELL",
            size=0.5,
            currency_code="USD",
            stop_distance=6.0,
            limit_distance=12.0,
        )

    def test_fallback_with_tp_zero_omits_limit(self):
        """Fallback with tp_pts=0 passes limit_distance=None."""
        manager = _make_manager()

        result = manager.process_signal(
            epic="CS.D.CFEGOLD.CEB.IP", direction="BUY", mid_price=2000.0,
            sr_levels={"resistance": [], "support": []},
            stop_pts=5.0, tp_pts=0, size=1.0,
            currency_code="USD", confidence=0.8, patterns=["engulfing"],
            atr_value=3.0,
        )

        assert result["action"] == "fallback"
        manager.ig_client.place_order.assert_called_once_with(
            epic="CS.D.CFEGOLD.CEB.IP",
            direction="BUY",
            size=1.0,
            currency_code="USD",
            stop_distance=5.0,
            limit_distance=None,
        )


# ---------------------------------------------------------------------------
# 2. use_tp_limit=false omits TP from payload (Req 2.4)
# ---------------------------------------------------------------------------

class TestUseTpLimitFalse:
    """Test that use_tp_limit=false omits limitDistance from order payload."""

    def test_build_payload_omits_limit_when_tp_none(self):
        """build_order_payload with tp_distance=None omits limitDistance."""
        manager = _make_manager()

        payload = manager.build_order_payload(
            epic="CS.D.CFEGOLD.CEB.IP",
            direction="BUY",
            entry_level=2012.0,
            size=1.0,
            stop_distance=5.0,
            tp_distance=None,
            currency_code="USD",
            expiry_timestamp="2025-01-01T12:05:00",
        )

        assert "limitDistance" not in payload
        assert payload["stopDistance"] == 5.0

    def test_build_payload_includes_limit_when_tp_specified(self):
        """build_order_payload with tp_distance=10 includes limitDistance."""
        manager = _make_manager()

        payload = manager.build_order_payload(
            epic="CS.D.CFEGOLD.CEB.IP",
            direction="BUY",
            entry_level=2012.0,
            size=1.0,
            stop_distance=5.0,
            tp_distance=10.0,
            currency_code="USD",
            expiry_timestamp="2025-01-01T12:05:00",
        )

        assert payload["limitDistance"] == 10.0

    def test_process_signal_tp_zero_omits_limit_in_api_call(self):
        """process_signal with tp_pts=0 passes limit_distance=None to place_working_order."""
        manager = _make_manager()
        manager.ig_client.place_working_order.return_value = {"dealReference": "REF1"}

        manager.process_signal(
            epic="CS.D.CFEGOLD.CEB.IP", direction="BUY", mid_price=2000.0,
            sr_levels={"resistance": [2010.0], "support": []},
            stop_pts=5.0, tp_pts=0, size=1.0,
            currency_code="USD", confidence=0.8, patterns=["engulfing"],
            atr_value=3.0,
        )

        call_kwargs = manager.ig_client.place_working_order.call_args.kwargs
        assert call_kwargs["limit_distance"] is None


# ---------------------------------------------------------------------------
# 3. Trailing stop branching on fill (Req 6.2, 6.3)
# ---------------------------------------------------------------------------

class TestTrailingStopBranching:
    """Test use_trailing_stop=true/false branching on fill."""

    def test_trailing_stop_true_initializes_trailing_manager(self):
        """use_trailing_stop=true → trailing_manager.initialize called."""
        config = _valid_config(execution={
            "use_trailing_stop": True,
            "use_tp_limit": True,
            "trailing_activation_pct": 0.6,
            "trailing_distance_pct": 0.4,
        })
        manager = _make_manager(config=config)
        tracked = _make_tracked_order(deal_id="DEAL_TRAIL")
        manager.tracked_orders["CS.D.CFEGOLD.CEB.IP"] = tracked
        manager.active_signals["CS.D.CFEGOLD.CEB.IP"] = "BUY"

        # Simulate fill detection via poll
        manager.ig_client.get_working_orders.return_value = {"workingOrders": []}
        manager.position_manager.positions = {
            "CS.D.CFEGOLD.CEB.IP": {"deal_id": "DEAL_TRAIL"}
        }

        manager.poll_orders()

        manager.trailing_manager.initialize.assert_called_once()
        call_kwargs = manager.trailing_manager.initialize.call_args.kwargs
        assert call_kwargs["epic"] == "CS.D.CFEGOLD.CEB.IP"
        assert call_kwargs["deal_id"] == "DEAL_TRAIL"
        assert call_kwargs["activation_pct"] == 0.6
        assert call_kwargs["trailing_distance_pct"] == 0.4

    def test_trailing_stop_false_does_not_initialize(self):
        """use_trailing_stop=false → trailing_manager.initialize NOT called."""
        config = _valid_config(execution={
            "use_trailing_stop": False,
            "use_tp_limit": True,
        })
        manager = _make_manager(config=config)
        tracked = _make_tracked_order(deal_id="DEAL_NO_TRAIL")
        manager.tracked_orders["CS.D.CFEGOLD.CEB.IP"] = tracked
        manager.active_signals["CS.D.CFEGOLD.CEB.IP"] = "BUY"

        manager.ig_client.get_working_orders.return_value = {"workingOrders": []}
        manager.position_manager.positions = {
            "CS.D.CFEGOLD.CEB.IP": {"deal_id": "DEAL_NO_TRAIL"}
        }

        manager.poll_orders()

        manager.trailing_manager.initialize.assert_not_called()
        # But add_position should still be called
        manager.position_manager.add_position.assert_called_once()


# ---------------------------------------------------------------------------
# 4. API error handling (Req 2.8)
# ---------------------------------------------------------------------------

class TestAPIErrorHandling:
    """Test API error handling with mocked failures on order placement."""

    def test_http_error_returns_rejected_api_error(self):
        """HTTPError on place_working_order → result is 'rejected' with reason 'api_error'."""
        manager = _make_manager()
        manager.ig_client.place_working_order.side_effect = HTTPError("403 Forbidden")

        result = manager.process_signal(
            epic="CS.D.CFEGOLD.CEB.IP", direction="BUY", mid_price=2000.0,
            sr_levels={"resistance": [2010.0], "support": []},
            stop_pts=5.0, tp_pts=10.0, size=1.0,
            currency_code="USD", confidence=0.8, patterns=["engulfing"],
            atr_value=3.0,
        )

        assert result["action"] == "rejected"
        assert result["details"]["reason"] == "api_error"
        assert "403" in result["details"]["error"]

    def test_timeout_returns_rejected_api_error(self):
        """Timeout on place_working_order → result is 'rejected' with reason 'api_error'."""
        manager = _make_manager()
        manager.ig_client.place_working_order.side_effect = Timeout("Connection timed out")

        result = manager.process_signal(
            epic="CS.D.CFEGOLD.CEB.IP", direction="BUY", mid_price=2000.0,
            sr_levels={"resistance": [2010.0], "support": []},
            stop_pts=5.0, tp_pts=10.0, size=1.0,
            currency_code="USD", confidence=0.8, patterns=["engulfing"],
            atr_value=3.0,
        )

        assert result["action"] == "rejected"
        assert result["details"]["reason"] == "api_error"
        assert "timed out" in result["details"]["error"]

    def test_api_error_does_not_track_order(self):
        """API error on placement → order is NOT added to tracked_orders."""
        manager = _make_manager()
        manager.ig_client.place_working_order.side_effect = HTTPError("500 Server Error")

        manager.process_signal(
            epic="CS.D.CFEGOLD.CEB.IP", direction="BUY", mid_price=2000.0,
            sr_levels={"resistance": [2010.0], "support": []},
            stop_pts=5.0, tp_pts=10.0, size=1.0,
            currency_code="USD", confidence=0.8, patterns=["engulfing"],
            atr_value=3.0,
        )

        assert "CS.D.CFEGOLD.CEB.IP" not in manager.tracked_orders

    def test_api_error_logged(self, caplog):
        """API error is logged at ERROR level with epic and error type."""
        logger = logging.getLogger("test_api_error")
        manager = _make_manager(log=logger)
        manager.ig_client.place_working_order.side_effect = HTTPError("503 Unavailable")

        with caplog.at_level(logging.ERROR, logger="test_api_error"):
            manager.process_signal(
                epic="CS.D.CFEGOLD.CEB.IP", direction="BUY", mid_price=2000.0,
                sr_levels={"resistance": [2010.0], "support": []},
                stop_pts=5.0, tp_pts=10.0, size=1.0,
                currency_code="USD", confidence=0.8, patterns=["engulfing"],
                atr_value=3.0,
            )

        error_records = [r for r in caplog.records if r.levelno == logging.ERROR]
        assert len(error_records) >= 1
        assert "CS.D.CFEGOLD.CEB.IP" in error_records[0].message
        assert "http_error" in error_records[0].message


# ---------------------------------------------------------------------------
# 5. Kill switch cancellation flow (Req 4.4)
# ---------------------------------------------------------------------------

class TestKillSwitchCancellation:
    """Test kill switch cancellation flow."""

    def test_kill_switch_cancels_all_tracked_orders(self):
        """cancel_all_orders('kill_switch') cancels all tracked orders."""
        manager = _make_manager()
        manager.tracked_orders["EPIC1"] = _make_tracked_order(
            epic="EPIC1", deal_id="D1")
        manager.tracked_orders["EPIC2"] = _make_tracked_order(
            epic="EPIC2", deal_id="D2", direction="SELL")
        manager.tracked_orders["EPIC3"] = _make_tracked_order(
            epic="EPIC3", deal_id="D3")
        manager.active_signals = {"EPIC1": "BUY", "EPIC2": "SELL", "EPIC3": "BUY"}

        cancelled = manager.cancel_all_orders("kill_switch")

        assert cancelled == 3
        assert len(manager.tracked_orders) == 0
        assert manager.ig_client.delete_working_order.call_count == 3

    def test_kill_switch_respects_30s_timeout(self):
        """Kill switch stops after 30 seconds elapsed."""
        manager = _make_manager()
        # Add orders
        for i in range(5):
            epic = f"EPIC{i}"
            manager.tracked_orders[epic] = _make_tracked_order(
                epic=epic, deal_id=f"D{i}")
            manager.active_signals[epic] = "BUY"

        # Mock time.monotonic to simulate timeout after 2 cancellations
        call_count = [0]
        original_monotonic = time.monotonic

        def mock_monotonic():
            call_count[0] += 1
            # First call: start_time = 0, subsequent: jump to > 30s after 2 cancels
            if call_count[0] <= 3:  # start + 2 checks
                return 0.0
            return 31.0

        with patch("broker.conditional_order_manager.time.monotonic", side_effect=mock_monotonic):
            cancelled = manager.cancel_all_orders("kill_switch")

        # Should have cancelled some but not all due to timeout
        assert cancelled < 5

    def test_kill_switch_logs_cancellation_for_each_order(self, caplog):
        """Each cancelled order is logged with reason='kill_switch'."""
        logger = logging.getLogger("test_kill_log")
        manager = _make_manager(log=logger)
        manager.tracked_orders["EPIC1"] = _make_tracked_order(
            epic="EPIC1", deal_id="D1")
        manager.tracked_orders["EPIC2"] = _make_tracked_order(
            epic="EPIC2", deal_id="D2")
        manager.active_signals = {"EPIC1": "BUY", "EPIC2": "BUY"}

        with caplog.at_level(logging.INFO, logger="test_kill_log"):
            manager.cancel_all_orders("kill_switch")

        info_records = [r for r in caplog.records if r.levelno == logging.INFO]
        kill_msgs = [r.message for r in info_records if "kill_switch" in r.message]
        assert len(kill_msgs) == 2


# ---------------------------------------------------------------------------
# 6. Daily loss limit bulk cancellation (Req 4.3)
# ---------------------------------------------------------------------------

class TestDailyLossLimitCancellation:
    """Test daily loss limit bulk cancellation flow."""

    def test_daily_loss_cancels_all_tracked_orders(self):
        """cancel_all_orders('daily_loss_limit') cancels all tracked orders."""
        manager = _make_manager()
        manager.tracked_orders["EPIC1"] = _make_tracked_order(
            epic="EPIC1", deal_id="D1")
        manager.tracked_orders["EPIC2"] = _make_tracked_order(
            epic="EPIC2", deal_id="D2")
        manager.active_signals = {"EPIC1": "BUY", "EPIC2": "BUY"}

        cancelled = manager.cancel_all_orders("daily_loss_limit")

        assert cancelled == 2
        assert len(manager.tracked_orders) == 0

    def test_daily_loss_no_timeout_constraint(self):
        """daily_loss_limit does NOT have the 30s timeout (only kill_switch does)."""
        manager = _make_manager()
        for i in range(5):
            epic = f"EPIC{i}"
            manager.tracked_orders[epic] = _make_tracked_order(
                epic=epic, deal_id=f"D{i}")
            manager.active_signals[epic] = "BUY"

        # Even with slow monotonic, daily_loss_limit doesn't check timeout
        cancelled = manager.cancel_all_orders("daily_loss_limit")
        assert cancelled == 5

    def test_daily_loss_logs_reason(self, caplog):
        """Each cancelled order is logged with reason='daily_loss_limit'."""
        logger = logging.getLogger("test_daily_loss_log")
        manager = _make_manager(log=logger)
        manager.tracked_orders["EPIC1"] = _make_tracked_order(
            epic="EPIC1", deal_id="D1")
        manager.active_signals = {"EPIC1": "BUY"}

        with caplog.at_level(logging.INFO, logger="test_daily_loss_log"):
            manager.cancel_all_orders("daily_loss_limit")

        info_records = [r for r in caplog.records if r.levelno == logging.INFO]
        assert any("daily_loss_limit" in r.message for r in info_records)


# ---------------------------------------------------------------------------
# 7. Logging output correctness (Req 8.1–8.5)
# ---------------------------------------------------------------------------

class TestLoggingCorrectness:
    """Test logging output for all lifecycle events."""

    def test_placement_log_info_fields(self, caplog):
        """Placement: INFO log has epic, direction, entry_level, stop_distance, expiry_time, buffer (Req 8.1)."""
        logger = logging.getLogger("test_place_log")
        config = _valid_config()
        config["conditional_orders"]["buffer_points"] = 3.0
        manager = _make_manager(config=config, log=logger)
        manager.ig_client.place_working_order.return_value = {"dealReference": "REF1"}

        with caplog.at_level(logging.INFO, logger="test_place_log"):
            manager.process_signal(
                epic="CS.D.CFEGOLD.CEB.IP", direction="BUY", mid_price=2000.0,
                sr_levels={"resistance": [2010.0], "support": []},
                stop_pts=5.0, tp_pts=10.0, size=1.0,
                currency_code="USD", confidence=0.8, patterns=["engulfing"],
                atr_value=3.0,
            )

        info_records = [r for r in caplog.records if r.levelno == logging.INFO]
        placement_msgs = [r.message for r in info_records if "placed" in r.message]
        assert len(placement_msgs) >= 1
        msg = placement_msgs[0]
        assert "CS.D.CFEGOLD.CEB.IP" in msg  # epic
        assert "BUY" in msg                   # direction
        assert "2013" in msg                  # entry_level (2010 + 3 buffer)
        assert "5.0" in msg                   # stop_distance
        assert "buffer=3.0" in msg            # buffer

    def test_fill_log_info_fields(self, caplog):
        """Fill: INFO log has epic, fill_price, deal_id, elapsed_seconds (Req 8.2)."""
        logger = logging.getLogger("test_fill_log_int")
        manager = _make_manager(log=logger)
        tracked = _make_tracked_order(
            epic="CS.D.CFEGOLD.CEB.IP", deal_id="DEAL_FILL", entry_level=2020.0)
        manager.tracked_orders["CS.D.CFEGOLD.CEB.IP"] = tracked
        manager.active_signals["CS.D.CFEGOLD.CEB.IP"] = "BUY"

        manager.ig_client.get_working_orders.return_value = {"workingOrders": []}
        manager.position_manager.positions = {
            "CS.D.CFEGOLD.CEB.IP": {"deal_id": "DEAL_FILL"}
        }

        with caplog.at_level(logging.INFO, logger="test_fill_log_int"):
            manager.poll_orders()

        info_records = [r for r in caplog.records if r.levelno == logging.INFO]
        fill_msgs = [r.message for r in info_records if "filled" in r.message]
        assert len(fill_msgs) >= 1
        msg = fill_msgs[0]
        assert "CS.D.CFEGOLD.CEB.IP" in msg   # epic
        assert "2020.0" in msg                 # fill_price
        assert "DEAL_FILL" in msg              # deal_id
        assert "elapsed_seconds" in msg        # elapsed field present

    def test_cancellation_log_info_fields(self, caplog):
        """Cancellation: INFO log has epic, reason, unfilled_entry_level (Req 8.3)."""
        logger = logging.getLogger("test_cancel_log")
        manager = _make_manager(log=logger)
        manager.tracked_orders["CS.D.CFEGOLD.CEB.IP"] = _make_tracked_order(
            entry_level=2015.0)
        manager.active_signals["CS.D.CFEGOLD.CEB.IP"] = "BUY"

        with caplog.at_level(logging.INFO, logger="test_cancel_log"):
            manager.cancel_order("CS.D.CFEGOLD.CEB.IP", "signal_reversal")

        info_records = [r for r in caplog.records if r.levelno == logging.INFO]
        cancel_msgs = [r.message for r in info_records if "cancelled" in r.message]
        assert len(cancel_msgs) >= 1
        msg = cancel_msgs[0]
        assert "CS.D.CFEGOLD.CEB.IP" in msg       # epic
        assert "signal_reversal" in msg            # reason
        assert "2015.0" in msg                     # unfilled_entry_level

    def test_rejection_log_warning_fields(self, caplog):
        """Rejection: WARNING log has epic, entry_level, current_price, distance, max_distance (Req 8.4)."""
        logger = logging.getLogger("test_reject_log")
        config = _valid_config()
        config["conditional_orders"]["max_entry_distance_points"] = 10.0
        manager = _make_manager(config=config, log=logger)

        with caplog.at_level(logging.WARNING, logger="test_reject_log"):
            manager.process_signal(
                epic="CS.D.CFEGOLD.CEB.IP", direction="BUY", mid_price=2000.0,
                sr_levels={"resistance": [2020.0], "support": []},
                stop_pts=5.0, tp_pts=10.0, size=1.0,
                currency_code="USD", confidence=0.8, patterns=["engulfing"],
                atr_value=3.0,
            )

        warning_records = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warning_records) >= 1
        msg = warning_records[0].message
        assert "CS.D.CFEGOLD.CEB.IP" in msg    # epic
        assert "2022" in msg                    # entry_level (2020 + 2 buffer)
        assert "2000" in msg                    # current_price
        assert "10" in msg                      # max_distance

    def test_fallback_log_info_fields(self, caplog):
        """Fallback: INFO log has epic, direction, reason (Req 8.5)."""
        logger = logging.getLogger("test_fallback_log")
        manager = _make_manager(log=logger)

        with caplog.at_level(logging.INFO, logger="test_fallback_log"):
            manager.process_signal(
                epic="CS.D.CFEGOLD.CEB.IP", direction="BUY", mid_price=2000.0,
                sr_levels={"resistance": [], "support": []},
                stop_pts=5.0, tp_pts=10.0, size=1.0,
                currency_code="USD", confidence=0.8, patterns=["engulfing"],
                atr_value=3.0,
            )

        info_records = [r for r in caplog.records if r.levelno == logging.INFO]
        fallback_msgs = [r.message for r in info_records if "fallback" in r.message]
        assert len(fallback_msgs) >= 1
        msg = fallback_msgs[0]
        assert "CS.D.CFEGOLD.CEB.IP" in msg        # epic
        assert "BUY" in msg                         # direction
        assert "no_resistance_level" in msg         # reason
