"""Unit tests for ConditionalOrderManager.process_signal — max distance validation.

Tests task 2.3: Reject signal if |entry_level - mid_price| > max_entry_distance_points.
Validates: Requirements 1.6, 8.4
"""

import logging
from unittest.mock import MagicMock

import pytest

from broker.conditional_order_manager import ConditionalOrderManager


def _make_manager(config: dict, log=None) -> ConditionalOrderManager:
    """Create a ConditionalOrderManager with mocked dependencies."""
    ig_client = MagicMock()
    position_manager = MagicMock()
    trailing_manager = MagicMock()
    sr_detector = MagicMock()
    if log is None:
        log = logging.getLogger("test_process_signal")
    return ConditionalOrderManager(
        ig_client=ig_client,
        config=config,
        position_manager=position_manager,
        trailing_manager=trailing_manager,
        sr_detector=sr_detector,
        log=log,
    )


def _valid_config(max_dist=30.0, buffer=2.0):
    """Return a valid conditional_orders config dict."""
    return {
        "conditional_orders": {
            "enabled": True,
            "buffer_points": buffer,
            "order_expiry_seconds": 300,
            "max_entry_distance_points": max_dist,
        }
    }


class TestProcessSignalDisabled:
    """process_signal returns skipped when manager is disabled."""

    def test_returns_skipped_when_disabled(self):
        config = {
            "conditional_orders": {
                "enabled": True,
                "buffer_points": 2.0,
                "order_expiry_seconds": 10,  # Invalid — out of range
                "max_entry_distance_points": 30.0,
            }
        }
        manager = _make_manager(config)
        assert manager.enabled is False

        result = manager.process_signal(
            epic="CS.D.CFEGOLD.CEB.IP", direction="BUY", mid_price=2000.0,
            sr_levels={"resistance": [2010.0], "support": []},
            stop_pts=5.0, tp_pts=10.0, size=1.0,
            currency_code="USD", confidence=0.8, patterns=["engulfing"],
            atr_value=3.0,
        )
        assert result["action"] == "skipped"
        assert result["details"]["reason"] == "disabled"


class TestProcessSignalMaxDistanceRejection:
    """process_signal rejects when |entry_level - mid_price| > max_entry_distance_points."""

    def test_buy_rejected_when_distance_exceeds_max(self):
        """BUY: resistance at 2050, buffer=2, mid=2000 → entry=2052, dist=52 > max=30."""
        config = _valid_config(max_dist=30.0, buffer=2.0)
        manager = _make_manager(config)

        result = manager.process_signal(
            epic="CS.D.CFEGOLD.CEB.IP", direction="BUY", mid_price=2000.0,
            sr_levels={"resistance": [2050.0], "support": []},
            stop_pts=5.0, tp_pts=10.0, size=1.0,
            currency_code="USD", confidence=0.8, patterns=["engulfing"],
            atr_value=3.0,
        )

        assert result["action"] == "rejected"
        assert result["details"]["reason"] == "max_distance_exceeded"
        assert result["details"]["epic"] == "CS.D.CFEGOLD.CEB.IP"
        assert result["details"]["entry_level"] == 2052.0
        assert result["details"]["current_price"] == 2000.0
        assert result["details"]["distance"] == 52.0
        assert result["details"]["max_distance"] == 30.0

    def test_sell_rejected_when_distance_exceeds_max(self):
        """SELL: support at 1940, buffer=2, mid=2000 → entry=1938, dist=62 > max=30."""
        config = _valid_config(max_dist=30.0, buffer=2.0)
        manager = _make_manager(config)

        result = manager.process_signal(
            epic="CS.D.CFEGOLD.CEB.IP", direction="SELL", mid_price=2000.0,
            sr_levels={"resistance": [], "support": [1940.0]},
            stop_pts=5.0, tp_pts=10.0, size=1.0,
            currency_code="USD", confidence=0.7, patterns=["doji"],
            atr_value=2.5,
        )

        assert result["action"] == "rejected"
        assert result["details"]["reason"] == "max_distance_exceeded"
        assert result["details"]["distance"] == 62.0

    def test_buy_accepted_when_distance_within_max(self):
        """BUY: resistance at 2010, buffer=2, mid=2000 → entry=2012, dist=12 < max=30."""
        config = _valid_config(max_dist=30.0, buffer=2.0)
        manager = _make_manager(config)

        result = manager.process_signal(
            epic="CS.D.CFEGOLD.CEB.IP", direction="BUY", mid_price=2000.0,
            sr_levels={"resistance": [2010.0], "support": []},
            stop_pts=5.0, tp_pts=10.0, size=1.0,
            currency_code="USD", confidence=0.8, patterns=["engulfing"],
            atr_value=3.0,
        )

        # Should not be rejected (proceeds to order placement)
        assert result["action"] != "rejected"

    def test_sell_accepted_when_distance_within_max(self):
        """SELL: support at 1990, buffer=2, mid=2000 → entry=1988, dist=12 < max=30."""
        config = _valid_config(max_dist=30.0, buffer=2.0)
        manager = _make_manager(config)

        result = manager.process_signal(
            epic="CS.D.CFEGOLD.CEB.IP", direction="SELL", mid_price=2000.0,
            sr_levels={"resistance": [], "support": [1990.0]},
            stop_pts=5.0, tp_pts=10.0, size=1.0,
            currency_code="USD", confidence=0.7, patterns=["doji"],
            atr_value=2.5,
        )

        assert result["action"] != "rejected"

    def test_boundary_distance_equal_to_max_accepted(self):
        """Exactly at boundary: dist == max → should NOT be rejected (not strictly >)."""
        # resistance at 2028, buffer=2 → entry=2030, dist=30 == max=30
        config = _valid_config(max_dist=30.0, buffer=2.0)
        manager = _make_manager(config)

        result = manager.process_signal(
            epic="CS.D.CFEGOLD.CEB.IP", direction="BUY", mid_price=2000.0,
            sr_levels={"resistance": [2028.0], "support": []},
            stop_pts=5.0, tp_pts=10.0, size=1.0,
            currency_code="USD", confidence=0.8, patterns=["hammer"],
            atr_value=3.0,
        )

        # distance == max → NOT rejected (requirement says "exceeds", i.e. strictly >)
        assert result["action"] != "rejected"

    def test_warning_logged_on_rejection(self, caplog):
        """Verify WARNING log contains epic, entry_level, current_price, distance, max."""
        config = _valid_config(max_dist=30.0, buffer=2.0)
        logger = logging.getLogger("test_warning_log")
        manager = _make_manager(config, log=logger)

        with caplog.at_level(logging.WARNING, logger="test_warning_log"):
            manager.process_signal(
                epic="CS.D.CFEGOLD.CEB.IP", direction="BUY", mid_price=2000.0,
                sr_levels={"resistance": [2050.0], "support": []},
                stop_pts=5.0, tp_pts=10.0, size=1.0,
                currency_code="USD", confidence=0.8, patterns=["engulfing"],
                atr_value=3.0,
            )

        # Check that WARNING was logged with expected fields
        warning_records = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warning_records) >= 1
        msg = warning_records[0].message
        assert "CS.D.CFEGOLD.CEB.IP" in msg
        assert "2052" in msg  # entry_level
        assert "2000" in msg  # current_price
        assert "30" in msg    # max_distance


class TestProcessSignalFallback:
    """process_signal falls back to market order when no S/R level found."""

    def test_buy_fallback_no_resistance(self):
        """BUY with no resistance above mid_price → fallback."""
        config = _valid_config()
        manager = _make_manager(config)

        result = manager.process_signal(
            epic="CS.D.CFEGOLD.CEB.IP", direction="BUY", mid_price=2000.0,
            sr_levels={"resistance": [1990.0], "support": [1980.0]},  # resistance below mid
            stop_pts=5.0, tp_pts=10.0, size=1.0,
            currency_code="USD", confidence=0.8, patterns=["engulfing"],
            atr_value=3.0,
        )

        assert result["action"] == "fallback"
        assert result["details"]["reason"] == "no_resistance_level"

    def test_sell_fallback_no_support(self):
        """SELL with no support below mid_price → fallback."""
        config = _valid_config()
        manager = _make_manager(config)

        result = manager.process_signal(
            epic="CS.D.CFEGOLD.CEB.IP", direction="SELL", mid_price=2000.0,
            sr_levels={"resistance": [2010.0], "support": [2005.0]},  # support above mid
            stop_pts=5.0, tp_pts=10.0, size=1.0,
            currency_code="USD", confidence=0.7, patterns=["doji"],
            atr_value=2.5,
        )

        assert result["action"] == "fallback"
        assert result["details"]["reason"] == "no_support_level"
