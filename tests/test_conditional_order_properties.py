"""Property-based tests for ConditionalOrderManager using Hypothesis.

# Feature: conditional-order-entry, Property 11: Configuration Validation
# Feature: conditional-order-entry, Property 1: Entry Level Calculation
# Feature: conditional-order-entry, Property 2: Max Distance Rejection
# Feature: conditional-order-entry, Property 4: Expiry Timestamp Calculation
# Feature: conditional-order-entry, Property 7: One-Order-Per-Epic Invariant

Validates correctness properties defined in the design document.
"""

import logging
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

from hypothesis import assume, given, settings
from hypothesis.stateful import RuleBasedStateMachine, invariant, rule
from hypothesis.strategies import (
    booleans,
    composite,
    datetimes,
    dictionaries,
    fixed_dictionaries,
    floats,
    integers,
    just,
    lists,
    none,
    one_of,
    sampled_from,
    text,
)

from broker.conditional_order_manager import ConditionalOrderManager, TrackedOrder, _REQUIRED_KEYS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_manager(config: dict) -> ConditionalOrderManager:
    """Create a ConditionalOrderManager with mocked dependencies and given config."""
    ig_client = MagicMock()
    position_manager = MagicMock()
    trailing_manager = MagicMock()
    sr_detector = MagicMock()
    log = logging.getLogger("test_config_validation")
    return ConditionalOrderManager(
        ig_client=ig_client,
        config=config,
        position_manager=position_manager,
        trailing_manager=trailing_manager,
        sr_detector=sr_detector,
        log=log,
    )


# ---------------------------------------------------------------------------
# Strategies (generators)
# ---------------------------------------------------------------------------

@composite
def valid_conditional_order_configs(draw):
    """Generate valid conditional_orders config dicts.

    All required keys present and order_expiry_seconds within [60, 86400].
    """
    enabled = draw(booleans())
    buffer_points = draw(floats(min_value=0.5, max_value=50.0, allow_nan=False, allow_infinity=False))
    order_expiry_seconds = draw(integers(min_value=60, max_value=86400))
    max_entry_distance_points = draw(floats(min_value=1.0, max_value=200.0, allow_nan=False, allow_infinity=False))

    return {
        "conditional_orders": {
            "enabled": enabled,
            "buffer_points": buffer_points,
            "order_expiry_seconds": order_expiry_seconds,
            "max_entry_distance_points": max_entry_distance_points,
        }
    }


@composite
def configs_missing_required_key(draw):
    """Generate config dicts missing at least one required key.

    Starts with a valid config and removes one or more required keys.
    """
    # Build a full valid config
    enabled = draw(booleans())
    buffer_points = draw(floats(min_value=0.5, max_value=50.0, allow_nan=False, allow_infinity=False))
    order_expiry_seconds = draw(integers(min_value=60, max_value=86400))
    max_entry_distance_points = draw(floats(min_value=1.0, max_value=200.0, allow_nan=False, allow_infinity=False))

    co_config = {
        "enabled": enabled,
        "buffer_points": buffer_points,
        "order_expiry_seconds": order_expiry_seconds,
        "max_entry_distance_points": max_entry_distance_points,
    }

    # Remove at least one required key
    key_to_remove = draw(sampled_from(list(_REQUIRED_KEYS)))
    del co_config[key_to_remove]

    return {"conditional_orders": co_config}


@composite
def configs_with_out_of_range_expiry(draw):
    """Generate config dicts with order_expiry_seconds outside [60, 86400].

    All required keys are present, but expiry is invalid.
    """
    enabled = draw(booleans())
    buffer_points = draw(floats(min_value=0.5, max_value=50.0, allow_nan=False, allow_infinity=False))
    max_entry_distance_points = draw(floats(min_value=1.0, max_value=200.0, allow_nan=False, allow_infinity=False))

    # Generate expiry outside valid range: either < 60 or > 86400
    expiry = draw(one_of(
        integers(min_value=-1000, max_value=59),
        integers(min_value=86401, max_value=200000),
    ))

    return {
        "conditional_orders": {
            "enabled": enabled,
            "buffer_points": buffer_points,
            "order_expiry_seconds": expiry,
            "max_entry_distance_points": max_entry_distance_points,
        }
    }


# ---------------------------------------------------------------------------
# Property 11: Configuration Validation
# Validates: Requirements 3.6, 7.3
# ---------------------------------------------------------------------------

class TestConfigurationValidation:
    """Property 11: Configuration Validation.

    For any configuration where `order_expiry_seconds` is outside [60, 86400],
    or any required parameter (`enabled`, `buffer_points`, `order_expiry_seconds`,
    `max_entry_distance_points`) is missing, the system SHALL reject the
    configuration and fall back to market order execution (manager.enabled == False).

    **Validates: Requirements 3.6, 7.3**
    """

    @given(config=configs_missing_required_key())
    @settings(max_examples=100)
    def test_missing_required_key_disables_manager(self, config: dict):
        """Missing any required key → manager.enabled == False."""
        manager = _make_manager(config)
        assert manager.enabled is False, (
            f"Manager should be disabled when a required key is missing. "
            f"Config keys present: {list(config.get('conditional_orders', {}).keys())}"
        )

    @given(config=configs_with_out_of_range_expiry())
    @settings(max_examples=100)
    def test_out_of_range_expiry_disables_manager(self, config: dict):
        """order_expiry_seconds outside [60, 86400] → manager.enabled == False."""
        manager = _make_manager(config)
        expiry = config["conditional_orders"]["order_expiry_seconds"]
        assert manager.enabled is False, (
            f"Manager should be disabled when order_expiry_seconds={expiry} "
            f"is outside [60, 86400]"
        )

    @given(config=valid_conditional_order_configs())
    @settings(max_examples=100)
    def test_valid_config_enables_manager(self, config: dict):
        """Valid config with all keys present and expiry in range → manager.enabled == True."""
        manager = _make_manager(config)
        assert manager.enabled is True, (
            f"Manager should be enabled with valid config: "
            f"{config.get('conditional_orders', {})}"
        )


# ---------------------------------------------------------------------------
# Strategies for Property 1: Entry Level Calculation
# ---------------------------------------------------------------------------

@composite
def entry_level_buy_scenario(draw):
    """Generate a BUY scenario with at least one resistance level above mid_price.

    Returns (mid_price, sr_levels, buffer, expected_nearest_resistance).
    """
    mid_price = draw(floats(min_value=10.0, max_value=10000.0, allow_nan=False, allow_infinity=False))
    buffer = draw(floats(min_value=0.5, max_value=50.0, allow_nan=False, allow_infinity=False))

    # Generate resistance levels above mid_price
    resistance_above = draw(lists(
        floats(min_value=mid_price + 0.01, max_value=mid_price + 500.0,
               allow_nan=False, allow_infinity=False),
        min_size=1, max_size=10,
    ))
    # Optionally add some support levels below mid_price (irrelevant for BUY)
    support_below = draw(lists(
        floats(min_value=max(0.01, mid_price - 500.0), max_value=mid_price - 0.01,
               allow_nan=False, allow_infinity=False),
        min_size=0, max_size=5,
    ))

    sr_levels = {
        "resistance": resistance_above,
        "support": support_below,
    }

    # The nearest resistance is the one with smallest distance from mid_price
    nearest_resistance = min(resistance_above, key=lambda lvl: abs(lvl - mid_price))

    return mid_price, sr_levels, buffer, nearest_resistance


@composite
def entry_level_sell_scenario(draw):
    """Generate a SELL scenario with at least one support level below mid_price.

    Returns (mid_price, sr_levels, buffer, expected_nearest_support).
    """
    mid_price = draw(floats(min_value=10.0, max_value=10000.0, allow_nan=False, allow_infinity=False))
    buffer = draw(floats(min_value=0.5, max_value=50.0, allow_nan=False, allow_infinity=False))

    # Generate support levels below mid_price
    support_below = draw(lists(
        floats(min_value=max(0.01, mid_price - 500.0), max_value=mid_price - 0.01,
               allow_nan=False, allow_infinity=False),
        min_size=1, max_size=10,
    ))
    # Optionally add some resistance levels above mid_price (irrelevant for SELL)
    resistance_above = draw(lists(
        floats(min_value=mid_price + 0.01, max_value=mid_price + 500.0,
               allow_nan=False, allow_infinity=False),
        min_size=0, max_size=5,
    ))

    sr_levels = {
        "resistance": resistance_above,
        "support": support_below,
    }

    # The nearest support is the one with smallest distance from mid_price
    nearest_support = min(support_below, key=lambda lvl: abs(lvl - mid_price))

    return mid_price, sr_levels, buffer, nearest_support


@composite
def entry_level_no_resistance_scenario(draw):
    """Generate a BUY scenario with no resistance levels above mid_price.

    Returns (mid_price, sr_levels, buffer).
    """
    mid_price = draw(floats(min_value=10.0, max_value=10000.0, allow_nan=False, allow_infinity=False))
    buffer = draw(floats(min_value=0.5, max_value=50.0, allow_nan=False, allow_infinity=False))

    # All resistance levels are below or at mid_price (no valid candidates for BUY)
    resistance_below = draw(lists(
        floats(min_value=max(0.01, mid_price - 500.0), max_value=mid_price,
               allow_nan=False, allow_infinity=False),
        min_size=0, max_size=5,
    ))

    sr_levels = {
        "resistance": resistance_below,
        "support": [],
    }

    return mid_price, sr_levels, buffer


@composite
def entry_level_no_support_scenario(draw):
    """Generate a SELL scenario with no support levels below mid_price.

    Returns (mid_price, sr_levels, buffer).
    """
    mid_price = draw(floats(min_value=10.0, max_value=10000.0, allow_nan=False, allow_infinity=False))
    buffer = draw(floats(min_value=0.5, max_value=50.0, allow_nan=False, allow_infinity=False))

    # All support levels are above or at mid_price (no valid candidates for SELL)
    support_above = draw(lists(
        floats(min_value=mid_price, max_value=mid_price + 500.0,
               allow_nan=False, allow_infinity=False),
        min_size=0, max_size=5,
    ))

    sr_levels = {
        "resistance": [],
        "support": support_above,
    }

    return mid_price, sr_levels, buffer


# ---------------------------------------------------------------------------
# Property 1: Entry Level Calculation
# Feature: conditional-order-entry, Property 1: Entry Level Calculation
# Validates: Requirements 1.1, 1.2, 1.7
# ---------------------------------------------------------------------------

class TestEntryLevelCalculation:
    """Property 1: Entry Level Calculation.

    For any signal direction (BUY or SELL), list of S/R levels, mid-price,
    and buffer value within [0.5, 50.0], the calculated entry level SHALL equal
    the nearest resistance above mid-price plus buffer (for BUY) or the nearest
    support below mid-price minus buffer (for SELL), where "nearest" means the
    level with the smallest absolute distance from mid-price among valid candidates.

    **Validates: Requirements 1.1, 1.2, 1.7**
    """

    @given(scenario=entry_level_buy_scenario())
    @settings(max_examples=100)
    def test_buy_entry_equals_nearest_resistance_plus_buffer(self, scenario):
        """BUY: entry == nearest_resistance_above_mid + buffer."""
        mid_price, sr_levels, buffer, nearest_resistance = scenario

        config = {
            "conditional_orders": {
                "enabled": True,
                "buffer_points": buffer,
                "order_expiry_seconds": 300,
                "max_entry_distance_points": 1000.0,
            }
        }
        manager = _make_manager(config)

        entry = manager.calculate_entry_level("BUY", mid_price, sr_levels)

        expected = nearest_resistance + buffer
        assert entry is not None, "Entry should not be None when resistance levels exist above mid_price"
        assert abs(entry - expected) < 1e-9, (
            f"BUY entry should be nearest_resistance + buffer. "
            f"Got {entry}, expected {expected} "
            f"(nearest_resistance={nearest_resistance}, buffer={buffer})"
        )

    @given(scenario=entry_level_sell_scenario())
    @settings(max_examples=100)
    def test_sell_entry_equals_nearest_support_minus_buffer(self, scenario):
        """SELL: entry == nearest_support_below_mid - buffer."""
        mid_price, sr_levels, buffer, nearest_support = scenario

        config = {
            "conditional_orders": {
                "enabled": True,
                "buffer_points": buffer,
                "order_expiry_seconds": 300,
                "max_entry_distance_points": 1000.0,
            }
        }
        manager = _make_manager(config)

        entry = manager.calculate_entry_level("SELL", mid_price, sr_levels)

        expected = nearest_support - buffer
        assert entry is not None, "Entry should not be None when support levels exist below mid_price"
        assert abs(entry - expected) < 1e-9, (
            f"SELL entry should be nearest_support - buffer. "
            f"Got {entry}, expected {expected} "
            f"(nearest_support={nearest_support}, buffer={buffer})"
        )

    @given(scenario=entry_level_no_resistance_scenario())
    @settings(max_examples=100)
    def test_buy_returns_none_when_no_resistance_above(self, scenario):
        """BUY with no resistance above mid_price → returns None."""
        mid_price, sr_levels, buffer = scenario

        config = {
            "conditional_orders": {
                "enabled": True,
                "buffer_points": buffer,
                "order_expiry_seconds": 300,
                "max_entry_distance_points": 1000.0,
            }
        }
        manager = _make_manager(config)

        entry = manager.calculate_entry_level("BUY", mid_price, sr_levels)

        assert entry is None, (
            f"BUY should return None when no resistance levels above mid_price. "
            f"Got {entry}, mid_price={mid_price}, resistance={sr_levels['resistance']}"
        )

    @given(scenario=entry_level_no_support_scenario())
    @settings(max_examples=100)
    def test_sell_returns_none_when_no_support_below(self, scenario):
        """SELL with no support below mid_price → returns None."""
        mid_price, sr_levels, buffer = scenario

        config = {
            "conditional_orders": {
                "enabled": True,
                "buffer_points": buffer,
                "order_expiry_seconds": 300,
                "max_entry_distance_points": 1000.0,
            }
        }
        manager = _make_manager(config)

        entry = manager.calculate_entry_level("SELL", mid_price, sr_levels)

        assert entry is None, (
            f"SELL should return None when no support levels below mid_price. "
            f"Got {entry}, mid_price={mid_price}, support={sr_levels['support']}"
        )


# ---------------------------------------------------------------------------
# Strategies for Property 2: Max Distance Rejection
# ---------------------------------------------------------------------------

@composite
def max_distance_exceeded_scenario(draw):
    """Generate a scenario where |entry_level - mid_price| > max_distance.

    Returns (mid_price, direction, sr_levels, buffer, max_distance).
    The S/R levels are crafted so that the calculated entry_level produces
    a distance exceeding max_distance.
    """
    mid_price = draw(floats(min_value=100.0, max_value=5000.0, allow_nan=False, allow_infinity=False))
    max_distance = draw(floats(min_value=1.0, max_value=200.0, allow_nan=False, allow_infinity=False))
    buffer = draw(floats(min_value=0.5, max_value=50.0, allow_nan=False, allow_infinity=False))
    direction = draw(sampled_from(["BUY", "SELL"]))

    # We need |entry_level - mid_price| > max_distance
    # For BUY: entry = nearest_resistance + buffer, distance = entry - mid_price
    #   So we need nearest_resistance + buffer - mid_price > max_distance
    #   nearest_resistance > mid_price + max_distance - buffer
    # For SELL: entry = nearest_support - buffer, distance = mid_price - entry
    #   So we need mid_price - (nearest_support - buffer) > max_distance
    #   nearest_support < mid_price - max_distance - buffer

    # Generate an extra offset to guarantee exceeding
    extra_offset = draw(floats(min_value=0.01, max_value=100.0, allow_nan=False, allow_infinity=False))

    if direction == "BUY":
        # nearest_resistance must be > mid_price + max_distance - buffer + extra_offset
        # but also > mid_price (requirement for valid BUY candidate)
        sr_level = mid_price + max_distance - buffer + extra_offset
        # Ensure sr_level is above mid_price (it will be since max_distance > 0 and extra_offset > 0)
        assume(sr_level > mid_price)
        sr_levels = {"resistance": [sr_level], "support": []}
    else:
        # nearest_support must be < mid_price - max_distance + buffer - extra_offset
        # but also < mid_price (requirement for valid SELL candidate)
        sr_level = mid_price - max_distance + buffer - extra_offset
        # Ensure sr_level is below mid_price
        assume(sr_level < mid_price)
        # Ensure sr_level is positive
        assume(sr_level > 0)
        sr_levels = {"resistance": [], "support": [sr_level]}

    return mid_price, direction, sr_levels, buffer, max_distance


@composite
def max_distance_within_scenario(draw):
    """Generate a scenario where |entry_level - mid_price| <= max_distance.

    Returns (mid_price, direction, sr_levels, buffer, max_distance).
    The S/R levels are crafted so that the calculated entry_level produces
    a distance within max_distance (inclusive).
    """
    mid_price = draw(floats(min_value=100.0, max_value=5000.0, allow_nan=False, allow_infinity=False))
    max_distance = draw(floats(min_value=10.0, max_value=200.0, allow_nan=False, allow_infinity=False))
    buffer = draw(floats(min_value=0.5, max_value=5.0, allow_nan=False, allow_infinity=False))
    direction = draw(sampled_from(["BUY", "SELL"]))

    # We need |entry_level - mid_price| <= max_distance
    # Generate a fraction of max_distance for the total distance
    # distance_fraction in (0, 1] so distance = fraction * max_distance <= max_distance
    distance_fraction = draw(floats(min_value=0.01, max_value=0.99, allow_nan=False, allow_infinity=False))
    target_distance = distance_fraction * max_distance

    if direction == "BUY":
        # entry = nearest_resistance + buffer
        # distance = entry - mid_price = nearest_resistance + buffer - mid_price = target_distance
        # nearest_resistance = mid_price + target_distance - buffer
        sr_level = mid_price + target_distance - buffer
        # Ensure sr_level > mid_price (valid resistance candidate)
        assume(sr_level > mid_price)
        sr_levels = {"resistance": [sr_level], "support": []}
    else:
        # entry = nearest_support - buffer
        # distance = mid_price - entry = mid_price - nearest_support + buffer = target_distance
        # nearest_support = mid_price - target_distance + buffer
        sr_level = mid_price - target_distance + buffer
        # Ensure sr_level < mid_price (valid support candidate)
        assume(sr_level < mid_price)
        # Ensure sr_level is positive
        assume(sr_level > 0)
        sr_levels = {"resistance": [], "support": [sr_level]}

    return mid_price, direction, sr_levels, buffer, max_distance


# ---------------------------------------------------------------------------
# Property 2: Max Distance Rejection
# Feature: conditional-order-entry, Property 2: Max Distance Rejection
# Validates: Requirements 1.6
# ---------------------------------------------------------------------------

class TestMaxDistanceRejection:
    """Property 2: Max Distance Rejection.

    For any calculated entry level and current mid-price, if the absolute distance
    |entry_level - mid_price| exceeds the configured max_entry_distance_points,
    then the signal SHALL be rejected and no working order SHALL be placed.

    **Validates: Requirements 1.6**
    """

    @given(scenario=max_distance_exceeded_scenario())
    @settings(max_examples=100)
    def test_distance_exceeds_max_rejects_signal(self, scenario):
        """Distance > max_entry_distance_points → action == 'rejected'."""
        mid_price, direction, sr_levels, buffer, max_distance = scenario

        config = {
            "conditional_orders": {
                "enabled": True,
                "buffer_points": buffer,
                "order_expiry_seconds": 300,
                "max_entry_distance_points": max_distance,
            }
        }
        manager = _make_manager(config)

        result = manager.process_signal(
            epic="TEST.EPIC",
            direction=direction,
            mid_price=mid_price,
            sr_levels=sr_levels,
            stop_pts=10.0,
            tp_pts=20.0,
            size=1.0,
            currency_code="USD",
            confidence=0.8,
            patterns=["test_pattern"],
            atr_value=5.0,
        )

        assert result["action"] == "rejected", (
            f"Signal should be rejected when distance > max. "
            f"Got action='{result['action']}', "
            f"mid_price={mid_price}, direction={direction}, "
            f"max_distance={max_distance}, buffer={buffer}, "
            f"sr_levels={sr_levels}"
        )
        assert result["details"]["reason"] == "max_distance_exceeded"

    @given(scenario=max_distance_within_scenario())
    @settings(max_examples=100)
    def test_distance_within_max_does_not_reject(self, scenario):
        """Distance <= max_entry_distance_points → action != 'rejected'."""
        mid_price, direction, sr_levels, buffer, max_distance = scenario

        config = {
            "conditional_orders": {
                "enabled": True,
                "buffer_points": buffer,
                "order_expiry_seconds": 300,
                "max_entry_distance_points": max_distance,
            }
        }
        manager = _make_manager(config)

        result = manager.process_signal(
            epic="TEST.EPIC",
            direction=direction,
            mid_price=mid_price,
            sr_levels=sr_levels,
            stop_pts=10.0,
            tp_pts=20.0,
            size=1.0,
            currency_code="USD",
            confidence=0.8,
            patterns=["test_pattern"],
            atr_value=5.0,
        )

        assert result["action"] != "rejected", (
            f"Signal should NOT be rejected when distance <= max. "
            f"Got action='{result['action']}', "
            f"mid_price={mid_price}, direction={direction}, "
            f"max_distance={max_distance}, buffer={buffer}, "
            f"sr_levels={sr_levels}"
        )


# ---------------------------------------------------------------------------
# Strategies for Property 3: Stop and Take-Profit Calculation
# ---------------------------------------------------------------------------

@composite
def stop_tp_scenario(draw):
    """Generate arbitrary parameters for stop/TP calculation.

    Returns (atr_value, stop_multiplier, min_stop_pts, market_min_stop, rr_take, use_tp_limit).
    """
    atr_value = draw(floats(min_value=0.1, max_value=500.0, allow_nan=False, allow_infinity=False))
    stop_multiplier = draw(floats(min_value=0.1, max_value=10.0, allow_nan=False, allow_infinity=False))
    min_stop_pts = draw(floats(min_value=0.1, max_value=200.0, allow_nan=False, allow_infinity=False))
    market_min_stop = draw(floats(min_value=0.0, max_value=200.0, allow_nan=False, allow_infinity=False))
    rr_take = draw(floats(min_value=0.1, max_value=10.0, allow_nan=False, allow_infinity=False))
    use_tp_limit = draw(booleans())

    return atr_value, stop_multiplier, min_stop_pts, market_min_stop, rr_take, use_tp_limit


# ---------------------------------------------------------------------------
# Property 3: Stop and Take-Profit Calculation with Market Rules
# Feature: conditional-order-entry, Property 3: Stop and Take-Profit Calculation
# Validates: Requirements 2.2, 2.3, 2.7, 2.9
# ---------------------------------------------------------------------------

class TestStopTPCalculation:
    """Property 3: Stop and Take-Profit Calculation with Market Rules.

    For any ATR value, stop_multiplier, min_stop_pts, market minimum stop distance,
    and rr_take ratio, the final stop distance SHALL be
    `max(ATR * stop_multiplier, min_stop_pts, market_min_stop)` and when
    `use_tp_limit` is true, the take-profit distance SHALL equal `final_stop * rr_take`.

    **Validates: Requirements 2.2, 2.3, 2.7, 2.9**
    """

    @given(scenario=stop_tp_scenario())
    @settings(max_examples=100)
    def test_final_stop_is_max_of_three_values(self, scenario):
        """final_stop == max(atr * multiplier, min_stop_pts, market_min_stop)."""
        atr_value, stop_multiplier, min_stop_pts, market_min_stop, rr_take, use_tp_limit = scenario

        config = {
            "conditional_orders": {
                "enabled": True,
                "buffer_points": 2.0,
                "order_expiry_seconds": 300,
                "max_entry_distance_points": 30.0,
            },
            "ai_strategy": {
                "stop_multiplier": stop_multiplier,
                "min_stop_pts": min_stop_pts,
                "rr_take": rr_take,
            },
            "execution": {
                "use_tp_limit": use_tp_limit,
            },
        }
        manager = _make_manager(config)

        final_stop, tp_distance = manager.calculate_stop_tp(atr_value, market_min_stop)

        expected_stop = max(atr_value * stop_multiplier, min_stop_pts, market_min_stop)
        assert abs(final_stop - expected_stop) < 1e-9, (
            f"final_stop should be max(atr*multiplier, min_stop_pts, market_min_stop). "
            f"Got {final_stop}, expected {expected_stop} "
            f"(atr={atr_value}, multiplier={stop_multiplier}, "
            f"min_stop_pts={min_stop_pts}, market_min={market_min_stop})"
        )

    @given(scenario=stop_tp_scenario())
    @settings(max_examples=100)
    def test_tp_equals_final_stop_times_rr_when_enabled(self, scenario):
        """When use_tp_limit=True: tp == final_stop * rr_take."""
        atr_value, stop_multiplier, min_stop_pts, market_min_stop, rr_take, _ = scenario

        config = {
            "conditional_orders": {
                "enabled": True,
                "buffer_points": 2.0,
                "order_expiry_seconds": 300,
                "max_entry_distance_points": 30.0,
            },
            "ai_strategy": {
                "stop_multiplier": stop_multiplier,
                "min_stop_pts": min_stop_pts,
                "rr_take": rr_take,
            },
            "execution": {
                "use_tp_limit": True,
            },
        }
        manager = _make_manager(config)

        final_stop, tp_distance = manager.calculate_stop_tp(atr_value, market_min_stop)

        expected_tp = final_stop * rr_take
        assert tp_distance is not None, "TP should not be None when use_tp_limit=True"
        assert abs(tp_distance - expected_tp) < 1e-9, (
            f"TP should be final_stop * rr_take. "
            f"Got {tp_distance}, expected {expected_tp} "
            f"(final_stop={final_stop}, rr_take={rr_take})"
        )

    @given(scenario=stop_tp_scenario())
    @settings(max_examples=100)
    def test_tp_is_none_when_disabled(self, scenario):
        """When use_tp_limit=False: tp is None."""
        atr_value, stop_multiplier, min_stop_pts, market_min_stop, rr_take, _ = scenario

        config = {
            "conditional_orders": {
                "enabled": True,
                "buffer_points": 2.0,
                "order_expiry_seconds": 300,
                "max_entry_distance_points": 30.0,
            },
            "ai_strategy": {
                "stop_multiplier": stop_multiplier,
                "min_stop_pts": min_stop_pts,
                "rr_take": rr_take,
            },
            "execution": {
                "use_tp_limit": False,
            },
        }
        manager = _make_manager(config)

        final_stop, tp_distance = manager.calculate_stop_tp(atr_value, market_min_stop)

        assert tp_distance is None, (
            f"TP should be None when use_tp_limit=False. Got {tp_distance}"
        )


# ---------------------------------------------------------------------------
# Strategies for Property 4: Expiry Timestamp Calculation
# ---------------------------------------------------------------------------

@composite
def expiry_timestamp_scenario(draw):
    """Generate arbitrary UTC time and expiry_seconds in [60, 86400].

    Returns (frozen_utc_time, expiry_seconds).
    """
    frozen_time = draw(datetimes(
        min_value=datetime(2020, 1, 1),
        max_value=datetime(2030, 12, 31),
    ))
    expiry_seconds = draw(integers(min_value=60, max_value=86400))
    return frozen_time, expiry_seconds


# ---------------------------------------------------------------------------
# Property 4: Expiry Timestamp Calculation
# Feature: conditional-order-entry, Property 4: Expiry Timestamp Calculation
# Validates: Requirements 2.5, 3.1
# ---------------------------------------------------------------------------

class TestExpiryTimestampCalculation:
    """Property 4: Expiry Timestamp Calculation.

    For any current UTC time and `order_expiry_seconds` value within [60, 86400],
    the `goodTillDate` field SHALL be an ISO 8601 UTC string representing
    current time plus `order_expiry_seconds`.

    **Validates: Requirements 2.5, 3.1**
    """

    @given(scenario=expiry_timestamp_scenario())
    @settings(max_examples=100)
    def test_expiry_equals_frozen_time_plus_seconds(self, scenario):
        """goodTillDate == frozen_utc_time + expiry_seconds, formatted as ISO 8601."""
        frozen_time, expiry_seconds = scenario

        config = {
            "conditional_orders": {
                "enabled": True,
                "buffer_points": 2.0,
                "order_expiry_seconds": expiry_seconds,
                "max_entry_distance_points": 30.0,
            }
        }
        manager = _make_manager(config)

        # Freeze datetime.now to return our known UTC time
        frozen_utc = frozen_time.replace(tzinfo=timezone.utc)
        with patch("broker.conditional_order_manager.datetime") as mock_datetime:
            mock_datetime.now.return_value = frozen_utc
            # Ensure timedelta is still accessible (not mocked)
            mock_datetime.side_effect = lambda *args, **kwargs: datetime(*args, **kwargs)

            result = manager.compute_expiry_timestamp()

        expected_dt = frozen_utc + timedelta(seconds=expiry_seconds)
        expected_str = expected_dt.strftime("%Y/%m/%d %H:%M:%S")

        assert result == expected_str, (
            f"Expiry timestamp should be frozen_time + expiry_seconds. "
            f"Got '{result}', expected '{expected_str}' "
            f"(frozen_time={frozen_utc.isoformat()}, expiry_seconds={expiry_seconds})"
        )


# ---------------------------------------------------------------------------
# Strategies for Property 5: Order Payload Construction
# ---------------------------------------------------------------------------

@composite
def order_payload_scenario(draw):
    """Generate arbitrary valid signal parameters for order payload construction.

    Returns (epic, direction, entry_level, size, stop_distance, tp_distance,
             currency_code, expiry_timestamp).
    """
    epic = draw(text(min_size=1, max_size=30, alphabet="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._"))
    direction = draw(sampled_from(["BUY", "SELL"]))
    entry_level = draw(floats(min_value=0.01, max_value=100000.0, allow_nan=False, allow_infinity=False))
    size = draw(floats(min_value=0.01, max_value=10000.0, allow_nan=False, allow_infinity=False))
    stop_distance = draw(floats(min_value=0.01, max_value=5000.0, allow_nan=False, allow_infinity=False))
    tp_distance = draw(one_of(
        none(),
        floats(min_value=0.01, max_value=10000.0, allow_nan=False, allow_infinity=False),
    ))
    currency_code = draw(text(min_size=3, max_size=3, alphabet="ABCDEFGHIJKLMNOPQRSTUVWXYZ"))
    # ISO 8601 timestamp (simplified valid format)
    year = draw(integers(min_value=2024, max_value=2030))
    month = draw(integers(min_value=1, max_value=12))
    day = draw(integers(min_value=1, max_value=28))
    hour = draw(integers(min_value=0, max_value=23))
    minute = draw(integers(min_value=0, max_value=59))
    second = draw(integers(min_value=0, max_value=59))
    expiry_timestamp = f"{year:04d}-{month:02d}-{day:02d}T{hour:02d}:{minute:02d}:{second:02d}"

    return epic, direction, entry_level, size, stop_distance, tp_distance, currency_code, expiry_timestamp


# ---------------------------------------------------------------------------
# Property 5: Order Payload Construction
# Feature: conditional-order-entry, Property 5: Order Payload Construction
# Validates: Requirements 2.1, 2.6
# ---------------------------------------------------------------------------

class TestOrderPayloadConstruction:
    """Property 5: Order Payload Construction.

    For any valid signal parameters (epic, direction, entry level, size,
    stop distance, optional TP distance, currency code, and expiry timestamp),
    the constructed order payload SHALL contain `orderType: "STOP"`,
    `timeInForce: "GOOD_TILL_DATE"`, and all provided parameters mapped to
    their correct IG API field names.

    **Validates: Requirements 2.1, 2.6**
    """

    @given(scenario=order_payload_scenario())
    @settings(max_examples=100)
    def test_payload_type_is_stop(self, scenario):
        """payload["type"] == "STOP" for all valid inputs."""
        epic, direction, entry_level, size, stop_distance, tp_distance, currency_code, expiry_timestamp = scenario

        config = {
            "conditional_orders": {
                "enabled": True,
                "buffer_points": 2.0,
                "order_expiry_seconds": 300,
                "max_entry_distance_points": 30.0,
            }
        }
        manager = _make_manager(config)

        payload = manager.build_order_payload(
            epic=epic, direction=direction, entry_level=entry_level,
            size=size, stop_distance=stop_distance, tp_distance=tp_distance,
            currency_code=currency_code, expiry_timestamp=expiry_timestamp,
        )

        assert payload["type"] == "STOP", (
            f"payload['type'] should be 'STOP', got '{payload['type']}'"
        )

    @given(scenario=order_payload_scenario())
    @settings(max_examples=100)
    def test_payload_time_in_force_is_good_till_date(self, scenario):
        """payload["timeInForce"] == "GOOD_TILL_DATE" for all valid inputs."""
        epic, direction, entry_level, size, stop_distance, tp_distance, currency_code, expiry_timestamp = scenario

        config = {
            "conditional_orders": {
                "enabled": True,
                "buffer_points": 2.0,
                "order_expiry_seconds": 300,
                "max_entry_distance_points": 30.0,
            }
        }
        manager = _make_manager(config)

        payload = manager.build_order_payload(
            epic=epic, direction=direction, entry_level=entry_level,
            size=size, stop_distance=stop_distance, tp_distance=tp_distance,
            currency_code=currency_code, expiry_timestamp=expiry_timestamp,
        )

        assert payload["timeInForce"] == "GOOD_TILL_DATE", (
            f"payload['timeInForce'] should be 'GOOD_TILL_DATE', got '{payload['timeInForce']}'"
        )

    @given(scenario=order_payload_scenario())
    @settings(max_examples=100)
    def test_payload_field_mappings(self, scenario):
        """All provided parameters are mapped to their correct IG API field names."""
        epic, direction, entry_level, size, stop_distance, tp_distance, currency_code, expiry_timestamp = scenario

        config = {
            "conditional_orders": {
                "enabled": True,
                "buffer_points": 2.0,
                "order_expiry_seconds": 300,
                "max_entry_distance_points": 30.0,
            }
        }
        manager = _make_manager(config)

        payload = manager.build_order_payload(
            epic=epic, direction=direction, entry_level=entry_level,
            size=size, stop_distance=stop_distance, tp_distance=tp_distance,
            currency_code=currency_code, expiry_timestamp=expiry_timestamp,
        )

        # Verify all field mappings
        assert payload["epic"] == epic, f"epic mismatch: {payload['epic']} != {epic}"
        assert payload["direction"] == direction, f"direction mismatch: {payload['direction']} != {direction}"
        assert payload["level"] == entry_level, f"level mismatch: {payload['level']} != {entry_level}"
        assert payload["size"] == size, f"size mismatch: {payload['size']} != {size}"
        assert payload["stopDistance"] == stop_distance, f"stopDistance mismatch: {payload['stopDistance']} != {stop_distance}"
        assert payload["goodTillDate"] == expiry_timestamp, f"goodTillDate mismatch: {payload['goodTillDate']} != {expiry_timestamp}"
        assert payload["currencyCode"] == currency_code, f"currencyCode mismatch: {payload['currencyCode']} != {currency_code}"
        assert payload["forceOpen"] is True, f"forceOpen should be True, got {payload['forceOpen']}"
        assert payload["guaranteedStop"] is False, f"guaranteedStop should be False, got {payload['guaranteedStop']}"

    @given(scenario=order_payload_scenario())
    @settings(max_examples=100)
    def test_payload_limit_distance_present_when_tp_provided(self, scenario):
        """When tp_distance is not None: payload["limitDistance"] == tp_distance."""
        epic, direction, entry_level, size, stop_distance, tp_distance, currency_code, expiry_timestamp = scenario

        assume(tp_distance is not None)

        config = {
            "conditional_orders": {
                "enabled": True,
                "buffer_points": 2.0,
                "order_expiry_seconds": 300,
                "max_entry_distance_points": 30.0,
            }
        }
        manager = _make_manager(config)

        payload = manager.build_order_payload(
            epic=epic, direction=direction, entry_level=entry_level,
            size=size, stop_distance=stop_distance, tp_distance=tp_distance,
            currency_code=currency_code, expiry_timestamp=expiry_timestamp,
        )

        assert "limitDistance" in payload, "limitDistance should be present when tp_distance is not None"
        assert payload["limitDistance"] == tp_distance, (
            f"limitDistance mismatch: {payload['limitDistance']} != {tp_distance}"
        )

    @given(scenario=order_payload_scenario())
    @settings(max_examples=100)
    def test_payload_limit_distance_absent_when_tp_none(self, scenario):
        """When tp_distance is None: "limitDistance" not in payload."""
        epic, direction, entry_level, size, stop_distance, _, currency_code, expiry_timestamp = scenario

        config = {
            "conditional_orders": {
                "enabled": True,
                "buffer_points": 2.0,
                "order_expiry_seconds": 300,
                "max_entry_distance_points": 30.0,
            }
        }
        manager = _make_manager(config)

        payload = manager.build_order_payload(
            epic=epic, direction=direction, entry_level=entry_level,
            size=size, stop_distance=stop_distance, tp_distance=None,
            currency_code=currency_code, expiry_timestamp=expiry_timestamp,
        )

        assert "limitDistance" not in payload, (
            f"limitDistance should NOT be in payload when tp_distance is None, "
            f"but found limitDistance={payload.get('limitDistance')}"
        )


# ---------------------------------------------------------------------------
# Strategies for Property 6: Signal Direction Handling
# ---------------------------------------------------------------------------

@composite
def signal_direction_scenario(draw):
    """Generate an epic with existing order direction and new signal direction.

    Returns (epic, existing_direction, new_direction).
    """
    epic = draw(text(min_size=3, max_size=20, alphabet="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._"))
    existing_direction = draw(sampled_from(["BUY", "SELL"]))
    new_direction = draw(sampled_from(["BUY", "SELL"]))
    return epic, existing_direction, new_direction


# ---------------------------------------------------------------------------
# Property 6: Signal Direction Handling
# Feature: conditional-order-entry, Property 6: Signal Direction Handling
# Validates: Requirements 4.1, 4.2
# ---------------------------------------------------------------------------

class TestSignalDirectionHandling:
    """Property 6: Signal Direction Handling.

    For any epic with an existing pending order in direction D and a new signal
    in direction D', the existing order SHALL be cancelled if and only if D ≠ D'.
    If D = D', the existing order SHALL be kept unchanged and no duplicate placed.

    **Validates: Requirements 4.1, 4.2**
    """

    @given(scenario=signal_direction_scenario())
    @settings(max_examples=100)
    def test_same_direction_skips_duplicate(self, scenario):
        """Same direction D == D' → existing order kept, result is 'skipped' with reason 'duplicate_order'."""
        epic, existing_direction, new_direction = scenario
        assume(existing_direction == new_direction)

        config = {
            "conditional_orders": {
                "enabled": True,
                "buffer_points": 2.0,
                "order_expiry_seconds": 300,
                "max_entry_distance_points": 30.0,
            }
        }

        ig_client = MagicMock()
        position_manager = MagicMock()
        position_manager.positions = {}  # No open positions
        trailing_manager = MagicMock()
        sr_detector = MagicMock()
        log = logging.getLogger("test_signal_direction")

        manager = ConditionalOrderManager(
            ig_client=ig_client,
            config=config,
            position_manager=position_manager,
            trailing_manager=trailing_manager,
            sr_detector=sr_detector,
            log=log,
        )

        # Simulate an existing tracked order by placing one via mock
        ig_client.place_working_order.return_value = {"dealReference": "test-ref-001"}
        existing_order = TrackedOrder(
            epic=epic,
            deal_id="test-ref-001",
            direction=existing_direction,
            entry_level=100.0,
            stop_distance=10.0,
            tp_distance=20.0,
            size=1.0,
            currency_code="USD",
            placed_at=datetime.now(timezone.utc),
            expiry_at=datetime.now(timezone.utc) + timedelta(seconds=300),
            confidence=0.8,
            patterns=["test_pattern"],
        )
        manager.tracked_orders[epic] = existing_order
        manager.active_signals[epic] = existing_direction

        # Send new signal with same direction
        result = manager.process_signal(
            epic=epic,
            direction=new_direction,
            mid_price=100.0,
            sr_levels={"resistance": [105.0], "support": [95.0]},
            stop_pts=10.0,
            tp_pts=20.0,
            size=1.0,
            currency_code="USD",
            confidence=0.8,
            patterns=["test_pattern"],
            atr_value=5.0,
        )

        # Verify: skipped with duplicate_order reason
        assert result["action"] == "skipped", (
            f"Same direction should skip. Got action='{result['action']}'"
        )
        assert result["details"]["reason"] == "duplicate_order", (
            f"Reason should be 'duplicate_order'. Got '{result['details']['reason']}'"
        )

        # Verify: delete_working_order was NOT called
        ig_client.delete_working_order.assert_not_called()

        # Verify: existing order is still tracked
        assert epic in manager.tracked_orders, (
            f"Existing order should remain tracked for epic={epic}"
        )

    @given(scenario=signal_direction_scenario())
    @settings(max_examples=100)
    def test_opposite_direction_cancels_existing(self, scenario):
        """Opposite direction D ≠ D' → existing order cancelled (delete_working_order called)."""
        epic, existing_direction, new_direction = scenario
        assume(existing_direction != new_direction)

        config = {
            "conditional_orders": {
                "enabled": True,
                "buffer_points": 2.0,
                "order_expiry_seconds": 300,
                "max_entry_distance_points": 30.0,
            }
        }

        ig_client = MagicMock()
        position_manager = MagicMock()
        position_manager.positions = {}  # No open positions
        trailing_manager = MagicMock()
        sr_detector = MagicMock()
        log = logging.getLogger("test_signal_direction")

        manager = ConditionalOrderManager(
            ig_client=ig_client,
            config=config,
            position_manager=position_manager,
            trailing_manager=trailing_manager,
            sr_detector=sr_detector,
            log=log,
        )

        # Simulate an existing tracked order
        existing_order = TrackedOrder(
            epic=epic,
            deal_id="test-ref-001",
            direction=existing_direction,
            entry_level=100.0,
            stop_distance=10.0,
            tp_distance=20.0,
            size=1.0,
            currency_code="USD",
            placed_at=datetime.now(timezone.utc),
            expiry_at=datetime.now(timezone.utc) + timedelta(seconds=300),
            confidence=0.8,
            patterns=["test_pattern"],
        )
        manager.tracked_orders[epic] = existing_order
        manager.active_signals[epic] = existing_direction

        # Mock place_working_order for the new order placement
        ig_client.place_working_order.return_value = {"dealReference": "test-ref-002"}

        # Send new signal with opposite direction
        result = manager.process_signal(
            epic=epic,
            direction=new_direction,
            mid_price=100.0,
            sr_levels={"resistance": [105.0], "support": [95.0]},
            stop_pts=10.0,
            tp_pts=20.0,
            size=1.0,
            currency_code="USD",
            confidence=0.8,
            patterns=["test_pattern"],
            atr_value=5.0,
        )

        # Verify: delete_working_order WAS called to cancel existing order
        ig_client.delete_working_order.assert_called_once_with("test-ref-001")

        # Verify: result is not 'skipped' (new order should proceed)
        assert result["action"] != "skipped", (
            f"Opposite direction should not skip. Got action='{result['action']}'"
        )



# ---------------------------------------------------------------------------
# Property 7: One-Order-Per-Epic Invariant
# Feature: conditional-order-entry, Property 7: One-Order-Per-Epic Invariant
# Validates: Requirements 5.1
# ---------------------------------------------------------------------------

# Small set of epics for stateful testing (keeps state space manageable)
_TEST_EPICS = ["IX.D.FTSE.DAILY.IP", "CS.D.EURUSD.CFD.IP", "CS.D.GBPUSD.CFD.IP"]


class OneOrderPerEpicStateMachine(RuleBasedStateMachine):
    """Property 7: One-Order-Per-Epic Invariant.

    For any sequence of order placements and cancellations, the internal
    tracking state SHALL contain at most one active working order per epic
    at any point in time.

    Uses stateful testing to simulate arbitrary interleaving of process_signal
    and cancel_order calls, verifying the invariant holds after every step.

    **Validates: Requirements 5.1**
    """

    def __init__(self):
        super().__init__()
        # Set up mocked dependencies
        self.ig_client = MagicMock()
        # place_working_order returns a successful response with a deal reference
        self.ig_client.place_working_order.return_value = {"dealReference": "MOCK_DEAL_REF"}
        self.ig_client.delete_working_order.return_value = {"status": "SUCCESS"}

        self.position_manager = MagicMock()
        # No open positions by default — allow placements
        self.position_manager.positions = {}

        self.trailing_manager = MagicMock()

        self.sr_detector = MagicMock()

        self.log = logging.getLogger("test_one_order_per_epic")

        config = {
            "conditional_orders": {
                "enabled": True,
                "buffer_points": 2.0,
                "order_expiry_seconds": 300,
                "max_entry_distance_points": 1000.0,  # Large to avoid rejections
            }
        }

        self.manager = ConditionalOrderManager(
            ig_client=self.ig_client,
            config=config,
            position_manager=self.position_manager,
            trailing_manager=self.trailing_manager,
            sr_detector=self.sr_detector,
            log=self.log,
        )

    @rule(
        epic=sampled_from(_TEST_EPICS),
        direction=sampled_from(["BUY", "SELL"]),
        mid_price=floats(min_value=100.0, max_value=5000.0, allow_nan=False, allow_infinity=False),
    )
    def place_order(self, epic, direction, mid_price):
        """Simulate a signal arriving — call process_signal with valid S/R levels."""
        # Build S/R levels that will produce a valid entry for the given direction
        if direction == "BUY":
            sr_levels = {"resistance": [mid_price + 5.0], "support": []}
        else:
            sr_levels = {"resistance": [], "support": [mid_price - 5.0]}

        self.manager.process_signal(
            epic=epic,
            direction=direction,
            mid_price=mid_price,
            sr_levels=sr_levels,
            stop_pts=10.0,
            tp_pts=20.0,
            size=1.0,
            currency_code="USD",
            confidence=0.8,
            patterns=["test_pattern"],
            atr_value=5.0,
        )

    @rule(epic=sampled_from(_TEST_EPICS))
    def cancel_order(self, epic):
        """Simulate cancellation of a tracked order for a random epic."""
        self.manager.cancel_order(epic, "test_cancellation")

    @invariant()
    def at_most_one_order_per_epic(self):
        """The tracked_orders dict must have at most one entry per epic.

        Since tracked_orders uses epic as the key, this is structurally
        enforced by the dict, but we verify:
        1. No duplicate keys (inherent in dict)
        2. Number of unique epics == number of entries
        3. Each epic appears at most once
        """
        tracked = self.manager.tracked_orders
        # Dict keys are inherently unique, but verify count consistency
        epic_keys = list(tracked.keys())
        assert len(epic_keys) == len(set(epic_keys)), (
            f"Duplicate epic keys found in tracked_orders! Keys: {epic_keys}"
        )
        # Also verify no epic has more than one tracked order
        assert len(tracked) <= len(_TEST_EPICS), (
            f"More tracked orders ({len(tracked)}) than possible epics ({len(_TEST_EPICS)}). "
            f"Tracked epics: {epic_keys}"
        )


# Run the state machine as a test
TestOneOrderPerEpic = OneOrderPerEpicStateMachine.TestCase
TestOneOrderPerEpic.settings = settings(max_examples=100, stateful_step_count=50)


# ---------------------------------------------------------------------------
# Strategies for Property 8: Order Lifecycle State Machine
# ---------------------------------------------------------------------------

@composite
def lifecycle_epic_scenario(draw):
    """Generate an epic and direction for lifecycle state machine testing.

    Returns (epic, direction, mid_price).
    """
    epic = draw(text(min_size=3, max_size=20, alphabet="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._"))
    direction = draw(sampled_from(["BUY", "SELL"]))
    mid_price = draw(floats(min_value=100.0, max_value=5000.0, allow_nan=False, allow_infinity=False))
    return epic, direction, mid_price


# ---------------------------------------------------------------------------
# Property 8: Order Lifecycle State Machine
# Feature: conditional-order-entry, Property 8: Order Lifecycle State Machine
# Validates: Requirements 5.2, 5.3, 5.4
# ---------------------------------------------------------------------------

class TestOrderLifecycleStateMachine:
    """Property 8: Order Lifecycle State Machine.

    For any epic, when a working order is filled, it SHALL be removed from
    tracking and new signals for that epic SHALL be rejected while
    `PositionManager` holds an open position. When the position is closed,
    new signals SHALL be accepted again.

    **Validates: Requirements 5.2, 5.3, 5.4**
    """

    @given(scenario=lifecycle_epic_scenario())
    @settings(max_examples=100)
    def test_filled_order_removed_from_tracking(self, scenario):
        """After fill (poll_orders detects order gone + position exists): tracked_orders[epic] is removed."""
        epic, direction, mid_price = scenario

        config = {
            "conditional_orders": {
                "enabled": True,
                "buffer_points": 2.0,
                "order_expiry_seconds": 300,
                "max_entry_distance_points": 1000.0,
            }
        }

        ig_client = MagicMock()
        position_manager = MagicMock()
        position_manager.positions = {}
        trailing_manager = MagicMock()
        sr_detector = MagicMock()
        log = logging.getLogger("test_lifecycle")

        manager = ConditionalOrderManager(
            ig_client=ig_client,
            config=config,
            position_manager=position_manager,
            trailing_manager=trailing_manager,
            sr_detector=sr_detector,
            log=log,
        )

        # Place a tracked order manually
        tracked = TrackedOrder(
            epic=epic,
            deal_id="FILL_DEAL_001",
            direction=direction,
            entry_level=mid_price + 5.0 if direction == "BUY" else mid_price - 5.0,
            stop_distance=10.0,
            tp_distance=20.0,
            size=1.0,
            currency_code="USD",
            placed_at=datetime.now(timezone.utc),
            expiry_at=datetime.now(timezone.utc) + timedelta(seconds=300),
            confidence=0.8,
            patterns=["test_pattern"],
        )
        manager.tracked_orders[epic] = tracked
        manager.active_signals[epic] = direction

        # Simulate fill: IG returns empty working orders (order gone)
        # AND position_manager.positions includes the epic (position opened)
        ig_client.get_working_orders.return_value = {"workingOrders": []}
        position_manager.positions = {epic: {"direction": direction, "size": 1.0}}

        # Poll orders — should detect fill and remove from tracking
        manager.poll_orders()

        assert epic not in manager.tracked_orders, (
            f"Filled order should be removed from tracked_orders. "
            f"Epic '{epic}' still present in: {list(manager.tracked_orders.keys())}"
        )

    @given(scenario=lifecycle_epic_scenario())
    @settings(max_examples=100)
    def test_new_signal_rejected_while_position_open(self, scenario):
        """While position is open: process_signal returns 'skipped' with reason 'position_open'."""
        epic, direction, mid_price = scenario

        config = {
            "conditional_orders": {
                "enabled": True,
                "buffer_points": 2.0,
                "order_expiry_seconds": 300,
                "max_entry_distance_points": 1000.0,
            }
        }

        ig_client = MagicMock()
        position_manager = MagicMock()
        # Position is open for this epic
        position_manager.positions = {epic: {"direction": direction, "size": 1.0}}
        trailing_manager = MagicMock()
        sr_detector = MagicMock()
        log = logging.getLogger("test_lifecycle")

        manager = ConditionalOrderManager(
            ig_client=ig_client,
            config=config,
            position_manager=position_manager,
            trailing_manager=trailing_manager,
            sr_detector=sr_detector,
            log=log,
        )

        # Build valid S/R levels for the signal
        if direction == "BUY":
            sr_levels = {"resistance": [mid_price + 5.0], "support": []}
        else:
            sr_levels = {"resistance": [], "support": [mid_price - 5.0]}

        # Send a new signal while position is open
        result = manager.process_signal(
            epic=epic,
            direction=direction,
            mid_price=mid_price,
            sr_levels=sr_levels,
            stop_pts=10.0,
            tp_pts=20.0,
            size=1.0,
            currency_code="USD",
            confidence=0.8,
            patterns=["test_pattern"],
            atr_value=5.0,
        )

        assert result["action"] == "skipped", (
            f"Signal should be skipped while position is open. "
            f"Got action='{result['action']}'"
        )
        assert result["details"]["reason"] == "position_open", (
            f"Reason should be 'position_open'. "
            f"Got '{result['details']['reason']}'"
        )

    @given(scenario=lifecycle_epic_scenario())
    @settings(max_examples=100)
    def test_new_signal_accepted_after_position_closed(self, scenario):
        """After position closed: process_signal accepts new signals (placed/fallback)."""
        epic, direction, mid_price = scenario

        config = {
            "conditional_orders": {
                "enabled": True,
                "buffer_points": 2.0,
                "order_expiry_seconds": 300,
                "max_entry_distance_points": 1000.0,
            }
        }

        ig_client = MagicMock()
        ig_client.place_working_order.return_value = {"dealReference": "NEW_DEAL_001"}
        position_manager = MagicMock()
        # Position is CLOSED — epic NOT in positions
        position_manager.positions = {}
        trailing_manager = MagicMock()
        sr_detector = MagicMock()
        log = logging.getLogger("test_lifecycle")

        manager = ConditionalOrderManager(
            ig_client=ig_client,
            config=config,
            position_manager=position_manager,
            trailing_manager=trailing_manager,
            sr_detector=sr_detector,
            log=log,
        )

        # Build valid S/R levels for the signal
        if direction == "BUY":
            sr_levels = {"resistance": [mid_price + 5.0], "support": []}
        else:
            sr_levels = {"resistance": [], "support": [mid_price - 5.0]}

        # Send a new signal after position is closed
        result = manager.process_signal(
            epic=epic,
            direction=direction,
            mid_price=mid_price,
            sr_levels=sr_levels,
            stop_pts=10.0,
            tp_pts=20.0,
            size=1.0,
            currency_code="USD",
            confidence=0.8,
            patterns=["test_pattern"],
            atr_value=5.0,
        )

        # Should proceed — either "placed" or "fallback" (not "skipped" with position_open)
        assert result["action"] in ("placed", "fallback"), (
            f"Signal should be accepted after position closed. "
            f"Got action='{result['action']}', details={result.get('details', {})}"
        )



# ---------------------------------------------------------------------------
# Strategies for Property 9: Expired Order Removal
# ---------------------------------------------------------------------------

@composite
def expired_order_removal_scenario(draw):
    """Generate a set of tracked orders and a subset that are "expired" (missing from IG).

    Returns (all_tracked_epics, expired_epics, remaining_epics).
    Each entry in all_tracked_epics is a tuple (epic, deal_id, direction).
    """
    # Generate between 1 and 6 tracked orders with unique epics
    num_orders = draw(integers(min_value=1, max_value=6))
    epics = [f"EPIC.{i}.{draw(text(min_size=2, max_size=5, alphabet='ABCDEFGHIJKLMNOPQRSTUVWXYZ'))}" for i in range(num_orders)]
    # Ensure uniqueness
    epics = list(set(epics))
    assume(len(epics) >= 1)

    all_tracked = []
    for epic in epics:
        deal_id = f"DEAL_{epic}_{draw(integers(min_value=1000, max_value=9999))}"
        direction = draw(sampled_from(["BUY", "SELL"]))
        all_tracked.append((epic, deal_id, direction))

    # Choose a non-empty subset to be "expired" (not present in IG response)
    # At least 1 expired, at most all expired
    num_expired = draw(integers(min_value=1, max_value=len(all_tracked)))
    expired_indices = draw(
        lists(
            sampled_from(list(range(len(all_tracked)))),
            min_size=num_expired,
            max_size=num_expired,
            unique=True,
        )
    )

    expired_epics = [all_tracked[i][0] for i in expired_indices]
    remaining_epics = [t[0] for i, t in enumerate(all_tracked) if i not in expired_indices]

    return all_tracked, expired_epics, remaining_epics


# ---------------------------------------------------------------------------
# Property 9: Expired Order Removal
# Feature: conditional-order-entry, Property 9: Expired Order Removal
# Validates: Requirements 3.3
# ---------------------------------------------------------------------------

class TestExpiredOrderRemoval:
    """Property 9: Expired Order Removal.

    For any tracked order that the IG API reports as cancelled or expired,
    it SHALL be removed from internal tracking state after detection during polling.

    **Validates: Requirements 3.3**
    """

    @given(scenario=expired_order_removal_scenario())
    @settings(max_examples=100)
    def test_expired_orders_removed_from_tracking(self, scenario):
        """Orders not present in IG response (and not in positions) are removed from tracked_orders."""
        all_tracked, expired_epics, remaining_epics = scenario

        config = {
            "conditional_orders": {
                "enabled": True,
                "buffer_points": 2.0,
                "order_expiry_seconds": 300,
                "max_entry_distance_points": 30.0,
            }
        }

        ig_client = MagicMock()
        position_manager = MagicMock()
        # Expired epics are NOT in positions (so they're treated as expired, not filled)
        position_manager.positions = {}
        trailing_manager = MagicMock()
        sr_detector = MagicMock()
        log = logging.getLogger("test_expired_order_removal")

        manager = ConditionalOrderManager(
            ig_client=ig_client,
            config=config,
            position_manager=position_manager,
            trailing_manager=trailing_manager,
            sr_detector=sr_detector,
            log=log,
        )

        # Setup: populate tracked_orders and active_signals with all orders
        for epic, deal_id, direction in all_tracked:
            tracked = TrackedOrder(
                epic=epic,
                deal_id=deal_id,
                direction=direction,
                entry_level=100.0,
                stop_distance=10.0,
                tp_distance=20.0,
                size=1.0,
                currency_code="USD",
                placed_at=datetime.now(timezone.utc),
                expiry_at=datetime.now(timezone.utc) + timedelta(seconds=300),
                confidence=0.8,
                patterns=["pattern"],
            )
            manager.tracked_orders[epic] = tracked
            manager.active_signals[epic] = direction

        # Mock get_working_orders to return only the "remaining" orders (expired ones are gone)
        remaining_deal_ids = {
            t[1] for t in all_tracked if t[0] in remaining_epics
        }
        working_orders_response = {
            "workingOrders": [
                {
                    "workingOrderData": {
                        "dealId": deal_id,
                        "epic": epic,
                        "direction": direction,
                    }
                }
                for epic, deal_id, direction in all_tracked
                if deal_id in remaining_deal_ids
            ]
        }
        ig_client.get_working_orders.return_value = working_orders_response

        # Act: poll_orders should detect expired orders and remove them
        manager.poll_orders()

        # Verify: all expired epics are removed from tracked_orders
        for epic in expired_epics:
            assert epic not in manager.tracked_orders, (
                f"Expired epic '{epic}' should be removed from tracked_orders after poll. "
                f"Still tracked: {list(manager.tracked_orders.keys())}"
            )

        # Verify: all expired epics are removed from active_signals
        for epic in expired_epics:
            assert epic not in manager.active_signals, (
                f"Expired epic '{epic}' should be removed from active_signals after poll. "
                f"Still in active_signals: {list(manager.active_signals.keys())}"
            )

    @given(scenario=expired_order_removal_scenario())
    @settings(max_examples=100)
    def test_remaining_orders_still_tracked(self, scenario):
        """Orders still present in IG response remain in tracked_orders."""
        all_tracked, expired_epics, remaining_epics = scenario
        assume(len(remaining_epics) >= 1)

        config = {
            "conditional_orders": {
                "enabled": True,
                "buffer_points": 2.0,
                "order_expiry_seconds": 300,
                "max_entry_distance_points": 30.0,
            }
        }

        ig_client = MagicMock()
        position_manager = MagicMock()
        # No positions for any epic — expired orders are treated as expired, not filled
        position_manager.positions = {}
        trailing_manager = MagicMock()
        sr_detector = MagicMock()
        log = logging.getLogger("test_expired_order_removal")

        manager = ConditionalOrderManager(
            ig_client=ig_client,
            config=config,
            position_manager=position_manager,
            trailing_manager=trailing_manager,
            sr_detector=sr_detector,
            log=log,
        )

        # Setup: populate tracked_orders and active_signals
        for epic, deal_id, direction in all_tracked:
            tracked = TrackedOrder(
                epic=epic,
                deal_id=deal_id,
                direction=direction,
                entry_level=100.0,
                stop_distance=10.0,
                tp_distance=20.0,
                size=1.0,
                currency_code="USD",
                placed_at=datetime.now(timezone.utc),
                expiry_at=datetime.now(timezone.utc) + timedelta(seconds=300),
                confidence=0.8,
                patterns=["pattern"],
            )
            manager.tracked_orders[epic] = tracked
            manager.active_signals[epic] = direction

        # Mock get_working_orders — remaining orders are still on IG
        remaining_deal_ids = {
            t[1] for t in all_tracked if t[0] in remaining_epics
        }
        working_orders_response = {
            "workingOrders": [
                {
                    "workingOrderData": {
                        "dealId": deal_id,
                        "epic": epic,
                        "direction": direction,
                    }
                }
                for epic, deal_id, direction in all_tracked
                if deal_id in remaining_deal_ids
            ]
        }
        ig_client.get_working_orders.return_value = working_orders_response

        # Act
        manager.poll_orders()

        # Verify: remaining epics are still tracked
        for epic in remaining_epics:
            assert epic in manager.tracked_orders, (
                f"Remaining epic '{epic}' should still be in tracked_orders after poll. "
                f"Currently tracked: {list(manager.tracked_orders.keys())}"
            )
            assert epic in manager.active_signals, (
                f"Remaining epic '{epic}' should still be in active_signals after poll. "
                f"Currently in active_signals: {list(manager.active_signals.keys())}"
            )


# ---------------------------------------------------------------------------
# Strategies for Property 10: Cancellation Retry Logic
# ---------------------------------------------------------------------------

@composite
def cancel_attempts_scenario(draw):
    """Generate a number of cancel attempts to test retry logic.

    Returns (num_attempts,) where num_attempts is between 1 and 6.
    """
    num_attempts = draw(integers(min_value=1, max_value=6))
    return (num_attempts,)


@composite
def cancel_success_before_max_scenario(draw):
    """Generate a scenario where cancellation succeeds before reaching max retries.

    Returns (fail_count,) where fail_count is 0, 1, or 2 (success on next attempt).
    """
    fail_count = draw(integers(min_value=0, max_value=2))
    return (fail_count,)


# ---------------------------------------------------------------------------
# Property 10: Cancellation Retry Logic
# Feature: conditional-order-entry, Property 10: Cancellation Retry Logic
# Validates: Requirements 4.5, 4.6
# ---------------------------------------------------------------------------

class TestCancellationRetryLogic:
    """Property 10: Cancellation Retry Logic.

    For any working order cancellation that fails due to API error, the retry
    count SHALL increment by 1 per failed polling cycle. After 3 consecutive
    failures, no further retry attempts SHALL be made for that order.

    **Validates: Requirements 4.5, 4.6**
    """

    @given(scenario=cancel_attempts_scenario())
    @settings(max_examples=100)
    def test_retry_count_increments_and_stops_at_3(self, scenario):
        """Each failed cancel increments retry_count; after 3 failures, order is abandoned."""
        (num_attempts,) = scenario

        config = {
            "conditional_orders": {
                "enabled": True,
                "buffer_points": 2.0,
                "order_expiry_seconds": 300,
                "max_entry_distance_points": 30.0,
            }
        }

        ig_client = MagicMock()
        # Always raise an exception on delete to simulate API failure
        ig_client.delete_working_order.side_effect = Exception("API error: service unavailable")

        position_manager = MagicMock()
        position_manager.positions = {}
        trailing_manager = MagicMock()
        sr_detector = MagicMock()
        log = logging.getLogger("test_cancel_retry")

        manager = ConditionalOrderManager(
            ig_client=ig_client,
            config=config,
            position_manager=position_manager,
            trailing_manager=trailing_manager,
            sr_detector=sr_detector,
            log=log,
        )

        epic = "TEST.EPIC.CANCEL"
        tracked = TrackedOrder(
            epic=epic,
            deal_id="deal-cancel-001",
            direction="BUY",
            entry_level=100.0,
            stop_distance=10.0,
            tp_distance=20.0,
            size=1.0,
            currency_code="USD",
            placed_at=datetime.now(timezone.utc),
            expiry_at=datetime.now(timezone.utc) + timedelta(seconds=300),
            confidence=0.8,
            patterns=["pattern_a"],
        )
        manager.tracked_orders[epic] = tracked
        manager.active_signals[epic] = "BUY"

        # Call cancel_order N times
        for i in range(num_attempts):
            result = manager.cancel_order(epic, "test_retry")

            if i < 3:
                # First 3 attempts: API is called, fails, retry_count increments
                assert result is False, (
                    f"Attempt {i+1}: cancel_order should return False on API failure"
                )
                if i < 2:
                    # Order still tracked (retry_count < 3 after this attempt)
                    assert epic in manager.tracked_orders, (
                        f"Attempt {i+1}: order should still be tracked (retry_count={i+1})"
                    )
                    assert manager.tracked_orders[epic].cancel_retry_count == i + 1, (
                        f"Attempt {i+1}: retry_count should be {i+1}, "
                        f"got {manager.tracked_orders[epic].cancel_retry_count}"
                    )
                elif i == 2:
                    # After 3rd failure, retry_count == 3, still tracked
                    # (abandoned only on the NEXT attempt when count >= 3)
                    assert epic in manager.tracked_orders, (
                        f"Attempt 3: order should still be tracked after 3rd failure"
                    )
                    assert manager.tracked_orders[epic].cancel_retry_count == 3, (
                        f"Attempt 3: retry_count should be 3, "
                        f"got {manager.tracked_orders[epic].cancel_retry_count}"
                    )
            else:
                # 4th+ attempt: retry_count >= 3, no API call, order abandoned
                assert result is False, (
                    f"Attempt {i+1}: cancel_order should return False (abandoned)"
                )
                assert epic not in manager.tracked_orders, (
                    f"Attempt {i+1}: order should be removed from tracking (abandoned)"
                )
                # No further iterations make sense — order is gone
                break

        # Verify: no more than 3 actual API calls were made
        assert ig_client.delete_working_order.call_count <= 3, (
            f"At most 3 API calls should be made, "
            f"got {ig_client.delete_working_order.call_count}"
        )

    @given(scenario=cancel_success_before_max_scenario())
    @settings(max_examples=100)
    def test_successful_cancel_removes_order(self, scenario):
        """If API succeeds on any attempt before 3 failures, order is removed successfully."""
        (fail_count,) = scenario

        config = {
            "conditional_orders": {
                "enabled": True,
                "buffer_points": 2.0,
                "order_expiry_seconds": 300,
                "max_entry_distance_points": 30.0,
            }
        }

        ig_client = MagicMock()
        # First fail_count calls raise, then succeed
        side_effects = [Exception("API error")] * fail_count + [{"status": "SUCCESS"}]
        ig_client.delete_working_order.side_effect = side_effects

        position_manager = MagicMock()
        position_manager.positions = {}
        trailing_manager = MagicMock()
        sr_detector = MagicMock()
        log = logging.getLogger("test_cancel_success")

        manager = ConditionalOrderManager(
            ig_client=ig_client,
            config=config,
            position_manager=position_manager,
            trailing_manager=trailing_manager,
            sr_detector=sr_detector,
            log=log,
        )

        epic = "TEST.EPIC.SUCCESS"
        tracked = TrackedOrder(
            epic=epic,
            deal_id="deal-success-001",
            direction="SELL",
            entry_level=200.0,
            stop_distance=15.0,
            tp_distance=30.0,
            size=2.0,
            currency_code="GBP",
            placed_at=datetime.now(timezone.utc),
            expiry_at=datetime.now(timezone.utc) + timedelta(seconds=300),
            confidence=0.9,
            patterns=["pattern_b"],
        )
        manager.tracked_orders[epic] = tracked
        manager.active_signals[epic] = "SELL"

        # First fail_count attempts should fail
        for i in range(fail_count):
            result = manager.cancel_order(epic, "test_retry_then_success")
            assert result is False, (
                f"Attempt {i+1}: should fail (API error)"
            )
            assert epic in manager.tracked_orders, (
                f"Attempt {i+1}: order should still be tracked"
            )
            assert manager.tracked_orders[epic].cancel_retry_count == i + 1, (
                f"Attempt {i+1}: retry_count should be {i+1}"
            )

        # Next attempt should succeed
        result = manager.cancel_order(epic, "test_retry_then_success")
        assert result is True, (
            f"Attempt {fail_count+1}: should succeed"
        )
        assert epic not in manager.tracked_orders, (
            f"After successful cancel, order should be removed from tracking"
        )
        assert epic not in manager.active_signals, (
            f"After successful cancel, active signal should be removed"
        )

        # Total API calls = fail_count + 1 (the successful one)
        assert ig_client.delete_working_order.call_count == fail_count + 1, (
            f"Expected {fail_count+1} API calls, got {ig_client.delete_working_order.call_count}"
        )


# ---------------------------------------------------------------------------
# Strategies for Property 12: Fill Handoff Correctness
# ---------------------------------------------------------------------------

@composite
def tracked_order_fill_scenario(draw):
    """Generate arbitrary TrackedOrder parameters for fill handoff testing.

    Returns a dict with all TrackedOrder fields needed to verify add_position call.
    """
    epic = draw(text(
        alphabet="ABCDEFGHIJKLMNOPQRSTUVWXYZ._",
        min_size=3, max_size=20,
    ))
    deal_id = draw(text(
        alphabet="abcdefghijklmnopqrstuvwxyz0123456789-",
        min_size=5, max_size=30,
    ))
    direction = draw(sampled_from(["BUY", "SELL"]))
    entry_level = draw(floats(min_value=1.0, max_value=50000.0, allow_nan=False, allow_infinity=False))
    stop_distance = draw(floats(min_value=0.1, max_value=500.0, allow_nan=False, allow_infinity=False))
    tp_distance = draw(one_of(
        none(),
        floats(min_value=0.1, max_value=1000.0, allow_nan=False, allow_infinity=False),
    ))
    size = draw(floats(min_value=0.01, max_value=100.0, allow_nan=False, allow_infinity=False))
    currency_code = draw(sampled_from(["USD", "GBP", "EUR", "AUD", "JPY"]))
    confidence = draw(floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False))
    patterns = draw(lists(
        text(alphabet="abcdefghijklmnopqrstuvwxyz_", min_size=2, max_size=15),
        min_size=0, max_size=5,
    ))

    return {
        "epic": epic,
        "deal_id": deal_id,
        "direction": direction,
        "entry_level": entry_level,
        "stop_distance": stop_distance,
        "tp_distance": tp_distance,
        "size": size,
        "currency_code": currency_code,
        "confidence": confidence,
        "patterns": patterns,
    }


# ---------------------------------------------------------------------------
# Property 12: Fill Handoff Correctness
# Feature: conditional-order-entry, Property 12: Fill Handoff Correctness
# Validates: Requirements 6.1
# ---------------------------------------------------------------------------

class TestFillHandoffCorrectness:
    """Property 12: Fill Handoff Correctness.

    For any fill event with a fill price, deal ID, direction, size, stop distance,
    and optional TP distance, the PositionManager.add_position() call SHALL be
    invoked with parameters matching the original order's signal data and the fill
    price from the IG API response.

    **Validates: Requirements 6.1**
    """

    @given(scenario=tracked_order_fill_scenario())
    @settings(max_examples=100)
    def test_add_position_called_with_correct_parameters(self, scenario):
        """add_position is called with parameters matching original order signal data."""
        config = {
            "conditional_orders": {
                "enabled": True,
                "buffer_points": 2.0,
                "order_expiry_seconds": 300,
                "max_entry_distance_points": 30.0,
            }
        }

        ig_client = MagicMock()
        # get_working_orders returns empty list — order is gone (filled)
        ig_client.get_working_orders.return_value = {"workingOrders": []}

        position_manager = MagicMock()
        # Position exists for the epic → indicates fill
        position_manager.positions = {scenario["epic"]: True}

        trailing_manager = MagicMock()
        sr_detector = MagicMock()
        log = logging.getLogger("test_fill_handoff")

        manager = ConditionalOrderManager(
            ig_client=ig_client,
            config=config,
            position_manager=position_manager,
            trailing_manager=trailing_manager,
            sr_detector=sr_detector,
            log=log,
        )

        # Manually add a TrackedOrder to manager.tracked_orders
        tracked = TrackedOrder(
            epic=scenario["epic"],
            deal_id=scenario["deal_id"],
            direction=scenario["direction"],
            entry_level=scenario["entry_level"],
            stop_distance=scenario["stop_distance"],
            tp_distance=scenario["tp_distance"],
            size=scenario["size"],
            currency_code=scenario["currency_code"],
            placed_at=datetime.now(timezone.utc),
            expiry_at=datetime.now(timezone.utc) + timedelta(seconds=300),
            confidence=scenario["confidence"],
            patterns=scenario["patterns"],
        )
        manager.tracked_orders[scenario["epic"]] = tracked
        manager.active_signals[scenario["epic"]] = scenario["direction"]

        # Call poll_orders — should detect the fill and call add_position
        manager.poll_orders()

        # Verify add_position was called exactly once
        position_manager.add_position.assert_called_once()

        # Verify the call arguments match original order signal data
        call_kwargs = position_manager.add_position.call_args[1]

        assert call_kwargs["epic"] == scenario["epic"], (
            f"epic mismatch: expected {scenario['epic']}, got {call_kwargs['epic']}"
        )
        assert call_kwargs["deal_id"] == scenario["deal_id"], (
            f"deal_id mismatch: expected {scenario['deal_id']}, got {call_kwargs['deal_id']}"
        )
        assert call_kwargs["direction"] == scenario["direction"], (
            f"direction mismatch: expected {scenario['direction']}, got {call_kwargs['direction']}"
        )
        assert call_kwargs["size"] == scenario["size"], (
            f"size mismatch: expected {scenario['size']}, got {call_kwargs['size']}"
        )
        assert call_kwargs["entry_price"] == scenario["entry_level"], (
            f"entry_price mismatch: expected {scenario['entry_level']}, got {call_kwargs['entry_price']}"
        )
        assert call_kwargs["stop"] == scenario["stop_distance"], (
            f"stop mismatch: expected {scenario['stop_distance']}, got {call_kwargs['stop']}"
        )

        expected_tp = scenario["tp_distance"] if scenario["tp_distance"] is not None else 0
        assert call_kwargs["tp"] == expected_tp, (
            f"tp mismatch: expected {expected_tp}, got {call_kwargs['tp']}"
        )
        assert call_kwargs["confidence"] == scenario["confidence"], (
            f"confidence mismatch: expected {scenario['confidence']}, got {call_kwargs['confidence']}"
        )
        assert call_kwargs["patterns"] == scenario["patterns"], (
            f"patterns mismatch: expected {scenario['patterns']}, got {call_kwargs['patterns']}"
        )

    @given(scenario=tracked_order_fill_scenario())
    @settings(max_examples=100)
    def test_filled_order_removed_from_tracking(self, scenario):
        """After fill handoff, the order is removed from tracked_orders."""
        config = {
            "conditional_orders": {
                "enabled": True,
                "buffer_points": 2.0,
                "order_expiry_seconds": 300,
                "max_entry_distance_points": 30.0,
            }
        }

        ig_client = MagicMock()
        ig_client.get_working_orders.return_value = {"workingOrders": []}

        position_manager = MagicMock()
        position_manager.positions = {scenario["epic"]: True}

        trailing_manager = MagicMock()
        sr_detector = MagicMock()
        log = logging.getLogger("test_fill_removal")

        manager = ConditionalOrderManager(
            ig_client=ig_client,
            config=config,
            position_manager=position_manager,
            trailing_manager=trailing_manager,
            sr_detector=sr_detector,
            log=log,
        )

        tracked = TrackedOrder(
            epic=scenario["epic"],
            deal_id=scenario["deal_id"],
            direction=scenario["direction"],
            entry_level=scenario["entry_level"],
            stop_distance=scenario["stop_distance"],
            tp_distance=scenario["tp_distance"],
            size=scenario["size"],
            currency_code=scenario["currency_code"],
            placed_at=datetime.now(timezone.utc),
            expiry_at=datetime.now(timezone.utc) + timedelta(seconds=300),
            confidence=scenario["confidence"],
            patterns=scenario["patterns"],
        )
        manager.tracked_orders[scenario["epic"]] = tracked
        manager.active_signals[scenario["epic"]] = scenario["direction"]

        manager.poll_orders()

        assert scenario["epic"] not in manager.tracked_orders, (
            f"Order for epic={scenario['epic']} should be removed from tracking after fill"
        )
        assert scenario["epic"] not in manager.active_signals, (
            f"Active signal for epic={scenario['epic']} should be removed after fill"
        )
