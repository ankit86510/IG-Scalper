import math


def enforce_market_rules(mkt, proposed_stop_pts, proposed_tp_pts, proposed_size):
    """
    Enforce broker market rules for stop distance and position size.
    Uses actual IG API response structure.
    """
    rules = mkt.get("dealingRules", {})
    instrument = mkt.get("instrument", {})

    # Get min stop distance - IG uses minNormalStopOrLimitDistance for regular stops
    # minControlledRiskStopDistance is for guaranteed stops (wider)
    min_stop = rules.get('minNormalStopOrLimitDistance', {}).get('value')

    # Fallback to controlled risk if normal not available
    if min_stop is None:
        min_stop = rules.get('minControlledRiskStopDistance', {}).get('value')

    # Ultimate fallback
    if min_stop is None:
        min_stop = 1.0
        print(f"Warning: Could not find minStopDistance, defaulting to {min_stop}")

    # Get size rules
    size_rule = rules.get("minDealSize", {})
    min_size = size_rule.get("value", 0.1)

    # For IG, deal size step is usually 0.1 for most instruments
    # Some indices might be 1.0
    size_step = 0.1 if min_size <= 0.1 else 1.0

    # Enforce minimum stop distance
    stop_pts = max(proposed_stop_pts, min_stop)

    # Enforce size rules (must be multiple of step, at least min_size)
    if size_step > 0:
        size = max(min_size, math.floor(proposed_size / size_step) * size_step)
    else:
        size = max(min_size, proposed_size)

    # Ensure TP is at least 1.2x stop (good practice for positive expectancy)
    tp_pts = max(proposed_tp_pts, stop_pts * 1.2)

    print(f"Market rules enforcement:")
    print(f"  Min stop: {min_stop}")
    print(f"  Min size: {min_size}")
    print(f"  Proposed stop: {proposed_stop_pts:.2f} -> Enforced: {stop_pts:.2f}")
    print(f"  Proposed TP: {proposed_tp_pts:.2f} -> Enforced: {tp_pts:.2f}")
    print(f"  Proposed size: {proposed_size:.2f} -> Enforced: {size:.2f}")

    return stop_pts, tp_pts, size


def estimate_pip_value(mkt):
    """
    Estimate pip value per contract for position sizing.
    Now uses actual IG API response data.
    """
    instrument = mkt.get('instrument', {})
    epic = instrument.get('epic', '')

    # IG provides this directly!
    value_of_one_pip = instrument.get('valueOfOnePip')
    if value_of_one_pip:
        try:
            return float(value_of_one_pip)
        except (ValueError, TypeError):
            pass

    # Fallback logic based on instrument type
    instrument_type = instrument.get('type', '').upper()
    instrument_name = instrument.get('name', '').lower()

    # Forex pairs (CURRENCIES type)
    if instrument_type == 'CURRENCIES':
        if 'CS.D.' in epic and 'CFD' in epic:
            # Standard forex pair
            return 10.0  # $10 per pip per standard lot
        # Gold/Silver are also classified as CURRENCIES
        if 'gold' in instrument_name or 'xau' in instrument_name:
            return 1.0  # $1 per point
        if 'silver' in instrument_name or 'xag' in instrument_name:
            return 1.0

    # Indices
    if instrument_type == 'INDICES' or 'IX.D.' in epic:
        return 1.0  # £1 per point per contract

    # Commodities
    if instrument_type == 'COMMODITIES':
        if 'oil' in instrument_name or 'crude' in instrument_name:
            return 1.0
        if 'gold' in instrument_name:
            return 1.0

    # Default conservative estimate
    print(f"Warning: Using default pip value for {epic}")
    return 1.0


def get_market_info_summary(mkt):
    """
    Extract and display key market information from IG API response.
    Useful for debugging.
    """
    instrument = mkt.get('instrument', {})
    rules = mkt.get('dealingRules', {})
    snapshot = mkt.get('snapshot', {})

    info = {
        'epic': instrument.get('epic'),
        'name': instrument.get('name'),
        'type': instrument.get('type'),
        'market_status': snapshot.get('marketStatus'),
        'bid': snapshot.get('bid'),
        'offer': snapshot.get('offer'),
        'spread': snapshot.get('offer', 0) - snapshot.get('bid', 0),
        'pip_value': instrument.get('valueOfOnePip'),
        'min_stop': rules.get('minNormalStopOrLimitDistance', {}).get('value'),
        'min_size': rules.get('minDealSize', {}).get('value'),
        'margin_factor': instrument.get('marginFactor'),
        'margin_unit': instrument.get('marginFactorUnit'),
    }

    return info


def calculate_position_value(size, price, pip_value):
    """
    Calculate the notional value of a position.
    Useful for risk management.
    """
    return size * price * pip_value


def calculate_margin_required(mkt, size, price=None):
    """
    Calculate margin required for a position based on IG rules.
    """
    instrument = mkt.get('instrument', {})
    snapshot = mkt.get('snapshot', {})

    # Use current price if not provided
    if price is None:
        price = (snapshot.get('bid', 0) + snapshot.get('offer', 0)) / 2

    # Get margin factor (percentage)
    margin_factor = instrument.get('marginFactor', 5)  # Default 5%
    margin_unit = instrument.get('marginFactorUnit', 'PERCENTAGE')

    # Calculate notional value
    lot_size = instrument.get('lotSize', 1.0)
    notional = size * lot_size * price

    # Calculate margin
    if margin_unit == 'PERCENTAGE':
        margin = notional * (margin_factor / 100.0)
    else:
        margin = notional * margin_factor

    return margin


def validate_order_before_placement(mkt, direction, size, stop_pts, tp_pts):
    """
    Validate order parameters before sending to broker.
    Returns (is_valid, error_message)
    """
    rules = mkt.get('dealingRules', {})
    snapshot = mkt.get('snapshot', {})

    # Check market is tradeable
    if snapshot.get('marketStatus') != 'TRADEABLE':
        return False, f"Market not tradeable: {snapshot.get('marketStatus')}"

    # Check minimum size
    min_size = rules.get('minDealSize', {}).get('value', 0.1)
    if size < min_size:
        return False, f"Size {size} below minimum {min_size}"

    # Check minimum stop
    min_stop = rules.get('minNormalStopOrLimitDistance', {}).get('value', 1.0)
    if stop_pts < min_stop:
        return False, f"Stop {stop_pts} below minimum {min_stop}"

    # Check TP > Stop (basic sanity)
    if tp_pts <= stop_pts:
        return False, f"TP {tp_pts} must be greater than stop {stop_pts}"

    # Check direction is valid
    if direction not in ['BUY', 'SELL']:
        return False, f"Invalid direction: {direction}"

    return True, "Valid"


# Example usage and testing
if __name__ == "__main__":
    # Example market data from IG API
    example_market = {
        "instrument": {
            "epic": "CS.D.CFEGOLD.CEB.IP",
            "name": "Spot Gold ($1)",
            "type": "CURRENCIES",
            "valueOfOnePip": "1.00",
            "marginFactor": 5,
            "marginFactorUnit": "PERCENTAGE"
        },
        "dealingRules": {
            "minNormalStopOrLimitDistance": {"unit": "POINTS", "value": 1.0},
            "minDealSize": {"unit": "POINTS", "value": 0.1},
        },
        "snapshot": {
            "marketStatus": "TRADEABLE",
            "bid": 3971.28,
            "offer": 3971.88,
        }
    }

    # Test the functions
    print("=== Testing Market Functions ===\n")

    # Get market info
    info = get_market_info_summary(example_market)
    print("Market Info:")
    for k, v in info.items():
        print(f"  {k}: {v}")

    # Test pip value
    pip_val = estimate_pip_value(example_market)
    print(f"\nEstimated pip value: ${pip_val}")

    # Test rule enforcement
    print("\n=== Testing Rule Enforcement ===")
    stop, tp, size = enforce_market_rules(
        example_market,
        proposed_stop_pts=0.5,  # Too small
        proposed_tp_pts=0.6,  # Too small
        proposed_size=0.05  # Too small
    )

    # Test validation
    print("\n=== Testing Validation ===")
    is_valid, msg = validate_order_before_placement(
        example_market,
        direction="BUY",
        size=size,
        stop_pts=stop,
        tp_pts=tp
    )
    print(f"Order valid: {is_valid}, Message: {msg}")