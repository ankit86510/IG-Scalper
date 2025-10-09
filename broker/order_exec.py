import math


def enforce_market_rules(mkt, proposed_stop_pts, proposed_tp_pts, proposed_size):
    """
    Enforce broker market rules for stop distance and position size.
    Handles different IG API response structures.
    """
    rules = mkt.get("dealingRules", {})

    # Get min stop distance - can be in several places
    min_stop = None
    for key in ['minStopOrLimitDistance', 'minNormalStopOrLimitDistance',
                'minControlledRiskStopDistance']:
        if key in rules:
            min_stop = rules[key].get('value')
            if min_stop is not None:
                break

    # Default to 5 points if we can't find it
    if min_stop is None:
        min_stop = 5.0
        print(f"Warning: Could not find minStopDistance, defaulting to {min_stop}")

    # Get size rules
    size_rule = rules.get("minDealSize", {"value": 0.1, "step": 0.1})
    size_step = size_rule.get("step", 0.1)
    min_size = size_rule.get("value", 0.1)

    # Enforce minimum stop distance
    stop_pts = max(proposed_stop_pts, min_stop)

    # Enforce size rules (must be multiple of step, at least min_size)
    if size_step > 0:
        size = max(min_size, math.floor(proposed_size / size_step) * size_step)
    else:
        size = max(min_size, proposed_size)

    # Ensure TP is at least 1.2x stop (good practice)
    tp_pts = max(proposed_tp_pts, stop_pts * 1.2)

    return stop_pts, tp_pts, size


def estimate_pip_value(mkt):
    """
    Estimate pip value per contract for position sizing.
    This is a simplified version - real pip values vary by instrument.

    For more accuracy, you should set per-EPIC values:
    - Forex major pairs (EURUSD, GBPUSD): ~$10 per pip per lot
    - Forex crosses: varies
    - Indices (S&P, FTSE, DAX): £1 per point per contract
    - Commodities (Gold, Oil): varies by contract size
    """
    instrument_name = mkt.get('instrument', {}).get('name', '').lower()
    epic = mkt.get('instrument', {}).get('epic', '')

    # Forex pairs
    if 'CS.D.' in epic and 'CFD' in epic:
        # Most forex pairs traded in £ account
        return 10.0  # £10 per pip per lot

    # Indices
    if 'IX.D.' in epic:
        return 1.0  # £1 per point per contract

    # Commodities - Gold
    if 'GOLD' in instrument_name.upper():
        return 1.0  # £1 per point for spot gold

    # Default conservative estimate
    return 1.0