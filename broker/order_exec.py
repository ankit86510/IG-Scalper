import math

def enforce_market_rules(mkt, proposed_stop_pts, proposed_tp_pts, proposed_size):
    rules = mkt["dealingRules"]
    min_stop = rules["minStopOrLimitDistance"]["value"]
    size_rule = rules.get("minDealSize", {"value": 0.1, "step": 0.1})
    size_step = size_rule.get("step", 0.1)
    min_size = size_rule.get("value", 0.1)

    stop_pts = max(proposed_stop_pts, min_stop)
    size = max(min_size, math.floor(proposed_size / size_step) * size_step)
    tp_pts = max(proposed_tp_pts, stop_pts * 1.2)
    return stop_pts, tp_pts, size

def estimate_pip_value(mkt):
    # Conservative default: £1 per point per contract.
    # You can replace with per-EPIC constants later.
    return 1.0
