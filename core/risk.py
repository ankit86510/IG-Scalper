def round_step(value, step):
    if step <= 0:
        return value
    return round(round(value / step) * step, 8)

def size_by_invested_capital(invest_amount_gbp, max_loss_pct, stop_pts, pip_value_per_contract, min_size=0.1, size_step=0.1):
    max_loss_cash = invest_amount_gbp * (max_loss_pct / 100.0)
    if stop_pts <= 0 or pip_value_per_contract <= 0:
        return 0.0, 0.0
    raw_size = max_loss_cash / (stop_pts * pip_value_per_contract)
    sized = max(min_size, round_step(raw_size, size_step))
    return sized, max_loss_cash

def daily_lockout(daily_pnl_pct, max_daily_loss_pct):
    return daily_pnl_pct <= -abs(max_daily_loss_pct)
