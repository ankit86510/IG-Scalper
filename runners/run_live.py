import os
import time
import pandas as pd
from core.config import load_settings
from core.logging_utils import setup_logging
from core.risk import size_by_invested_capital, daily_lockout
from data.ig_price_bars import bars_from_ig
from broker.ig_client import IGClient
from broker.order_exec import enforce_market_rules, estimate_pip_value
from strategy.ema_cross_breakout import EMACrossBreakout

RES = {"1min": "MINUTE", "5min": "MINUTE_5", "15min": "MINUTE_15"}

def get_bars(ig, epic, tf, lookback=200):
    pj = ig.get_prices(epic, resolution=RES[tf], max=lookback)
    return bars_from_ig(pj)

def trend_filter(df15):
    if df15 is None or len(df15) < 50:
        return None
    ema50 = df15["close"].ewm(span=50, adjust=False).mean()
    last = df15["close"].iloc[-1]; ema_last = ema50.iloc[-1]
    if last > ema_last: return "UP"
    if last < ema_last: return "DOWN"
    return None

def main():
    cfg = load_settings()
    log = setup_logging(cfg["logging"]["level"], cfg["logging"]["sink"])

    demo = cfg["ig"]["account_type"].upper() == "DEMO"
    ig = IGClient(
        api_key=cfg["ig"]["api_key"],
        username=cfg["ig"]["username"],
        password=cfg["ig"]["password"],
        demo=demo
    )
    ig.login()

    epics = cfg["symbols"]
    invest = cfg["risk"]["invest_per_trade"]
    max_loss_pct = cfg["risk"]["max_loss_pct_invest"]
    max_daily_loss_pct = cfg["risk"]["max_daily_loss_pct"]
    max_losers = cfg["risk"]["max_losing_trades"]

    strat = EMACrossBreakout(
        fast=cfg["strategy"]["params"]["fast"],
        slow=cfg["strategy"]["params"]["slow"],
        atr_period=cfg["strategy"]["params"]["atr_period"],
        rr_take=cfg["strategy"]["params"]["rr_take_profit"]
    )

    market_cache = {e: ig.market_details(e) for e in epics}
    last_bar_time = {e: None for e in epics}
    losing_trades = 0
    daily_pnl_pct = 0.0  # TODO: compute from account equity vs. start-of-day

    while True:
        try:
            # Kill switch via env
            if os.environ.get("KILL_SWITCH", "0") == "1":
                log.warning("Kill switch active. Exiting loop.")
                break

            if daily_lockout(daily_pnl_pct, max_daily_loss_pct) or losing_trades >= max_losers:
                log.warning("Daily lockout active. Sleeping 5 minutes.")
                time.sleep(300)
                continue

            for epic in epics:
                df1 = get_bars(ig, epic, cfg["timeframes"]["primary"])
                if df1 is None or df1.empty:
                    continue
                if last_bar_time[epic] is not None and df1.index[-1] == last_bar_time[epic]:
                    continue  # wait for a new bar

                df15 = get_bars(ig, epic, cfg["timeframes"]["trend"])
                direction_filter = trend_filter(df15)

                signal = strat.on_bar(df1)

                # Trend alignment optional
                if signal and direction_filter:
                    if signal["side"] == "BUY" and direction_filter == "DOWN":
                        signal = None
                    if signal and signal["side"] == "SELL" and direction_filter == "UP":
                        signal = None

                if signal:
                    mkt = market_cache[epic]
                    pip_value = estimate_pip_value(mkt)

                    proposed_size, _ = size_by_invested_capital(
                        invest_amount_gbp=invest,
                        max_loss_pct=max_loss_pct,
                        stop_pts=signal["stop_pts"],
                        pip_value_per_contract=pip_value,
                        min_size=mkt["dealingRules"]["minDealSize"]["value"],
                        size_step=mkt["dealingRules"]["minDealSize"].get("step", 0.1)
                    )
                    stop_pts, tp_pts, adj_size = enforce_market_rules(mkt, signal["stop_pts"], signal["tp_pts"], proposed_size)

                    if adj_size <= 0:
                        log.info(f"{epic}: size too small; skip")
                    else:
                        direction = "BUY" if signal["side"] == "BUY" else "SELL"
                        trailing = None
                        if cfg["execution"]["use_trailing_stop"]:
                            trailing = {"initial_distance": stop_pts, "increment": max(1.0, stop_pts * 0.25)}
                        try:
                            resp = ig.place_order(epic, direction, adj_size, stop_distance=None if trailing else stop_pts, limit_distance=tp_pts, trailing=trailing, tif=cfg["execution"]["time_in_force"])
                            log.info(f"Order placed {epic} dir={direction} size={adj_size} stop={stop_pts} tp={tp_pts} resp={resp.get('dealReference')}")
                        except Exception as e:
                            log.exception(f"Order error {epic}: {e}")

                last_bar_time[epic] = df1.index[-1]

            time.sleep(5)
        except Exception as e:
            log.exception(f"Main loop error: {e}")
            time.sleep(10)

if __name__ == "__main__":
    main()
