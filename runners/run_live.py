import os
import sys
import time
import pandas as pd

# Add parent directory to path so we can import from core/
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


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
    last = df15["close"].iloc[-1];
    ema_last = ema50.iloc[-1]
    if last > ema_last: return "UP"
    if last < ema_last: return "DOWN"
    return None


def sync_positions_from_broker(ig, log):
    """Fetch current open positions from broker"""
    try:
        positions_data = ig.positions()
        open_positions = {}
        for pos in positions_data.get('positions', []):
            epic = pos['market']['epic']
            open_positions[epic] = {
                'deal_id': pos['position']['dealId'],
                'direction': pos['position']['direction'],
                'size': pos['position']['dealSize'],
                'open_level': pos['position']['openLevel'],
                'created': pos['position']['createdDateUTC']
            }
        return open_positions
    except Exception as e:
        log.error(f"Failed to sync positions: {e}")
        return {}


def calculate_daily_pnl_pct(ig, start_equity, log):
    """Calculate daily P&L percentage"""
    try:
        acct = ig.account_summary()
        current_equity = acct['accounts'][0]['balance']['balance']
        pnl_pct = ((current_equity - start_equity) / start_equity) * 100
        return pnl_pct, current_equity
    except Exception as e:
        log.error(f"Failed to calculate P&L: {e}")
        return 0.0, start_equity


def should_close_position(epic, position_info, signal, log):
    """Check if we should close existing position based on new signal"""
    pos_direction = position_info['direction']

    if signal is None:
        return False

    # Close if signal is opposite direction
    if pos_direction == 'BUY' and signal['side'] == 'SELL':
        log.info(f"{epic}: Opposite signal detected, should close LONG position")
        return True
    elif pos_direction == 'SELL' and signal['side'] == 'BUY':
        log.info(f"{epic}: Opposite signal detected, should close SHORT position")
        return True

    return False


def main():
    cfg = load_settings()
    log = setup_logging(cfg["logging"]["level"], cfg["logging"]["sink"])

    demo = cfg["ig"]["account_type"].upper() == "DEMO"
    ig = IGClient(
        api_key=cfg["ig"]["api_key"],
        username=cfg["ig"]["username"],
        password=cfg["ig"]["password"],
        demo=demo,
        verify_ssl=False  # Disable SSL verification (for corporate proxy/AV issues)
    )
    ig.login()
    log.info("Logged in to IG successfully")

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

    # Position tracking
    open_positions = {}

    # P&L tracking
    start_equity = ig.account_summary()['accounts'][0]['balance']['balance']
    log.info(f"Starting equity: £{start_equity:.2f}")

    losing_trades = 0
    daily_pnl_pct = 0.0

    # Sync existing positions on startup
    open_positions = sync_positions_from_broker(ig, log)
    log.info(f"Synced {len(open_positions)} existing positions")

    loop_count = 0
    while True:
        try:
            loop_count += 1

            # Kill switch via env
            if os.environ.get("KILL_SWITCH", "0") == "1":
                log.warning("Kill switch active. Exiting loop.")
                break

            # Update P&L every 10 loops (every ~50 seconds)
            if loop_count % 10 == 0:
                daily_pnl_pct, current_equity = calculate_daily_pnl_pct(ig, start_equity, log)
                log.info(f"Daily P&L: {daily_pnl_pct:.2f}%, Current equity: £{current_equity:.2f}")

                # Resync positions periodically
                open_positions = sync_positions_from_broker(ig, log)

            if daily_lockout(daily_pnl_pct, max_daily_loss_pct) or losing_trades >= max_losers:
                log.warning(
                    f"Daily lockout active (P&L: {daily_pnl_pct:.2f}%, Losers: {losing_trades}). Sleeping 5 minutes.")
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

                # Check if we have existing position
                if epic in open_positions:
                    # Check if we should close the position
                    if should_close_position(epic, open_positions[epic], signal, log):
                        try:
                            pos = open_positions[epic]
                            close_direction = "SELL" if pos['direction'] == 'BUY' else "BUY"
                            resp = ig.close_position(
                                deal_id=pos['deal_id'],
                                direction=close_direction,
                                size=pos['size']
                            )
                            log.info(
                                f"Closed position {epic} deal_id={pos['deal_id']} resp={resp.get('dealReference')}")
                            del open_positions[epic]
                        except Exception as e:
                            log.exception(f"Error closing position {epic}: {e}")
                    else:
                        # Already have position, don't open another
                        last_bar_time[epic] = df1.index[-1]
                        continue

                # Open new position if we have a signal and no existing position
                if signal and epic not in open_positions:
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
                    stop_pts, tp_pts, adj_size = enforce_market_rules(mkt, signal["stop_pts"], signal["tp_pts"],
                                                                      proposed_size)

                    if adj_size <= 0:
                        log.info(f"{epic}: size too small; skip")
                    else:
                        direction = "BUY" if signal["side"] == "BUY" else "SELL"
                        trailing = None
                        if cfg["execution"]["use_trailing_stop"]:
                            trailing = {"initial_distance": stop_pts, "increment": max(1.0, stop_pts * 0.25)}
                        try:
                            resp = ig.place_order(
                                epic,
                                direction,
                                adj_size,
                                stop_distance=None if trailing else stop_pts,
                                limit_distance=tp_pts,
                                trailing=trailing,
                                tif=cfg["execution"]["time_in_force"]
                            )
                            deal_ref = resp.get('dealReference')
                            log.info(
                                f"Order placed {epic} dir={direction} size={adj_size} stop={stop_pts} tp={tp_pts} ref={deal_ref}")

                            # Add to tracked positions (deal_id will be updated on next sync)
                            open_positions[epic] = {
                                'deal_id': deal_ref,  # temporary, will be updated
                                'direction': direction,
                                'size': adj_size,
                                'entry_time': time.time()
                            }
                        except Exception as e:
                            log.exception(f"Order error {epic}: {e}")

                last_bar_time[epic] = df1.index[-1]

            time.sleep(5)
        except Exception as e:
            log.exception(f"Main loop error: {e}")
            time.sleep(10)


if __name__ == "__main__":
    main()