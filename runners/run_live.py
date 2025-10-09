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
    # Request more bars to ensure we have enough data
    pj = ig.get_prices(epic, resolution=RES[tf], max=max(lookback, 50))
    return bars_from_ig(pj)


def trend_filter(df15):
    if df15 is None or len(df15) < 50:
        return None
    ema50 = df15["close"].ewm(span=50, adjust=False).mean()
    last = df15["close"].iloc[-1]
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
        log.info(f"Found {len(open_positions)} open positions: {list(open_positions.keys())}")
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
        verify_ssl=False
    )
    ig.login()
    log.info("Logged in to IG successfully")

    epics = cfg["symbols"]
    log.info(f"Monitoring EPICs: {epics}")

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
    log.info(f"Strategy initialized: EMA({strat.fast}/{strat.slow}), ATR({strat.atr_period}), RR={strat.rr_take}")

    market_cache = {}
    for e in epics:
        try:
            market_cache[e] = ig.market_details(e)
            log.info(f"Loaded market details for {e}")
        except Exception as ex:
            log.error(f"Failed to load market details for {e}: {ex}")

    last_bar_time = {e: None for e in epics}

    # Position tracking
    open_positions = {}

    # P&L tracking
    try:
        start_equity = ig.account_summary()['accounts'][0]['balance']['balance']
        log.info(f"Starting equity: £{start_equity:.2f}")
    except Exception as e:
        log.error(f"Failed to get starting equity: {e}")
        start_equity = 10000.0

    losing_trades = 0
    daily_pnl_pct = 0.0

    # Sync existing positions on startup
    open_positions = sync_positions_from_broker(ig, log)

    loop_count = 0
    while True:
        try:
            loop_count += 1
            log.debug(f"=== Loop {loop_count} ===")

            # Kill switch via env
            if os.environ.get("KILL_SWITCH", "0") == "1":
                log.warning("Kill switch active. Exiting loop.")
                break

            # Update P&L every 10 loops
            if loop_count % 10 == 0:
                daily_pnl_pct, current_equity = calculate_daily_pnl_pct(ig, start_equity, log)
                log.info(f"Daily P&L: {daily_pnl_pct:.2f}%, Current equity: £{current_equity:.2f}")
                open_positions = sync_positions_from_broker(ig, log)

            # Check lockout conditions
            if daily_lockout(daily_pnl_pct, max_daily_loss_pct):
                log.warning(
                    f"Daily loss limit reached: {daily_pnl_pct:.2f}% <= -{max_daily_loss_pct}%. Sleeping 5 minutes.")
                time.sleep(300)
                continue

            if losing_trades >= max_losers:
                log.warning(f"Max losing trades reached: {losing_trades} >= {max_losers}. Sleeping 5 minutes.")
                time.sleep(300)
                continue

            for epic in epics:
                log.debug(f"Processing {epic}...")

                # Get primary timeframe bars
                try:
                    df1 = get_bars(ig, epic, cfg["timeframes"]["primary"])
                except Exception as e:
                    log.error(f"Failed to get bars for {epic}: {e}")
                    continue

                if df1 is None or df1.empty:
                    log.warning(f"{epic}: No data received")
                    continue

                log.debug(f"{epic}: Got {len(df1)} bars, latest: {df1.index[-1]}")

                # Check if we have a new bar
                if last_bar_time[epic] is not None and df1.index[-1] == last_bar_time[epic]:
                    log.debug(f"{epic}: Waiting for new bar (current: {df1.index[-1]})")
                    continue

                # Get trend timeframe
                try:
                    df15 = get_bars(ig, epic, cfg["timeframes"]["trend"])
                    direction_filter = trend_filter(df15)
                    log.debug(f"{epic}: Trend filter = {direction_filter}")
                except Exception as e:
                    log.warning(f"{epic}: Failed to get trend data: {e}")
                    direction_filter = None

                # Generate signal
                signal = strat.on_bar(df1)

                if signal:
                    log.info(f"{epic}: RAW SIGNAL detected: {signal}")
                else:
                    log.debug(f"{epic}: No signal generated")

                # Apply trend filter
                if signal and direction_filter:
                    if signal["side"] == "BUY" and direction_filter == "DOWN":
                        log.info(f"{epic}: BUY signal filtered out (trend is DOWN)")
                        signal = None
                    elif signal["side"] == "SELL" and direction_filter == "UP":
                        log.info(f"{epic}: SELL signal filtered out (trend is UP)")
                        signal = None

                # Check existing positions
                if epic in open_positions:
                    log.info(f"{epic}: Already have position: {open_positions[epic]['direction']}")

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
                        last_bar_time[epic] = df1.index[-1]
                        continue

                # Try to open new position
                if signal and epic not in open_positions:
                    log.info(f"{epic}: ATTEMPTING TO PLACE ORDER - Signal: {signal}")

                    if epic not in market_cache:
                        log.error(f"{epic}: No market details in cache!")
                        last_bar_time[epic] = df1.index[-1]
                        continue

                    mkt = market_cache[epic]
                    pip_value = estimate_pip_value(mkt)
                    log.info(f"{epic}: pip_value={pip_value}, stop_pts={signal['stop_pts']}")

                    proposed_size, max_loss = size_by_invested_capital(
                        invest_amount_gbp=invest,
                        max_loss_pct=max_loss_pct,
                        stop_pts=signal["stop_pts"],
                        pip_value_per_contract=pip_value,
                        min_size=mkt["dealingRules"]["minDealSize"]["value"],
                        size_step=mkt["dealingRules"]["minDealSize"].get("step", 0.1)
                    )
                    log.info(f"{epic}: proposed_size={proposed_size}, max_loss=£{max_loss:.2f}")

                    stop_pts, tp_pts, adj_size = enforce_market_rules(
                        mkt, signal["stop_pts"], signal["tp_pts"], proposed_size
                    )
                    log.info(f"{epic}: After rules - size={adj_size}, stop={stop_pts}, tp={tp_pts}")

                    if adj_size <= 0:
                        log.warning(f"{epic}: Adjusted size too small ({adj_size}), skipping order")
                    else:
                        direction = "BUY" if signal["side"] == "BUY" else "SELL"
                        trailing = None
                        if cfg["execution"]["use_trailing_stop"]:
                            trailing = {
                                "initial_distance": stop_pts,
                                "increment": max(1.0, stop_pts * 0.25)
                            }
                            log.info(f"{epic}: Using trailing stop: {trailing}")

                        try:
                            log.info(
                                f"{epic}: PLACING ORDER: dir={direction}, size={adj_size}, stop={stop_pts}, tp={tp_pts}")
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
                                f"✅ ORDER PLACED: {epic} dir={direction} size={adj_size} stop={stop_pts} tp={tp_pts} ref={deal_ref}")

                            open_positions[epic] = {
                                'deal_id': deal_ref,
                                'direction': direction,
                                'size': adj_size,
                                'entry_time': time.time()
                            }
                        except Exception as e:
                            log.exception(f"❌ ORDER FAILED {epic}: {e}")

                last_bar_time[epic] = df1.index[-1]

            time.sleep(5)

        except KeyboardInterrupt:
            log.info("Interrupted by user, exiting...")
            break
        except Exception as e:
            log.exception(f"Main loop error: {e}")
            time.sleep(10)


if __name__ == "__main__":
    main()