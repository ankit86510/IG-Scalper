import os
import sys
import time
import pandas as pd

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.config import load_settings
from core.logging_utils import setup_logging
from core.risk import size_by_invested_capital, daily_lockout
from data.ig_price_bars import bars_from_ig
from broker.ig_client import IGClient
from broker.order_exec import enforce_market_rules, estimate_pip_value

# Import all scalping strategies
from strategy.stochastic_scalper import StochasticScalper
from strategy.moving_average_scalper import MovingAverageScalper
from strategy.parabolic_sar_scalper import ParabolicSARScalper
from strategy.rsi_scalper import RSIScalper

RES = {"1sec": "SECOND", "1min": "MINUTE", "3min": "MINUTE_3", "5min": "MINUTE_5"}


def get_bars(ig, epic, tf, lookback=200):
    """Fetch price bars from IG"""
    resolution = RES.get(tf, "MINUTE")
    pj = ig.get_prices(epic, resolution=resolution, max=max(lookback, 200))
    return bars_from_ig(pj)


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
        log.info(f"Found {len(open_positions)} open positions")
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
    if signal is None:
        return False

    pos_direction = position_info['direction']

    # Close if signal is opposite direction
    if pos_direction == 'BUY' and signal['side'] == 'SELL':
        log.info(f"{epic}: Opposite signal detected, closing LONG")
        return True
    elif pos_direction == 'SELL' and signal['side'] == 'BUY':
        log.info(f"{epic}: Opposite signal detected, closing SHORT")
        return True

    return False


def main():
    cfg = load_settings()
    log = setup_logging(cfg["logging"]["level"], cfg["logging"]["sink"])

    log.info("=" * 60)
    log.info("MULTI-STRATEGY SCALPING BOT STARTING")
    log.info("=" * 60)

    # Login to IG
    demo = cfg["ig"]["account_type"].upper() == "DEMO"
    ig = IGClient(
        api_key=cfg["ig"]["api_key"],
        username=cfg["ig"]["username"],
        password=cfg["ig"]["password"],
        demo=demo,
        verify_ssl=False
    )
    ig.login()
    log.info("✓ Logged in to IG successfully")

    # Get configuration
    epics = cfg["symbols"]
    timeframe = cfg.get("timeframe", "3min")  # Use 3min for scalping
    invest = cfg["risk"]["invest_per_trade"]
    max_loss_pct = cfg["risk"]["max_loss_pct_invest"]
    max_daily_loss_pct = cfg["risk"]["max_daily_loss_pct"]
    max_losers = cfg["risk"]["max_losing_trades"]

    log.info(f"✓ Monitoring {len(epics)} instruments on {timeframe} timeframe")
    log.info(f"✓ Risk: £{invest} per trade, max {max_loss_pct}% loss")

    # Initialize strategies
    strategies = {
        "stochastic": StochasticScalper(
            k_period=14, d_period=3,
            oversold=20, overbought=80,
            stop_multiplier=1.0, rr_take=1.5
        ),
        "moving_average": MovingAverageScalper(
            fast_ma=5, slow_ma=20, trend_ma=200,
            stop_multiplier=1.0, rr_take=1.5
        ),
        "parabolic_sar": ParabolicSARScalper(
            acceleration=0.02, maximum=0.2,
            stop_multiplier=1.0, rr_take=1.5
        ),
        "rsi": RSIScalper(
            rsi_period=14, oversold=30, overbought=70,
            ma_fast=5, ma_med=20, ma_slow=50,
            stop_multiplier=1.0, rr_take=1.5
        )
    }

    # Select active strategy from config
    active_strategy_name = cfg["strategy"].get("name", "moving_average")
    if active_strategy_name not in strategies:
        log.warning(f"Unknown strategy '{active_strategy_name}', defaulting to 'moving_average'")
        active_strategy_name = "moving_average"

    strategy = strategies[active_strategy_name]
    log.info(f"✓ Active strategy: {active_strategy_name.upper()}")

    # Load market details
    market_cache = {}
    for e in epics:
        try:
            market_cache[e] = ig.market_details(e)
            log.info(f"✓ Loaded market details for {e}")
        except Exception as ex:
            log.error(f"✗ Failed to load market details for {e}: {ex}")

    # Initialize tracking
    last_bar_time = {e: None for e in epics}
    open_positions = sync_positions_from_broker(ig, log)

    # Get starting equity
    try:
        start_equity = ig.account_summary()['accounts'][0]['balance']['balance']
        log.info(f"✓ Starting equity: £{start_equity:.2f}")
    except Exception as e:
        log.error(f"Failed to get starting equity: {e}")
        start_equity = 10000.0

    losing_trades = 0
    daily_pnl_pct = 0.0
    loop_count = 0

    log.info("=" * 60)
    log.info("BOT IS NOW LIVE - Monitoring markets...")
    log.info("=" * 60)

    # Main trading loop
    while True:
        try:
            loop_count += 1

            # Kill switch
            if os.environ.get("KILL_SWITCH", "0") == "1":
                log.warning("⚠ Kill switch active. Exiting loop.")
                break

            # Update P&L every 10 loops
            if loop_count % 10 == 0:
                daily_pnl_pct, current_equity = calculate_daily_pnl_pct(ig, start_equity, log)
                log.info(f"📊 P&L: {daily_pnl_pct:+.2f}% | Equity: £{current_equity:.2f}")
                open_positions = sync_positions_from_broker(ig, log)

            # Check lockout conditions
            if daily_lockout(daily_pnl_pct, max_daily_loss_pct):
                log.warning(f"🛑 Daily loss limit reached: {daily_pnl_pct:.2f}%. Sleeping 5 min.")
                time.sleep(300)
                continue

            if losing_trades >= max_losers:
                log.warning(f"🛑 Max losing trades reached: {losing_trades}. Sleeping 5 min.")
                time.sleep(300)
                continue

            # Process each instrument
            for epic in epics:
                try:
                    # Get price bars
                    df = get_bars(ig, epic, timeframe, lookback=250)

                    if df is None or df.empty or len(df) < 50:
                        continue

                    # Check for new bar
                    if last_bar_time[epic] is not None and df.index[-1] == last_bar_time[epic]:
                        continue

                    # Generate signal
                    signal = strategy.on_bar(df)

                    # Handle existing positions
                    if epic in open_positions:
                        if should_close_position(epic, open_positions[epic], signal, log):
                            try:
                                pos = open_positions[epic]
                                close_direction = "SELL" if pos['direction'] == 'BUY' else "BUY"
                                resp = ig.close_position(
                                    deal_id=pos['deal_id'],
                                    direction=close_direction,
                                    size=pos['size']
                                )
                                log.info(f"✓ Closed {epic} position: {resp.get('dealReference')}")
                                del open_positions[epic]
                            except Exception as e:
                                log.exception(f"✗ Error closing {epic}: {e}")
                        else:
                            last_bar_time[epic] = df.index[-1]
                            continue

                    # Try to open new position
                    if signal and epic not in open_positions:
                        log.info(
                            f"🎯 SIGNAL: {epic} {signal['side']} - {signal.get('meta', {}).get('strategy', 'unknown')}")

                        if epic not in market_cache:
                            log.error(f"✗ No market details for {epic}")
                            last_bar_time[epic] = df.index[-1]
                            continue

                        mkt = market_cache[epic]
                        pip_value = estimate_pip_value(mkt)

                        # Calculate position size
                        proposed_size, max_loss = size_by_invested_capital(
                            invest_amount_gbp=invest,
                            max_loss_pct=max_loss_pct,
                            stop_pts=signal["stop_pts"],
                            pip_value_per_contract=pip_value,
                            min_size=mkt["dealingRules"]["minDealSize"]["value"],
                            size_step=0.1
                        )

                        # Enforce market rules
                        stop_pts, tp_pts, adj_size = enforce_market_rules(
                            mkt, signal["stop_pts"], signal["tp_pts"], proposed_size
                        )

                        if adj_size <= 0:
                            log.warning(f"✗ {epic}: Size too small ({adj_size}), skipping")
                        else:
                            direction = "BUY" if signal["side"] == "BUY" else "SELL"

                            # Use trailing stop if enabled
                            trailing = None
                            if cfg["execution"].get("use_trailing_stop", False):
                                trailing = {
                                    "initial_distance": stop_pts,
                                    "increment": max(1.0, stop_pts * 0.25)
                                }

                            try:
                                log.info(
                                    f"📈 PLACING ORDER: {epic} {direction} size={adj_size} stop={stop_pts} tp={tp_pts}")
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
                                log.info(f"✅ ORDER PLACED: {epic} ref={deal_ref}")

                                open_positions[epic] = {
                                    'deal_id': deal_ref,
                                    'direction': direction,
                                    'size': adj_size,
                                    'entry_time': time.time()
                                }
                            except Exception as e:
                                log.exception(f"❌ ORDER FAILED {epic}: {e}")

                    last_bar_time[epic] = df.index[-1]

                except Exception as e:
                    log.error(f"Error processing {epic}: {e}")
                    continue

            # Sleep between cycles
            time.sleep(10)

        except KeyboardInterrupt:
            log.info("⚠ Interrupted by user, exiting...")
            break
        except Exception as e:
            log.exception(f"Main loop error: {e}")
            time.sleep(10)


if __name__ == "__main__":
    main()