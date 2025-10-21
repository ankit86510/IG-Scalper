import os
import sys
import time
import pandas as pd
import json

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from datetime import datetime, UTC
from core.config import load_settings
from core.logging_utils import setup_logging
from core.risk import size_by_invested_capital, daily_lockout
from data.ig_price_bars import bars_from_ig
from broker.ig_client import IGClient
from broker.order_exec import enforce_market_rules, estimate_pip_value
from strategy.ai_pattern_recognizer import AIPatternRecognizer
from data.multi_data_provider import create_data_aggregator

RES = {"1sec": "SECOND", "1min": "MINUTE", "3min": "MINUTE_3", "5min": "MINUTE_5"}



class PositionManager:
    """Autonomous position management with AI decision tracking"""

    def __init__(self, log):
        self.log = log
        self.positions = {}
        self.trade_history = []
        self.decision_log = []

    def add_position(self, epic, deal_id, direction, size, entry_price, stop, tp, confidence, patterns):
        """Track new position with AI metadata"""
        self.positions[epic] = {
            'deal_id': deal_id,
            'direction': direction,
            'size': size,
            'entry_price': entry_price,
            'entry_time': datetime.utcnow(),
            'stop': stop,
            'tp': tp,
            'confidence': confidence,
            'patterns': patterns,
            'status': 'OPEN'
        }
        self.log.info(f"📝 Position tracked: {epic} {direction} @ {entry_price} (confidence: {confidence:.1%})")

    def remove_position(self, epic, exit_price=None, reason="CLOSED"):
        """Remove position and log to history"""
        if epic in self.positions:
            pos = self.positions[epic]
            pos['exit_time'] = datetime.utcnow()
            pos['exit_price'] = exit_price
            pos['status'] = reason

            if exit_price:
                # Calculate P&L
                if pos['direction'] == 'BUY':
                    pnl_pts = exit_price - pos['entry_price']
                else:
                    pnl_pts = pos['entry_price'] - exit_price

                pos['pnl_pts'] = pnl_pts
                duration = (pos['exit_time'] - pos['entry_time']).total_seconds() / 60
                pos['duration_minutes'] = duration

                self.log.info(f"📊 Closed {epic}: {pnl_pts:+.2f} pts in {duration:.1f} min")

            self.trade_history.append(pos)
            del self.positions[epic]

    def should_adjust_stop(self, epic, current_price, atr):
        """Determine if trailing stop should be adjusted"""
        if epic not in self.positions:
            return None

        pos = self.positions[epic]

        # Only trail winners
        if pos['direction'] == 'BUY':
            if current_price > pos['entry_price']:
                new_stop = current_price - (atr * 1.5)
                if new_stop > pos['stop']:
                    return new_stop
        else:  # SELL
            if current_price < pos['entry_price']:
                new_stop = current_price + (atr * 1.5)
                if new_stop < pos['stop']:
                    return new_stop

        return None

    def get_performance_stats(self):
        """Calculate performance statistics"""
        if not self.trade_history:
            return {"total_trades": 0}

        completed = [t for t in self.trade_history if 'pnl_pts' in t]

        if not completed:
            return {"total_trades": len(self.trade_history), "completed": 0}

        wins = [t for t in completed if t['pnl_pts'] > 0]
        losses = [t for t in completed if t['pnl_pts'] <= 0]

        total_pnl = sum(t['pnl_pts'] for t in completed)
        win_rate = len(wins) / len(completed) * 100 if completed else 0

        avg_win = sum(t['pnl_pts'] for t in wins) / len(wins) if wins else 0
        avg_loss = sum(t['pnl_pts'] for t in losses) / len(losses) if losses else 0

        return {
            "total_trades": len(self.trade_history),
            "completed": len(completed),
            "wins": len(wins),
            "losses": len(losses),
            "win_rate": win_rate,
            "total_pnl_pts": total_pnl,
            "avg_win_pts": avg_win,
            "avg_loss_pts": avg_loss,
            "profit_factor": abs(avg_win / avg_loss) if avg_loss != 0 else 0
        }


def get_bars_from_aggregator(aggregator, epic, tf, lookback=200):
    """Fetch price bars using smart aggregator"""
    return aggregator.get_bars(epic, tf, lookback)

def get_bars(ig, epic, tf, lookback=200):
    """Legacy function - kept for compatibility"""
    resolution = RES.get(tf, "MINUTE")
    pj = ig.get_prices(epic, resolution=resolution, max=max(lookback, 200))
    return bars_from_ig(pj)


def sync_positions_from_broker(ig, position_manager, log):
    """Sync positions from broker with position manager"""
    try:
        positions_data = ig.positions()
        broker_positions = {}

        for pos in positions_data.get('positions', []):
            epic = pos['market']['epic']
            broker_positions[epic] = {
                'deal_id': pos['position']['dealId'],
                'direction': pos['position']['direction'],
                'size': pos['position']['dealSize'],
                'open_level': pos['position']['openLevel']
            }

        # Remove positions from manager that aren't in broker
        for epic in list(position_manager.positions.keys()):
            if epic not in broker_positions:
                position_manager.remove_position(epic, reason="BROKER_CLOSED")
                log.info(f"Position {epic} removed - not found in broker")

        log.info(f"Synced {len(broker_positions)} positions from broker")
        return broker_positions

    except Exception as e:
        log.error(f"Failed to sync positions: {e}")
        return {}


def save_analysis_report(position_manager, filename="ai_analysis_report.json"):
    """Save AI analysis and performance report"""
    report = {
        "generated_at": datetime.now(UTC).isoformat(),
        "performance": position_manager.get_performance_stats(),
        "recent_trades": position_manager.trade_history[-20:],  # Last 20 trades
        "decision_log": position_manager.decision_log[-50:]  # Last 50 decisions
    }

    os.makedirs("data", exist_ok=True)
    filepath = os.path.join("data", filename)

    with open(filepath, 'w') as f:
        json.dump(report, f, indent=2, default=str)

    return filepath


def main():
    cfg = load_settings()
    log = setup_logging(cfg["logging"]["level"], cfg["logging"]["sink"])

    log.info("=" * 80)
    log.info("🤖 AI-POWERED AUTONOMOUS TRADING SYSTEM")
    log.info("=" * 80)

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
    log.info("✓ Connected to IG API")

    # After IG login
    log.info("Initializing multi-source data provider...")
    aggregator = create_data_aggregator(
        ig_client=ig,
        alpha_vantage_key=cfg["alphavantage"]["api_key"],
        twelve_data_key=cfg["12data"]["api_key"],
        # Add other keys as needed
    )
    log.info(f"✓ Data aggregator ready with {aggregator.get_stats()['providers_active']} providers")

    # Get configuration
    epics = cfg["symbols"]
    timeframe = cfg.get("timeframe", "1min")
    invest = cfg["risk"]["invest_per_trade"]
    max_loss_pct = cfg["risk"]["max_loss_pct_invest"]
    max_daily_loss_pct = cfg["risk"]["max_daily_loss_pct"]
    max_losers = cfg["risk"]["max_losing_trades"]

    log.info(f"✓ Monitoring {len(epics)} instruments")
    log.info(f"✓ Timeframe: {timeframe}")
    log.info(f"✓ Risk per trade: £{invest} (max {max_loss_pct}% loss)")

    # Initialize AI Strategy
    ai_config = cfg.get("ai_strategy", {})
    strategy = AIPatternRecognizer(
        atr_period=ai_config.get("atr_period", 14),
        stop_multiplier=ai_config.get("stop_multiplier", 1.5),
        rr_take=ai_config.get("rr_take", 2.0),
        confidence_threshold=ai_config.get("confidence_threshold", 0.65),
        lookback_candles=ai_config.get("lookback_candles", 50)
    )

    log.info(f"✓ AI Pattern Recognizer initialized")
    log.info(f"  - Confidence threshold: {ai_config.get('confidence_threshold', 0.65):.1%}")
    log.info(f"  - Risk/Reward ratio: 1:{ai_config.get('rr_take', 2.0)}")

    # Load market details
    market_cache = {}
    for e in epics:
        try:
            market_cache[e] = ig.market_details(e)
            log.info(f"✓ Market details loaded: {e}")
        except Exception as ex:
            log.error(f"✗ Failed to load {e}: {ex}")

    # Initialize position manager
    position_manager = PositionManager(log)
    last_bar_time = {e: None for e in epics}

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
    last_report_time = time.time()

    log.info("=" * 80)
    log.info("🚀 SYSTEM LIVE - AI analyzing markets autonomously...")
    log.info("=" * 80)

    # Main autonomous trading loop
    while True:
        try:
            loop_count += 1

            # Emergency kill switch
            if os.environ.get("KILL_SWITCH", "0") == "1":
                log.warning("⚠️ Kill switch activated - shutting down safely")
                break

            # Periodic reporting (every 5 minutes)
            if time.time() - last_report_time > 300:
                stats = position_manager.get_performance_stats()
                log.info("=" * 80)
                log.info(f"📊 PERFORMANCE UPDATE")
                log.info(f"  Trades: {stats.get('completed', 0)} | Win Rate: {stats.get('win_rate', 0):.1f}%")
                log.info(f"  Total P&L: {stats.get('total_pnl_pts', 0):+.2f} pts")
                log.info(f"  Open Positions: {len(position_manager.positions)}")
                log.info("=" * 80)

                # Save analysis report
                report_path = save_analysis_report(position_manager)
                log.info(f"📄 Report saved: {report_path}")

                last_report_time = time.time()

            # Update P&L and sync positions every 10 loops
            if loop_count % 10 == 0:
                try:
                    acct = ig.account_summary()
                    current_equity = acct['accounts'][0]['balance']['balance']
                    daily_pnl_pct = ((current_equity - start_equity) / start_equity) * 100
                    log.info(f"💰 Daily P&L: {daily_pnl_pct:+.2f}% | Equity: £{current_equity:.2f}")
                except Exception as e:
                    log.error(f"Failed to update equity: {e}")

                sync_positions_from_broker(ig, position_manager, log)

            # Check risk lockouts
            if daily_lockout(daily_pnl_pct, max_daily_loss_pct):
                log.warning(f"🛑 Daily loss limit reached: {daily_pnl_pct:.2f}%")
                log.warning("   Sleeping 10 minutes before resuming...")
                time.sleep(600)
                continue

            if losing_trades >= max_losers:
                log.warning(f"🛑 Max consecutive losses: {losing_trades}/{max_losers}")
                log.warning("   Cooling off for 10 minutes...")
                time.sleep(600)
                losing_trades = 0  # Reset after cooldown
                continue

            # Process each instrument
            for epic in epics:
                try:
                    # Get price data
                    df = get_bars_from_aggregator(aggregator, epic, timeframe, lookback=250)

                    # Validate data quality
                    if len(df) > 0:
                        # Check if all prices are the same (stale data)
                        if df['close'].nunique() == 1:
                            log.warning(f"{epic}: Stale data detected (all prices identical), skipping...")
                            continue

                        # Check for sufficient price movement
                        price_range = (df['high'].max() - df['low'].min()) / df['close'].iloc[-1]
                        if price_range < 0.001:  # Less than 0.1% range
                            log.warning(f"{epic}: Insufficient price movement, market may be closed")
                            continue

                    if df is None or df.empty or len(df) < 50:
                        continue

                    # Skip if bar hasn't updated
                    if last_bar_time[epic] is not None and df.index[-1] == last_bar_time[epic]:
                        continue

                    current_price = df['close'].iloc[-1]
                    current_atr = df['high'].rolling(14).max().iloc[-1] - df['low'].rolling(14).min().iloc[-1]

                    # Check for trailing stop adjustments on existing positions
                    if epic in position_manager.positions:
                        new_stop = position_manager.should_adjust_stop(epic, current_price, current_atr)
                        if new_stop:
                            log.info(f"📈 Trailing stop for {epic}: {new_stop:.2f}")
                            # Note: IG doesn't support stop modification via REST API easily
                            # This would require closing and reopening with new stop
                            # For now, we log it. In production, use Lightstreamer for dynamic updates

                    # Run AI analysis
                    log.info(f"🔍 Analyzing {epic}...")
                    signal = strategy.on_bar(df)

                    # Log AI decision
                    decision_entry = {
                        "timestamp": datetime.now(UTC).isoformat(),
                        "epic": epic,
                        "signal": signal is not None,
                        "confidence": signal["meta"]["confidence"] if signal else 0.0
                    }
                    position_manager.decision_log.append(decision_entry)

                    # Handle existing position - check for exit signals
                    if epic in position_manager.positions:
                        pos = position_manager.positions[epic]

                        # Exit on opposite signal with high confidence
                        if signal and signal["meta"]["confidence"] > 0.7:
                            opposite_direction = (
                                    (pos['direction'] == 'BUY' and signal['side'] == 'SELL') or
                                    (pos['direction'] == 'SELL' and signal['side'] == 'BUY')
                            )

                            if opposite_direction:
                                log.info(
                                    f"🔄 Opposite signal detected for {epic} with {signal['meta']['confidence']:.1%} confidence")
                                try:
                                    close_direction = "SELL" if pos['direction'] == 'BUY' else "BUY"
                                    resp = ig.close_position(
                                        deal_id=pos['deal_id'],
                                        direction=close_direction,
                                        size=pos['size']
                                    )
                                    log.info(f"✅ Position closed: {epic} - {resp.get('dealReference')}")
                                    position_manager.remove_position(epic, current_price, "AI_REVERSAL")
                                except Exception as e:
                                    log.error(f"❌ Failed to close {epic}: {e}")

                        last_bar_time[epic] = df.index[-1]
                        continue

                    # Try to open new position based on AI signal
                    if signal and epic not in position_manager.positions:
                        confidence = signal["meta"]["confidence"]
                        patterns = signal["meta"]["patterns_detected"]

                        log.info("=" * 60)
                        log.info(f"🎯 AI SIGNAL DETECTED: {epic}")
                        log.info(f"   Direction: {signal['side']}")
                        log.info(f"   Confidence: {confidence:.1%}")
                        log.info(f"   Patterns: {', '.join(patterns)}")
                        log.info(f"   Momentum: {signal['meta']['momentum']}")
                        log.info(f"   Trend Strength: {signal['meta']['trend_strength']:.2f}")
                        log.info("=" * 60)

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
                            log.warning(f"✗ Position size too small: {adj_size}")
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
                                log.info(f"📤 PLACING ORDER:")
                                log.info(f"   {epic} {direction}")
                                log.info(f"   Size: {adj_size}")
                                log.info(f"   Stop: {stop_pts:.2f} pts")
                                log.info(f"   Target: {tp_pts:.2f} pts")
                                log.info(f"   Risk/Reward: 1:{tp_pts / stop_pts:.2f}")

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
                                log.info(f"✅ ORDER FILLED: {deal_ref}")

                                # Track position
                                position_manager.add_position(
                                    epic=epic,
                                    deal_id=deal_ref,
                                    direction=direction,
                                    size=adj_size,
                                    entry_price=current_price,
                                    stop=stop_pts,
                                    tp=tp_pts,
                                    confidence=confidence,
                                    patterns=patterns
                                )

                            except Exception as e:
                                log.exception(f"❌ ORDER FAILED {epic}: {e}")
                                losing_trades += 1

                    last_bar_time[epic] = df.index[-1]

                except Exception as e:
                    log.error(f"Error processing {epic}: {e}")
                    continue

            # Sleep between analysis cycles
            time.sleep(15)  # 15 seconds between cycles

        except KeyboardInterrupt:
            log.info("⚠️ Interrupted by user")
            break
        except Exception as e:
            log.exception(f"Main loop error: {e}")
            time.sleep(30)

    # Shutdown - final report
    log.info("=" * 80)
    log.info("🛑 SHUTTING DOWN - Generating final report...")
    log.info("=" * 80)

    final_stats = position_manager.get_performance_stats()
    log.info(f"📊 FINAL STATISTICS:")
    log.info(f"   Total Trades: {final_stats.get('completed', 0)}")
    log.info(f"   Win Rate: {final_stats.get('win_rate', 0):.1f}%")
    log.info(f"   Wins: {final_stats.get('wins', 0)} | Losses: {final_stats.get('losses', 0)}")
    log.info(f"   Total P&L: {final_stats.get('total_pnl_pts', 0):+.2f} pts")
    log.info(f"   Profit Factor: {final_stats.get('profit_factor', 0):.2f}")

    final_report = save_analysis_report(position_manager, "final_ai_report.json")
    log.info(f"📄 Final report: {final_report}")
    log.info("=" * 80)


if __name__ == "__main__":
    main()
