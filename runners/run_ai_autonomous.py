"""
COMPLETE FIX for run_ai_autonomous.py

Key changes:
1. Fixed position syncing (correct IG API fields)
2. Added position monitoring (check if TP/SL hit)
3. Added position exit logic
4. Better error handling
5. One position per symbol at a time
6. AI Trading Bot with Trailing Stops
"""

import os
import sys
import time
import pandas as pd
import json
from datetime import datetime, UTC

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.config import load_settings
from core.logging_utils import setup_logging
from core.risk import size_by_invested_capital, daily_lockout
from data.ig_price_bars import bars_from_ig
from broker.ig_client import IGClient
from broker.order_exec import enforce_market_rules, estimate_pip_value
from strategy.ai_pattern_recognizer import AIPatternRecognizer
from data.multi_data_provider import create_data_aggregator


class TrailingStopManager:
    """Manages manual trailing stops for all positions"""

    def __init__(self, log):
        self.log = log
        self.trailing_stops = {}

    def initialize(self, epic, entry_price, direction, stop_distance,
                   activation_pct=0.3, trailing_pct=0.5):
        """
        Initialize trailing stop for new position

        Args:
            activation_pct: Profit % of stop needed to activate trailing (0.3 = 30%)
            trailing_pct: % of move to keep as stop buffer (0.5 = 50%)
        """
        self.trailing_stops[epic] = {
            'entry_price': entry_price,
            'direction': direction,
            'initial_stop_distance': stop_distance,
            'current_stop_level': (entry_price - stop_distance if direction == 'BUY'
                                   else entry_price + stop_distance),
            'best_price': entry_price,
            'trailing_pct': trailing_pct,
            'activation_distance': stop_distance * activation_pct,
            'active': False,
            'total_trailed': 0.0
        }

        self.log.info(f"🎯 Trailing stop initialized:")
        self.log.info(f"   Epic: {epic}")
        self.log.info(f"   Entry: {entry_price:.2f}")
        self.log.info(f"   Initial stop: {self.trailing_stops[epic]['current_stop_level']:.2f}")
        self.log.info(f"   Will activate after {self.trailing_stops[epic]['activation_distance']:.2f} pts profit")

    def update(self, epic, current_price):
        """
        Update trailing stop based on current price

        Returns:
            ('HIT', stop_level) if stop hit
            ('TRAILED', new_stop_level) if stop was moved
            (None, None) if no action
        """
        if epic not in self.trailing_stops:
            return None, None

        ts = self.trailing_stops[epic]
        direction = ts['direction']

        # Check if stop hit
        if direction == 'BUY':
            if current_price <= ts['current_stop_level']:
                self.log.info(f"🛑 TRAILING STOP HIT: {epic}")
                self.log.info(f"   Stop level: {ts['current_stop_level']:.2f}")
                self.log.info(f"   Current price: {current_price:.2f}")
                self.log.info(f"   Total profit protected: {current_price - ts['entry_price']:.2f} pts")
                return 'HIT', ts['current_stop_level']
        else:
            if current_price >= ts['current_stop_level']:
                self.log.info(f"🛑 TRAILING STOP HIT: {epic}")
                self.log.info(f"   Stop level: {ts['current_stop_level']:.2f}")
                self.log.info(f"   Current price: {current_price:.2f}")
                self.log.info(f"   Total profit protected: {ts['entry_price'] - current_price:.2f} pts")
                return 'HIT', ts['current_stop_level']

        # Update best price and check for trailing
        price_improved = False

        if direction == 'BUY':
            if current_price > ts['best_price']:
                ts['best_price'] = current_price
                price_improved = True
        else:
            if current_price < ts['best_price']:
                ts['best_price'] = current_price
                price_improved = True

        if not price_improved:
            return None, None

        # Check if trailing should activate
        if direction == 'BUY':
            profit = current_price - ts['entry_price']
        else:
            profit = ts['entry_price'] - current_price

        if not ts['active'] and profit >= ts['activation_distance']:
            ts['active'] = True
            self.log.info(f"✅ TRAILING ACTIVATED: {epic}")
            self.log.info(f"   Profit: {profit:.2f} pts (threshold: {ts['activation_distance']:.2f})")

        # Trail the stop if active
        if ts['active']:
            if direction == 'BUY':
                # New stop = current price - (initial stop * trailing_pct)
                new_stop = current_price - (ts['initial_stop_distance'] * ts['trailing_pct'])

                if new_stop > ts['current_stop_level']:
                    old_stop = ts['current_stop_level']
                    trail_amount = new_stop - old_stop
                    ts['current_stop_level'] = new_stop
                    ts['total_trailed'] += trail_amount

                    self.log.info(f"📈 STOP TRAILED: {epic}")
                    self.log.info(f"   {old_stop:.2f} → {new_stop:.2f} (+{trail_amount:.2f})")
                    self.log.info(f"   Current price: {current_price:.2f}")
                    self.log.info(f"   Total trailed: {ts['total_trailed']:.2f} pts")

                    return 'TRAILED', new_stop
            else:
                new_stop = current_price + (ts['initial_stop_distance'] * ts['trailing_pct'])

                if new_stop < ts['current_stop_level']:
                    old_stop = ts['current_stop_level']
                    trail_amount = old_stop - new_stop
                    ts['current_stop_level'] = new_stop
                    ts['total_trailed'] += trail_amount

                    self.log.info(f"📉 STOP TRAILED: {epic}")
                    self.log.info(f"   {old_stop:.2f} → {new_stop:.2f} (-{trail_amount:.2f})")
                    self.log.info(f"   Current price: {current_price:.2f}")
                    self.log.info(f"   Total trailed: {ts['total_trailed']:.2f} pts")

                    return 'TRAILED', new_stop

        return None, None

    def get_info(self, epic):
        """Get trailing stop info for position"""
        if epic not in self.trailing_stops:
            return None

        ts = self.trailing_stops[epic]
        return {
            'active': ts['active'],
            'current_stop': ts['current_stop_level'],
            'total_trailed': ts['total_trailed'],
            'best_price': ts['best_price']
        }

    def remove(self, epic):
        """Remove trailing stop"""
        if epic in self.trailing_stops:
            del self.trailing_stops[epic]


class PositionManager:
    """Enhanced position management with monitoring"""

    def __init__(self, log):
        self.log = log
        self.positions = {}
        self.trade_history = []
        self.decision_log = []

    def add_position(self, epic, deal_id, direction, size, entry_price, stop, tp, confidence, patterns):
        """Track new position"""
        self.positions[epic] = {
            'deal_id': deal_id,
            'direction': direction,
            'size': size,
            'entry_price': entry_price,
            'entry_time': datetime.now(UTC).isoformat(),
            'stop_distance': stop,  # Points from entry
            'tp_distance': tp,  # Points from entry
            'stop_level': entry_price - stop if direction == 'BUY' else entry_price + stop,
            'tp_level': entry_price + tp if direction == 'BUY' else entry_price - tp,
            'confidence': confidence,
            'patterns': patterns,
            'status': 'OPEN'
        }

        self.log.info(f"📝 Position tracked: {epic}")
        self.log.info(f"   Direction: {direction} @ {entry_price:.2f}")
        self.log.info(f"   Stop: {self.positions[epic]['stop_level']:.2f} ({stop:.2f} pts)")
        self.log.info(f"   Target: {self.positions[epic]['tp_level']:.2f} ({tp:.2f} pts)")
        self.log.info(f"   Confidence: {confidence:.1%}")

    def check_exit_conditions(self, epic, current_price):
        """Check if position should be closed (TP only, stops handled by trailing)"""
        if epic not in self.positions:
            return None, None

        pos = self.positions[epic]

        # Only check take-profit (stops managed by trailing stop manager)
        if pos['direction'] == 'BUY':
            if current_price >= pos['tp_level']:
                return 'EXIT', 'TAKE_PROFIT'
        else:
            if current_price <= pos['tp_level']:
                return 'EXIT', 'TAKE_PROFIT'

        return None, None

    def remove_position(self, epic, exit_price=None, reason="CLOSED"):
        """Remove position and log to history"""
        if epic not in self.positions:
            return

        pos = self.positions[epic]
        pos['exit_time'] = datetime.now(UTC).isoformat()
        pos['exit_price'] = exit_price
        pos['status'] = reason

        if exit_price:
            # Calculate P&L
            if pos['direction'] == 'BUY':
                pnl_pts = (exit_price - pos['entry_price']) * pos['size']
            else:
                pnl_pts = (pos['entry_price'] - exit_price) * pos['size']

            pos['pnl_pts'] = pnl_pts

            # Calculate duration
            from dateutil import parser
            entry_dt = parser.parse(pos['entry_time'])
            exit_dt = parser.parse(pos['exit_time'])
            duration = (exit_dt - entry_dt).total_seconds() / 60
            pos['duration_minutes'] = duration

            self.log.info(f"📊 Closed {epic}: {pnl_pts:+.2f} pts in {duration:.1f} min ({reason})")

        self.trade_history.append(pos)
        del self.positions[epic]

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


def sync_positions_from_broker(ig, position_manager, log):
    """
    FIXED: Sync positions from broker
    """
    try:
        positions_data = ig.positions()
        broker_positions = {}

        for pos in positions_data.get('positions', []):
            market = pos.get('market', {})
            position = pos.get('position', {})

            epic = market.get('epic')

            # ✅ FIXED: Correct field names
            broker_positions[epic] = {
                'deal_id': position.get('dealId'),
                'direction': position.get('direction'),
                'size': position.get('size'),  # ✅ Not 'dealSize'
                'open_level': position.get('level'),  # ✅ Not 'openLevel'
            }

        # Remove positions from manager that aren't in broker
        for epic in list(position_manager.positions.keys()):
            if epic not in broker_positions:
                position_manager.remove_position(epic, reason="BROKER_CLOSED")
                log.info(f"Position {epic} removed - closed at broker")

        log.info(f"✓ Synced {len(broker_positions)} positions")
        return broker_positions

    except Exception as e:
        log.error(f"Failed to sync positions: {e}")
        import traceback
        log.debug(traceback.format_exc())
        return {}


def monitor_open_positions(ig, position_manager, data_aggregator, log):
    """
    Monitor open positions with trailing stops
    """
    if not position_manager.positions:
        return

    log.info(f"🔍 Monitoring {len(position_manager.positions)} open positions...")

    for epic in list(position_manager.positions.keys()):
        try:
            # Get current price
            df = data_aggregator.get_bars(epic, timeframe="1min", limit=250)
            if df.empty:
                continue

            current_price = df['close'].iloc[-1]
            pos = position_manager.positions[epic]

            # Update trailing stop
            action, level = trailing_manager.update(epic, current_price)

            if action == 'HIT':
                # Trailing stop was hit - close position
                log.info(f"🎯 Trailing stop hit for {epic}")

                try:
                    close_direction = "SELL" if pos['direction'] == 'BUY' else "BUY"
                    resp = ig.close_position(
                        deal_id=pos['deal_id'],
                        direction=close_direction,
                        size=pos['size']
                    )

                    log.info(f"✅ Position closed: {resp.get('dealReference')}")
                    position_manager.remove_position(epic, current_price, "TRAILING_STOP")
                    trailing_manager.remove(epic)

                except Exception as e:
                    log.error(f"❌ Failed to close {epic}: {e}")

                continue

            # Check take-profit (stops handled by trailing)
            should_exit, reason = position_manager.check_exit_conditions(epic, current_price)

            if should_exit:
                pos = position_manager.positions[epic]

                log.info(f"🎯 {reason} triggered for {epic}")
                log.info(f"   Entry: {pos['entry_price']:.2f} → Current: {current_price:.2f}")

                try:
                    # Close position
                    close_direction = "SELL" if pos['direction'] == 'BUY' else "BUY"
                    resp = ig.close_position(
                        deal_id=pos['deal_id'],
                        direction=close_direction,
                        size=pos['size']
                    )

                    log.info(f"✅ Position closed: {resp.get('dealReference')}")
                    position_manager.remove_position(epic, current_price, reason)
                    trailing_manager.remove(epic)

                except Exception as e:
                    log.error(f"❌ Failed to close {epic}: {e}")
            else:
                # Log current status
                pnl_pts = ((current_price - pos['entry_price']) if pos['direction'] == 'BUY'
                          else (pos['entry_price'] - current_price)) * pos['size']

                # Get trailing stop info
                ts_info = trailing_manager.get_info(epic)

                if ts_info:
                    log.info(f"  {epic}: {current_price:.2f} ({pnl_pts:+.2f} pts)")
                    if ts_info['active']:
                        log.info(f"    Trailing: Active | Stop @ {ts_info['current_stop']:.2f} | Trailed {ts_info['total_trailed']:.2f} pts")
                    else:
                        log.info(f"    Trailing: Waiting for activation")

        except Exception as e:
            log.error(f"Error monitoring {epic}: {e}")


def save_analysis_report(position_manager, filename="ai_analysis_report.json"):
    """Save analysis report"""
    report = {
        "generated_at": datetime.now(UTC).isoformat(),
        "performance": position_manager.get_performance_stats(),
        "recent_trades": position_manager.trade_history[-20:],
        "decision_log": position_manager.decision_log[-50:]
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
    log.info("🤖 AI TRADING BOT WITH TRAILING STOPS")
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

    # Initialize data aggregator
    log.info("Initializing data provider...")
    aggregator = create_data_aggregator(
        ig_client=ig,
        alpha_vantage_key=cfg["alphavantage"]["api_key"],
        twelve_data_key=cfg["12data"]["api_key"],
    )
    log.info(f"✓ Data aggregator ready")

    # Get configuration
    epics = cfg["symbols"]
    timeframe = cfg.get("timeframe", "1min")
    invest = cfg["risk"]["invest_per_trade"]
    max_loss_pct = cfg["risk"]["max_loss_pct_invest"]
    max_daily_loss_pct = cfg["risk"]["max_daily_loss_pct"]

    # Trailing stop config
    use_trailing = cfg["execution"].get("use_trailing_stop", False)
    trailing_activation_pct = cfg["execution"].get("trailing_activation_pct", 0.3)
    trailing_distance_pct = cfg["execution"].get("trailing_distance_pct", 0.5)

    log.info(f"✓ Monitoring {len(epics)} instruments")
    log.info(f"✓ Timeframe: {timeframe}")
    log.info(f"✓ Risk per trade: £{invest} (max {max_loss_pct}% loss)")

    if use_trailing:
        log.info(f"✓ Trailing stops: ENABLED")
        log.info(f"  - Activation: {trailing_activation_pct*100:.0f}% of stop profit")
        log.info(f"  - Distance: {trailing_distance_pct*100:.0f}% of favorable move")
    else:
        log.info(f"✓ Trailing stops: DISABLED (fixed stops)")

    # Initialize AI Strategy
    ai_config = cfg.get("ai_strategy", {})
    strategy = AIPatternRecognizer(
        atr_period=ai_config.get("atr_period", 14),
        stop_multiplier=ai_config.get("stop_multiplier", 1.5),
        rr_take=ai_config.get("rr_take", 2.0),
        confidence_threshold=ai_config.get("confidence_threshold", 0.30),
        lookback_candles=ai_config.get("lookback_candles", 50),
        cfd_mode=ai_config.get("cfd_mode", True)
    )
    log.info(f"✓ AI Pattern Recognizer initialized")

    # Load market details
    market_cache = {}
    for e in epics:
        try:
            market_cache[e] = ig.market_details(e)
            log.info(f"✓ Market details loaded: {e}")
        except Exception as ex:
            log.error(f"✗ Failed to load {e}: {ex}")

    # Initialize managers
    position_manager = PositionManager(log)
    trailing_manager = TrailingStopManager(log)
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
    log.info("🚀 SYSTEM LIVE - Trading with trailing stops...")
    log.info("=" * 80)

    # Main trading loop
    while True:
        try:
            loop_count += 1

            # Kill switch
            if os.environ.get("KILL_SWITCH", "0") == "1":
                log.warning("⚠️ Kill switch activated")
                break

            # ✅ NEW: Monitor open positions every 5 loops (75 seconds)
            if loop_count % 5 == 0:
                monitor_open_positions(ig, position_manager, trailing_manager, aggregator, log)

            # Update P&L and sync every 10 loops
            if loop_count % 10 == 0:
                try:
                    acct = ig.account_summary()
                    current_equity = acct['accounts'][0]['balance']['balance']
                    daily_pnl_pct = ((current_equity - start_equity) / start_equity) * 100
                    log.info(f"💰 Daily P&L: {daily_pnl_pct:+.2f}% | Equity: £{current_equity:.2f}")
                except Exception as e:
                    log.error(f"Failed to update equity: {e}")

                sync_positions_from_broker(ig, position_manager, log)

            # Periodic reporting
            if time.time() - last_report_time > 300:
                stats = position_manager.get_performance_stats()
                log.info("=" * 80)
                log.info(f"📊 PERFORMANCE UPDATE")
                log.info(f"  Trades: {stats.get('completed', 0)} | Win Rate: {stats.get('win_rate', 0):.1f}%")
                log.info(f"  Total P&L: {stats.get('total_pnl_pts', 0):+.2f} pts")
                log.info(f"  Open Positions: {len(position_manager.positions)}")
                log.info("=" * 80)

                report_path = save_analysis_report(position_manager)
                log.info(f"📄 Report saved: {report_path}")
                last_report_time = time.time()

            # Check risk lockouts
            if daily_lockout(daily_pnl_pct, max_daily_loss_pct):
                log.warning(f"🛑 Daily loss limit: {daily_pnl_pct:.2f}%")
                time.sleep(600)
                continue

            # Process each instrument
            for epic in epics:
                try:
                    # ✅ RULE: Only one position per symbol at a time
                    if epic in position_manager.positions:
                        log.info(f"⏭️ Skipping {epic} - position already open")
                        continue

                    # Get price data
                    df = aggregator.get_bars(epic, timeframe, limit=250)

                    if df is None or df.empty or len(df) < 50:
                        continue

                    # Skip if bar hasn't updated
                    if last_bar_time[epic] is not None and df.index[-1] == last_bar_time[epic]:
                        continue

                    # Run AI analysis
                    log.info(f"🔍 Analyzing {epic}...")
                    signal = strategy.on_bar(df)

                    # Log decision
                    decision_entry = {
                        "timestamp": datetime.now(UTC).isoformat(),
                        "epic": epic,
                        "signal": signal is not None,
                        "confidence": signal["meta"]["confidence"] if signal else 0.0
                    }
                    position_manager.decision_log.append(decision_entry)

                    # Try to open new position
                    if signal:
                        confidence = signal["meta"]["confidence"]
                        patterns = signal["meta"]["patterns_detected"]

                        log.info("=" * 60)
                        log.info(f"🎯 AI SIGNAL: {epic} {signal['side']}")
                        log.info(f"   Confidence: {confidence:.1%}")
                        log.info(f"   Patterns: {', '.join(patterns) if patterns else 'None'}")
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
                            current_price = df['close'].iloc[-1]

                            try:
                                log.info(f"📤 PLACING ORDER: {epic} {direction}")
                                log.info(f"   Size: {adj_size} | Stop: {stop_pts:.2f} | TP: {tp_pts:.2f}")

                                if use_trailing:
                                    log.info(f"   Trailing: Will activate after {stop_pts * trailing_activation_pct:.2f} pts profit")

                                # Place order (NO stop if using manual trailing)
                                resp = ig.place_order(
                                    epic,
                                    direction,
                                    adj_size,
                                    stop_distance=None if use_trailing else stop_pts,
                                    limit_distance=tp_pts,
                                    tif=cfg["execution"]["time_in_force"]
                                )

                                deal_ref = resp.get('dealReference')
                                log.info(f"✅ ORDER FILLED: {deal_ref}")

                                # Get entry price
                                current_price = df['close'].iloc[-1]

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

                                # Initialize trailing stop
                                if use_trailing:
                                    trailing_manager.initialize(
                                        epic=epic,
                                        entry_price=current_price,
                                        direction=direction,
                                        stop_distance=stop_pts,
                                        activation_pct=trailing_activation_pct,
                                        trailing_pct=trailing_distance_pct
                                    )

                            except Exception as e:
                                log.exception(f"❌ ORDER FAILED {epic}: {e}")
                                losing_trades += 1

                    last_bar_time[epic] = df.index[-1]

                except Exception as e:
                    log.error(f"Error processing {epic}: {e}")
                    continue

            # Sleep between cycles
            time.sleep(15)

        except KeyboardInterrupt:
            log.info("⚠️ Interrupted by user")
            break
        except Exception as e:
            log.exception(f"Main loop error: {e}")
            time.sleep(30)

    # Shutdown
    log.info("=" * 80)
    log.info("🛑 SHUTTING DOWN")
    log.info("=" * 80)

    final_stats = position_manager.get_performance_stats()
    log.info(f"📊 FINAL STATISTICS:")
    log.info(f"   Total Trades: {final_stats.get('completed', 0)}")
    log.info(f"   Win Rate: {final_stats.get('win_rate', 0):.1f}%")
    log.info(f"   Total P&L: {final_stats.get('total_pnl_pts', 0):+.2f} pts")

    final_report = save_analysis_report(position_manager, "final_ai_report.json")
    log.info(f"📄 Final report: {final_report}")
    log.info("=" * 80)


if __name__ == "__main__":
    main()