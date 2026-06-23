"""
COMPLETE FIXED VERSION: run_ai_autonomous.py

Key fixes:
1. ✅ Trailing stops now work correctly (using broker API updates)
2. ✅ Initial order placed with FIXED stop (IG requirement)
3. ✅ Stop is then trailed manually via position updates
4. ✅ Support/Resistance zones integrated into AI analysis

Changes from original:
- TrailingStopManager now updates stops at broker via API
- Orders placed with fixed stop initially, then trailed
- Pass deal_id to trailing manager for broker updates
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
from strategy.fvg_strategy import FVGStrategy
from data.multi_data_provider import create_data_aggregator
import logging

logger = logging.getLogger(__name__)


class TrailingStopManager:
    """Enhanced Trailing Stop Manager with IG Broker Integration"""

    def __init__(self, ig_client, log):
        self.ig_client = ig_client
        self.log = log
        self.trailing_stops = {}

    def initialize(self, epic, deal_id, entry_price, direction, stop_distance,
                   activation_pct=0.3, trailing_pct=0.5):
        """Initialize trailing stop with deal ID for broker updates"""
        initial_stop_level = (entry_price - stop_distance if direction == 'BUY'
                             else entry_price + stop_distance)

        self.trailing_stops[epic] = {
            'deal_id': deal_id,  # ✅ Store deal ID
            'entry_price': entry_price,
            'direction': direction,
            'initial_stop_distance': stop_distance,
            'current_stop_level': initial_stop_level,
            'best_price': entry_price,
            'trailing_pct': trailing_pct,
            'activation_distance': stop_distance * activation_pct,
            'active': False,
            'total_trailed': 0.0
        }

        self.log.info(f"🎯 Trailing stop initialized:")
        self.log.info(f"   Epic: {epic}")
        self.log.info(f"   Deal ID: {deal_id}")
        self.log.info(f"   Entry: {entry_price:.2f}")
        self.log.info(f"   Initial stop: {initial_stop_level:.2f}")
        self.log.info(f"   Will activate after {self.trailing_stops[epic]['activation_distance']:.2f} pts profit")

    def update_stop_at_broker(self, epic, new_stop_level):
        """
        ✅ Update stop level at IG broker via API
        """
        try:
            ts = self.trailing_stops[epic]
            deal_id = ts['deal_id']

            # Use the proper IG client method
            result = self.ig_client.update_position(
                deal_id=deal_id,
                stop_level=new_stop_level
            )

            self.log.info(f"✅ Stop updated at broker: {new_stop_level:.2f}")
            return True

        except Exception as e:
            self.log.error(f"❌ Error updating stop at broker: {e}")
            return False

    def update(self, epic, current_price):
        """Update trailing stop and sync with broker"""
        if epic not in self.trailing_stops:
            return None, None

        ts = self.trailing_stops[epic]
        direction = ts['direction']

        # Check if stop hit
        if direction == 'BUY':
            if current_price <= ts['current_stop_level']:
                self.log.info(f"🛑 TRAILING STOP HIT: {epic}")
                self.log.info(f"   Stop: {ts['current_stop_level']:.2f} | Price: {current_price:.2f}")
                return 'HIT', ts['current_stop_level']
        else:
            if current_price >= ts['current_stop_level']:
                self.log.info(f"🛑 TRAILING STOP HIT: {epic}")
                self.log.info(f"   Stop: {ts['current_stop_level']:.2f} | Price: {current_price:.2f}")
                return 'HIT', ts['current_stop_level']

        # Update best price
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

        # Check activation
        profit = (current_price - ts['entry_price'] if direction == 'BUY'
                 else ts['entry_price'] - current_price)

        if not ts['active'] and profit >= ts['activation_distance']:
            ts['active'] = True
            self.log.info(f"✅ TRAILING ACTIVATED: {epic}")
            self.log.info(f"   Profit: {profit:.2f} pts (threshold: {ts['activation_distance']:.2f})")

        # Trail the stop if active
        if ts['active']:
            if direction == 'BUY':
                new_stop = current_price - (ts['initial_stop_distance'] * ts['trailing_pct'])

                if new_stop > ts['current_stop_level']:
                    old_stop = ts['current_stop_level']
                    trail_amount = new_stop - old_stop

                    # ✅ Update at broker
                    if self.update_stop_at_broker(epic, new_stop):
                        ts['current_stop_level'] = new_stop
                        ts['total_trailed'] += trail_amount

                        self.log.info(f"📈 STOP TRAILED: {epic}")
                        self.log.info(f"   {old_stop:.2f} → {new_stop:.2f} (+{trail_amount:.2f})")
                        self.log.info(f"   Total trailed: {ts['total_trailed']:.2f} pts")

                        return 'TRAILED', new_stop
            else:
                new_stop = current_price + (ts['initial_stop_distance'] * ts['trailing_pct'])

                if new_stop < ts['current_stop_level']:
                    old_stop = ts['current_stop_level']
                    trail_amount = old_stop - new_stop

                    # ✅ Update at broker
                    if self.update_stop_at_broker(epic, new_stop):
                        ts['current_stop_level'] = new_stop
                        ts['total_trailed'] += trail_amount

                        self.log.info(f"📉 STOP TRAILED: {epic}")
                        self.log.info(f"   {old_stop:.2f} → {new_stop:.2f} (-{trail_amount:.2f})")
                        self.log.info(f"   Total trailed: {ts['total_trailed']:.2f} pts")

                        return 'TRAILED', new_stop

        return None, None

    def get_info(self, epic):
        """Get trailing stop info"""
        if epic not in self.trailing_stops:
            return None

        ts = self.trailing_stops[epic]
        return {
            'active': ts['active'],
            'current_stop': ts['current_stop_level'],
            'total_trailed': ts['total_trailed'],
            'best_price': ts['best_price'],
            'deal_id': ts['deal_id']
        }

    def remove(self, epic):
        """Remove trailing stop"""
        if epic in self.trailing_stops:
            del self.trailing_stops[epic]


class PositionManager:
    """Position management with monitoring and persistent trade history"""

    TRADE_HISTORY_FILE = "data/trade_history.json"

    def __init__(self, log):
        self.log = log
        self.positions = {}
        self.trade_history = self._load_trade_history()
        self.decision_log = []

    def _load_trade_history(self):
        """Load trade history from disk."""
        try:
            import json
            with open(self.TRADE_HISTORY_FILE, 'r') as f:
                history = json.load(f)
                self.log.info(f"📂 Loaded {len(history)} trades from history")
                return history
        except (FileNotFoundError, json.JSONDecodeError):
            return []

    def _save_trade_history(self):
        """Persist trade history to disk."""
        try:
            import json
            os.makedirs(os.path.dirname(self.TRADE_HISTORY_FILE), exist_ok=True)
            with open(self.TRADE_HISTORY_FILE, 'w') as f:
                json.dump(self.trade_history, f, indent=2, default=str)
        except Exception as e:
            self.log.error(f"Failed to save trade history: {e}")

    def add_position(self, epic, deal_id, direction, size, entry_price, stop, tp, confidence, patterns):
        """Track new position"""
        self.positions[epic] = {
            'deal_id': deal_id,
            'direction': direction,
            'size': size,
            'entry_price': entry_price,
            'entry_time': datetime.now(UTC).isoformat(),
            'stop_distance': stop,
            'tp_distance': tp,
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
        """Check if position should be closed (TP only)"""
        if epic not in self.positions:
            return None, None

        pos = self.positions[epic]
        tp_level = pos.get('tp_level')

        # Skip TP check if no level set (e.g., restored from broker without limit)
        if tp_level is None:
            return None, None

        # Check take-profit
        if pos['direction'] == 'BUY':
            if current_price >= tp_level:
                return 'EXIT', 'TAKE_PROFIT'
        else:
            if current_price <= tp_level:
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
            if pos['direction'] == 'BUY':
                pnl_pts = (exit_price - pos['entry_price']) * pos['size']
            else:
                pnl_pts = (pos['entry_price'] - exit_price) * pos['size']

            pos['pnl_pts'] = pnl_pts

            from dateutil import parser
            entry_dt = parser.parse(pos['entry_time'])
            exit_dt = parser.parse(pos['exit_time'])
            duration = (exit_dt - entry_dt).total_seconds() / 60
            pos['duration_minutes'] = duration

            self.log.info(f"📊 Closed {epic}: {pnl_pts:+.2f} pts in {duration:.1f} min ({reason})")

        self.trade_history.append(pos)
        self._save_trade_history()
        del self.positions[epic]

    def get_performance_stats(self):
        """Calculate performance statistics for today and all-time."""
        from zoneinfo import ZoneInfo
        rome_tz = ZoneInfo("Europe/Rome")
        today_str = datetime.now(rome_tz).strftime("%Y-%m-%d")

        # All completed trades
        completed = [t for t in self.trade_history if 'pnl_pts' in t]

        # Today's trades (by exit_time in Rome timezone)
        today_trades = []
        for t in completed:
            try:
                from dateutil import parser
                exit_dt = parser.parse(t['exit_time'])
                if exit_dt.astimezone(rome_tz).strftime("%Y-%m-%d") == today_str:
                    today_trades.append(t)
            except (KeyError, ValueError, TypeError):
                continue

        def calc_stats(trades):
            if not trades:
                return {"completed": 0, "wins": 0, "losses": 0, "win_rate": 0,
                        "total_pnl_pts": 0, "avg_win_pts": 0, "avg_loss_pts": 0, "profit_factor": 0}
            wins = [t for t in trades if t['pnl_pts'] > 0]
            losses = [t for t in trades if t['pnl_pts'] <= 0]
            total_pnl = sum(t['pnl_pts'] for t in trades)
            win_rate = len(wins) / len(trades) * 100 if trades else 0
            avg_win = sum(t['pnl_pts'] for t in wins) / len(wins) if wins else 0
            avg_loss = sum(t['pnl_pts'] for t in losses) / len(losses) if losses else 0
            return {
                "completed": len(trades),
                "wins": len(wins),
                "losses": len(losses),
                "win_rate": win_rate,
                "total_pnl_pts": total_pnl,
                "avg_win_pts": avg_win,
                "avg_loss_pts": avg_loss,
                "profit_factor": abs(avg_win / avg_loss) if avg_loss != 0 else 0,
            }

        return {
            "total_trades": len(self.trade_history),
            "today": calc_stats(today_trades),
            "alltime": calc_stats(completed),
        }


def sync_positions_from_broker(ig, position_manager, log):
    """Sync positions from broker — adds missing positions and removes stale ones."""
    try:
        positions_data = ig.positions()
        broker_positions = {}

        for pos in positions_data.get('positions', []):
            market = pos.get('market', {})
            position = pos.get('position', {})

            epic = market.get('epic')

            broker_positions[epic] = {
                'deal_id': position.get('dealId'),
                'direction': position.get('direction'),
                'size': position.get('size'),
                'open_level': position.get('level'),
                'stop_level': position.get('stopLevel'),
                'limit_level': position.get('limitLevel'),
            }

        # Remove locally-tracked positions that no longer exist at broker
        for epic in list(position_manager.positions.keys()):
            if epic not in broker_positions:
                position_manager.remove_position(epic, reason="BROKER_CLOSED")
                log.info(f"Position {epic} removed - closed at broker")

        # Add broker positions that aren't tracked locally (e.g., after restart)
        for epic, bp in broker_positions.items():
            if epic not in position_manager.positions:
                position_manager.positions[epic] = {
                    'deal_id': bp['deal_id'],
                    'direction': bp['direction'],
                    'size': bp['size'],
                    'entry_price': bp['open_level'],
                    'entry_time': datetime.now(UTC).isoformat(),
                    'stop_distance': 0,
                    'tp_distance': 0,
                    'stop_level': bp.get('stop_level'),
                    'tp_level': bp.get('limit_level'),
                    'confidence': 0,
                    'patterns': [],
                    'status': 'OPEN',
                }
                log.info(f"📥 Restored position from broker: {epic} "
                         f"{bp['direction']} @ {bp['open_level']} "
                         f"(dealId: {bp['deal_id']})")

        log.info(f"✓ Synced {len(broker_positions)} positions")
        return broker_positions

    except Exception as e:
        log.error(f"Failed to sync positions: {e}")
        return {}


def monitor_open_positions(ig, position_manager, trailing_manager, data_aggregator, log):
    """Monitor open positions with trailing stops using live IG price data"""
    if not position_manager.positions:
        return

    log.info(f"🔍 Monitoring {len(position_manager.positions)} open positions...")

    for epic in list(position_manager.positions.keys()):
        try:
            # Use IG market prices for real-time data (not cached bar data)
            try:
                market = ig.market_details(epic)
                bid = market.get('snapshot', {}).get('bid')
                offer = market.get('snapshot', {}).get('offer')
                pos = position_manager.positions[epic]

                if bid and offer:
                    # Use bid for SELL positions (closing a SELL = buying at offer)
                    # Use offer for BUY positions (closing a BUY = selling at bid)
                    current_price = float(bid) if pos['direction'] == 'BUY' else float(offer)
                else:
                    # Fallback to bar data if market snapshot unavailable
                    df = data_aggregator.get_bars(epic, timeframe="1min", limit=250)
                    if df is None or df.empty:
                        continue
                    current_price = df['close'].iloc[-1]
            except Exception as e:
                # Fallback to bar data
                df = data_aggregator.get_bars(epic, timeframe="1min", limit=250)
                if df is None or df.empty:
                    continue
                current_price = df['close'].iloc[-1]

            pos = position_manager.positions[epic]

            # Update trailing stop
            action, level = trailing_manager.update(epic, current_price)

            if action == 'HIT':
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

            # Check take-profit
            should_exit, reason = position_manager.check_exit_conditions(epic, current_price)

            if should_exit:
                pos = position_manager.positions[epic]

                log.info(f"🎯 {reason} triggered for {epic}")
                log.info(f"   Entry: {pos['entry_price']:.2f} → Current: {current_price:.2f}")

                try:
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
                pnl_pts = ((current_price - pos['entry_price']) if pos['direction'] == 'BUY'
                          else (pos['entry_price'] - current_price))

                ts_info = trailing_manager.get_info(epic)

                if ts_info:
                    log.info(f"  {epic}: {current_price:.2f} (P&L: {pnl_pts:+.2f} pts)")
                    if ts_info['active']:
                        log.info(f"    Trailing: Active | Stop @ {ts_info['current_stop']:.2f} | Trailed {ts_info['total_trailed']:.2f} pts")
                    else:
                        log.info(f"    Trailing: Waiting | Need {trailing_manager.trailing_stops[epic]['activation_distance']:.2f} pts profit")

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
    log.info("🤖 AI TRADING BOT WITH FIXED TRAILING STOPS")
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
        alpha_vantage_key=cfg.get("alphavantage", {}).get("api_key"),
        twelve_data_key=cfg.get("12data", {}).get("api_key"),
        finnhub_key=cfg.get("finnhub", {}).get("api_key"),
        timeframe=cfg.get("timeframe", "5min"),
    )
    log.info(f"✓ Data aggregator ready")

    # Get configuration
    epics = cfg["symbols"]
    timeframe = cfg.get("timeframe", "5min")   # default 5min = 288 bars/24h
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
        log.info(f"✓ Trailing stops: DISABLED")

    # Initialize Strategy (configurable via strategy_type in YAML)
    strategy_type = cfg.get("strategy_type", "ai_pattern")  # "ai_pattern", "fvg", or "hybrid"
    ai_config = cfg.get("ai_strategy", {})

    # Always create AI Pattern Recognizer (used as primary or fallback)
    ai_strategy = AIPatternRecognizer(
        atr_period=ai_config.get("atr_period", 14),
        stop_multiplier=ai_config.get("stop_multiplier", 1.5),
        rr_take=ai_config.get("rr_take", 2.0),
        confidence_threshold=ai_config.get("confidence_threshold", 0.30),
        lookback_candles=ai_config.get("lookback_candles", 250),
        cfd_mode=ai_config.get("cfd_mode", True)
    )

    fvg_strategy_instance = None
    if strategy_type in ("fvg", "hybrid"):
        fvg_config = cfg.get("fvg_strategy", {})
        fvg_strategy_instance = FVGStrategy(
            config=fvg_config,
            data_provider=aggregator,
            symbol_epic=epics[0],
        )
        log.info(f"✓ FVG Multi-Timeframe Strategy initialized")
        log.info(f"  - Cycle interval: {fvg_config.get('cycle_interval_seconds', 300)}s")
        log.info(f"  - Timeframes: {fvg_config.get('timeframes', ['60min', '15min', '5min'])}")
        log.info(f"  - Min confidence: {fvg_config.get('min_bias_confidence', 0.6)}")

    if strategy_type == "fvg":
        strategy = fvg_strategy_instance
        log.info(f"✓ Strategy: FVG only (no fallback)")
    elif strategy_type == "hybrid":
        strategy = fvg_strategy_instance  # Primary — fallback handled in loop
        log.info(f"✓ Strategy: FVG primary + AI Pattern fallback")
    else:
        strategy = ai_strategy
        log.info(f"✓ Strategy: AI Pattern Recognizer")

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
    trailing_manager = TrailingStopManager(ig_client=ig, log=log)  # ✅ Pass IG client
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
    last_sl_time = {}  # Track last stop loss time per epic for cooldown
    cooldown_after_sl = ai_config.get("cooldown_after_sl", 600)  # Default 10 min

    log.info("=" * 80)
    log.info("🚀 SYSTEM LIVE - Trading with fixed trailing stops...")
    log.info("=" * 80)

    # Main trading loop
    while True:
        try:
            loop_count += 1

            if os.environ.get("KILL_SWITCH", "0") == "1":
                log.warning("⚠️ Kill switch activated")
                break

            # Sync positions from broker every loop, BEFORE epic processing
            positions_before_sync = set(position_manager.positions.keys())
            broker_positions = sync_positions_from_broker(ig, position_manager, log)
            positions_after_sync = set(position_manager.positions.keys())

            # Track cooldown for positions closed by broker (SL hit)
            closed_by_broker = positions_before_sync - positions_after_sync
            for epic in closed_by_broker:
                last_sl_time[epic] = time.time()
                log.info(f"⏸️ Cooldown started for {epic} ({cooldown_after_sl}s)")

            # Initialize trailing stops for positions restored from broker (e.g., after restart)
            if use_trailing and broker_positions:
                for epic, bp in broker_positions.items():
                    if epic in position_manager.positions and epic not in trailing_manager.trailing_stops:
                        pos = position_manager.positions[epic]
                        stop_level = bp.get('stop_level')
                        entry = bp.get('open_level', pos.get('entry_price', 0))
                        if stop_level and entry:
                            stop_distance = abs(entry - stop_level)
                            trailing_manager.initialize(
                                epic=epic,
                                deal_id=bp['deal_id'],
                                entry_price=entry,
                                direction=bp['direction'],
                                stop_distance=stop_distance,
                                activation_pct=trailing_activation_pct,
                                trailing_pct=trailing_distance_pct
                            )

            # Monitor open positions every loop (no-op when no positions exist)
            monitor_open_positions(ig, position_manager, trailing_manager, aggregator, log)

            # Update P&L every 10 loops (equity check at reduced frequency)
            if loop_count % 10 == 0:
                try:
                    acct = ig.account_summary()
                    current_equity = acct['accounts'][0]['balance']['balance']
                    daily_pnl_pct = ((current_equity - start_equity) / start_equity) * 100
                    log.info(f"💰 Daily P&L: {daily_pnl_pct:+.2f}% | Equity: £{current_equity:.2f}")
                except Exception as e:
                    log.error(f"Failed to update equity: {e}")

            # Periodic reporting
            if time.time() - last_report_time > 300:
                stats = position_manager.get_performance_stats()
                today = stats.get('today', {})
                alltime = stats.get('alltime', {})

                log.info("=" * 80)
                log.info(f"📊 PERFORMANCE UPDATE")
                log.info(f"  TODAY: {today.get('completed', 0)} trades | "
                         f"Win Rate: {today.get('win_rate', 0):.1f}% | "
                         f"P&L: {today.get('total_pnl_pts', 0):+.2f} pts")
                if today.get('completed', 0) > 0:
                    log.info(f"    Wins: {today.get('wins', 0)} ({today.get('avg_win_pts', 0):+.2f} avg) | "
                             f"Losses: {today.get('losses', 0)} ({today.get('avg_loss_pts', 0):+.2f} avg)")
                log.info(f"  ALL-TIME: {alltime.get('completed', 0)} trades | "
                         f"Win Rate: {alltime.get('win_rate', 0):.1f}% | "
                         f"P&L: {alltime.get('total_pnl_pts', 0):+.2f} pts")
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
                    if epic in position_manager.positions:
                        log.info(f"⏭️ Skipping {epic} - position already open")
                        continue

                    # Cooldown after stop loss — prevent re-entry too quickly
                    if epic in last_sl_time:
                        elapsed = time.time() - last_sl_time[epic]
                        if elapsed < cooldown_after_sl:
                            remaining = int(cooldown_after_sl - elapsed)
                            log.info(f"⏸️ Cooldown {epic} - {remaining}s remaining after SL")
                            continue

                    df = aggregator.get_bars(epic, timeframe, limit=250)

                    if df is None or df.empty or len(df) < 50:
                        continue

                    if last_bar_time[epic] is not None and df.index[-1] == last_bar_time[epic]:
                        continue

                    log.info(f"🔍 Analyzing {epic}...")
                    signal = strategy.on_bar(df)

                    # Hybrid mode: fallback to AI Pattern Recognizer if FVG returns no signal
                    if signal is None and strategy_type == "hybrid" and ai_strategy is not None:
                        signal = ai_strategy.on_bar(df)
                        if signal:
                            log.info(f"🔄 FVG neutral — AI Pattern fallback triggered")

                    decision_entry = {
                        "timestamp": datetime.now(UTC).isoformat(),
                        "epic": epic,
                        "signal": signal is not None,
                        "confidence": signal["meta"]["confidence"] if signal else 0.0
                    }
                    position_manager.decision_log.append(decision_entry)

                    if signal:
                        confidence = signal["meta"]["confidence"]
                        patterns = signal["meta"]["patterns_detected"]

                        log.info("=" * 60)
                        log.info(f"🎯 AI SIGNAL: {epic} {signal['side']}")
                        log.info(f"   Confidence: {confidence:.1%}")
                        log.info(f"   Patterns: {', '.join(patterns) if patterns else 'None'}")

                        # Show S/R info if available
                        if 'sr_zones' in signal.get('meta', {}):
                            sr = signal['meta']['sr_zones']
                            if sr.get('nearest_support'):
                                log.info(f"   Support: {sr['nearest_support']:.2f}")
                            if sr.get('nearest_resistance'):
                                log.info(f"   Resistance: {sr['nearest_resistance']:.2f}")
                            if sr.get('stop_adjusted'):
                                log.info(f"   ✓ Stop adjusted to S/R")
                            if sr.get('tp_adjusted'):
                                log.info(f"   ✓ TP adjusted to S/R")

                        log.info("=" * 60)

                        if epic not in market_cache:
                            log.error(f"✗ No market details for {epic}")
                            last_bar_time[epic] = df.index[-1]
                            continue

                        mkt = market_cache[epic]
                        pip_value = estimate_pip_value(mkt)

                        proposed_size, max_loss = size_by_invested_capital(
                            invest_amount_gbp=invest,
                            max_loss_pct=max_loss_pct,
                            stop_pts=signal["stop_pts"],
                            pip_value_per_contract=pip_value,
                            min_size=mkt["dealingRules"]["minDealSize"]["value"],
                            size_step=0.1
                        )

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

                                # ✅ FIX: Place order with FIXED stop (IG requirement)
                                resp = ig.place_order(
                                    epic,
                                    direction,
                                    adj_size,
                                    stop_distance=stop_pts,  # ✅ Fixed stop
                                    limit_distance=tp_pts,
                                    tif=cfg["execution"]["time_in_force"]
                                )

                                deal_ref = resp.get('dealReference')
                                log.info(f"✅ ORDER PLACED: {deal_ref}")

                                # Get actual dealId from confirmation
                                try:
                                    confirm = ig.confirm_deal(deal_ref)
                                    deal_id = confirm.get('dealId', deal_ref)
                                    deal_status = confirm.get('dealStatus', 'UNKNOWN')
                                    log.info(f"✅ CONFIRMED: dealId={deal_id} | status={deal_status}")
                                except Exception as e:
                                    log.warning(f"⚠️ Could not confirm deal, using dealReference: {e}")
                                    deal_id = deal_ref

                                current_price = df['close'].iloc[-1]

                                position_manager.add_position(
                                    epic=epic,
                                    deal_id=deal_id,
                                    direction=direction,
                                    size=adj_size,
                                    entry_price=current_price,
                                    stop=stop_pts,
                                    tp=tp_pts,
                                    confidence=confidence,
                                    patterns=patterns
                                )

                                # ✅ FIX: Initialize trailing stop with deal_id
                                if use_trailing:
                                    trailing_manager.initialize(
                                        epic=epic,
                                        deal_id=deal_id,  # ✅ Pass deal ID
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

            # Sleep between cycles — adaptive based on number of active symbols
            # The TwelveData provider internally calculates optimal interval
            # to never exceed 800/day and 8/min regardless of symbol count
            num_symbols = len(epics)
            # Formula: (symbols * 86400) / 720 budget, minimum 60s
            poll_interval = max(60, (num_symbols * 86400) / 720)
            time.sleep(poll_interval)

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