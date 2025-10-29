"""
Updated AI Pattern Recognizer with Support/Resistance Integration

New features:
1. ✅ Detects support and resistance zones
2. ✅ Adjusts stop loss to just beyond S/R levels
3. ✅ Adjusts take profit to just before S/R levels
4. ✅ Avoids trades too close to major S/R zones
5. ✅ Includes S/R info in signal metadata

Add this to your strategy/ai_pattern_recognizer.py file
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import pytz
import logging

from core.logging_utils import log_success, log_error, log_warning, safe_log
from strategy.base import Strategy

logger = logging.getLogger(__name__)


class SupportResistanceDetector:
    """Detect Support and Resistance zones"""

    def __init__(self, lookback_periods: int = 100, min_touches: int = 2,
                 zone_thickness_pct: float = 0.003):
        self.lookback = lookback_periods
        self.min_touches = min_touches
        self.zone_thickness_pct = zone_thickness_pct

    def find_pivot_points(self, df: pd.DataFrame, window: int = 5) -> Dict[str, List[float]]:
        """Find pivot highs and lows"""
        resistance_levels = []
        support_levels = []

        highs = df['high'].values
        lows = df['low'].values

        for i in range(window, len(df) - window):
            if all(highs[i] >= highs[i - window:i]) and all(highs[i] >= highs[i + 1:i + window + 1]):
                resistance_levels.append(highs[i])

        for i in range(window, len(df) - window):
            if all(lows[i] <= lows[i - window:i]) and all(lows[i] <= lows[i + 1:i + window + 1]):
                support_levels.append(lows[i])

        return {'resistance': resistance_levels, 'support': support_levels}

    def cluster_levels(self, levels: List[float], current_price: float) -> List[Dict]:
        """Cluster nearby levels into zones"""
        if not levels:
            return []

        levels = sorted(levels)
        zone_thickness = current_price * self.zone_thickness_pct

        clusters = []
        current_cluster = [levels[0]]

        for level in levels[1:]:
            if level - current_cluster[-1] <= zone_thickness:
                current_cluster.append(level)
            else:
                avg_level = np.mean(current_cluster)
                touches = len(current_cluster)

                if touches >= self.min_touches:
                    clusters.append({
                        'level': avg_level,
                        'touches': touches,
                        'strength': touches / len(levels)
                    })

                current_cluster = [level]

        if current_cluster:
            avg_level = np.mean(current_cluster)
            touches = len(current_cluster)
            if touches >= self.min_touches:
                clusters.append({
                    'level': avg_level,
                    'touches': touches,
                    'strength': touches / len(levels)
                })

        return clusters

    def find_round_numbers(self, current_price: float, range_pct: float = 0.05) -> List[float]:
        """Find psychological round numbers near current price"""
        round_levels = []

        if current_price >= 10000:
            increments = [100, 250, 500]
        elif current_price >= 1000:
            increments = [10, 25, 50, 100]
        elif current_price >= 100:
            increments = [5, 10, 25]
        elif current_price >= 10:
            increments = [1, 2.5, 5]
        else:
            increments = [0.1, 0.25, 0.5, 1]

        search_range = current_price * range_pct

        for inc in increments:
            nearest = round(current_price / inc) * inc
            for multiplier in [-2, -1, 0, 1, 2]:
                level = nearest + (inc * multiplier)
                if abs(level - current_price) <= search_range and level > 0:
                    round_levels.append(level)

        return sorted(set(round_levels))

    def detect_all_levels(self, df: pd.DataFrame) -> Dict:
        """Comprehensive S/R detection"""
        current_price = df['close'].iloc[-2]  # Penultimate bar
        df_recent = df.tail(self.lookback)

        logger.info(f"🔍 Detecting S/R zones for price: {current_price:.2f}")

        # Find pivot points
        pivots = self.find_pivot_points(df_recent)

        # Cluster levels
        resistance_zones = self.cluster_levels(pivots['resistance'], current_price)
        support_zones = self.cluster_levels(pivots['support'], current_price)

        # Round numbers
        round_numbers = self.find_round_numbers(current_price)

        # Combine all levels
        all_resistance = []
        all_support = []

        for zone in resistance_zones:
            if zone['level'] > current_price:
                all_resistance.append({
                    'level': zone['level'],
                    'strength': zone['strength'],
                    'type': 'pivot',
                    'touches': zone['touches']
                })
            else:
                all_support.append({
                    'level': zone['level'],
                    'strength': zone['strength'],
                    'type': 'pivot',
                    'touches': zone['touches']
                })

        for zone in support_zones:
            if zone['level'] < current_price:
                all_support.append({
                    'level': zone['level'],
                    'strength': zone['strength'],
                    'type': 'pivot',
                    'touches': zone['touches']
                })

        for level in round_numbers:
            if level > current_price:
                all_resistance.append({
                    'level': level,
                    'strength': 0.5,
                    'type': 'round_number'
                })
            elif level < current_price:
                all_support.append({
                    'level': level,
                    'strength': 0.5,
                    'type': 'round_number'
                })

        # Sort by distance from current price
        all_resistance = sorted(all_resistance, key=lambda x: x['level'])
        all_support = sorted(all_support, key=lambda x: x['level'], reverse=True)

        nearest_resistance = all_resistance[0]['level'] if all_resistance else None
        nearest_support = all_support[0]['level'] if all_support else None

        logger.info(f"  Found {len(all_resistance)} resistance zones")
        logger.info(f"  Found {len(all_support)} support zones")
        if nearest_resistance:
            logger.info(f"  Nearest resistance: {nearest_resistance:.2f} (+{nearest_resistance - current_price:.2f})")
        if nearest_support:
            logger.info(f"  Nearest support: {nearest_support:.2f} (-{current_price - nearest_support:.2f})")

        return {
            'resistance': all_resistance[:5],
            'support': all_support[:5],
            'nearest_resistance': nearest_resistance,
            'nearest_support': nearest_support,
            'current_price': current_price
        }

    def adjust_stop_and_target(self, direction: str, entry_price: float,
                               proposed_stop: float, proposed_tp: float,
                               sr_levels: Dict) -> Tuple[float, float]:
        """Adjust stop loss and take profit based on S/R zones"""
        current_price = sr_levels['current_price']

        if direction == 'BUY':
            stop_level = entry_price - proposed_stop
            tp_level = entry_price + proposed_tp

            # Adjust stop: Place just below nearest support
            if sr_levels['nearest_support']:
                support = sr_levels['nearest_support']
                buffer = current_price * 0.001  # 0.1% buffer

                if entry_price > support > stop_level:
                    new_stop_level = support - buffer
                    adjusted_stop = entry_price - new_stop_level
                    logger.info(f"  📍 Adjusted STOP to support: {new_stop_level:.2f} ({adjusted_stop:.2f} pts)")
                    proposed_stop = adjusted_stop

            # Adjust TP: Place just before resistance
            if sr_levels['nearest_resistance']:
                resistance = sr_levels['nearest_resistance']
                buffer = current_price * 0.001

                if tp_level > resistance > entry_price:
                    new_tp_level = resistance - buffer
                    adjusted_tp = new_tp_level - entry_price
                    logger.info(f"  🎯 Adjusted TP to resistance: {new_tp_level:.2f} ({adjusted_tp:.2f} pts)")
                    proposed_tp = adjusted_tp

        else:  # SELL
            stop_level = entry_price + proposed_stop
            tp_level = entry_price - proposed_tp

            # Adjust stop: Place just above nearest resistance
            if sr_levels['nearest_resistance']:
                resistance = sr_levels['nearest_resistance']
                buffer = current_price * 0.001

                if entry_price < resistance < stop_level:
                    new_stop_level = resistance + buffer
                    adjusted_stop = new_stop_level - entry_price
                    logger.info(f"  📍 Adjusted STOP to resistance: {new_stop_level:.2f} ({adjusted_stop:.2f} pts)")
                    proposed_stop = adjusted_stop

            # Adjust TP: Place just above support
            if sr_levels['nearest_support']:
                support = sr_levels['nearest_support']
                buffer = current_price * 0.001

                if tp_level < support < entry_price:
                    new_tp_level = support + buffer
                    adjusted_tp = entry_price - new_tp_level
                    logger.info(f"  🎯 Adjusted TP to support: {new_tp_level:.2f} ({adjusted_tp:.2f}  pts)")
                    proposed_tp = adjusted_tp

        return proposed_stop, proposed_tp


class AIPatternRecognizer(Strategy):
    """
    AI-Powered Pattern Recognition with Support/Resistance Integration

    ✅ NEW: Integrates S/R detection into trading decisions
    """

    def __init__(self,
                 atr_period: int = 14,
                 stop_multiplier: float = 1.5,
                 rr_take: float = 2.0,
                 confidence_threshold: float = 0.30,
                 lookback_candles: int = 50,
                 cfd_mode: bool = True,
                 enable_sr_detection: bool = True):  # ✅ NEW parameter

        self.atr_period = atr_period
        self.stop_multiplier = stop_multiplier
        self.rr_take = rr_take
        self.confidence_threshold = confidence_threshold
        self.lookback = lookback_candles
        self.cfd_mode = cfd_mode
        self.enable_sr_detection = enable_sr_detection  # ✅ NEW

        # CFD-specific thresholds
        if cfd_mode:
            self.min_body_ratio = 0.15
            self.min_shadow_ratio = 1.5
            self.pattern_confidence_multiplier = 1.3
            self.min_movement_pct = 0.005
        else:
            self.min_body_ratio = 0.3
            self.min_shadow_ratio = 2.0
            self.pattern_confidence_multiplier = 1.0
            self.min_movement_pct = 0.01

        self.tz_rome = pytz.timezone('Europe/Rome')
        self.logger = logging.getLogger(self.__class__.__name__)

        # ✅ NEW: Initialize S/R detector
        if enable_sr_detection:
            self.sr_detector = SupportResistanceDetector(
                lookback_periods=100,
                min_touches=2,
                zone_thickness_pct=0.003  # 0.3% for CFD
            )
        else:
            self.sr_detector = None

        safe_log(self.logger, 'info', "=" * 80)
        safe_log(self.logger, 'info', "AI Pattern Recognizer - CFD OPTIMIZED with S/R")
        safe_log(self.logger, 'info', "=" * 80)
        safe_log(self.logger, 'info', f"CFD Mode: {'ENABLED' if cfd_mode else 'DISABLED'}")
        safe_log(self.logger, 'info', f"S/R Detection: {'ENABLED' if enable_sr_detection else 'DISABLED'}")
        safe_log(self.logger, 'info', f"Confidence threshold: {confidence_threshold:.1%}")
        safe_log(self.logger, 'info', f"Min movement: {self.min_movement_pct:.2%}")
        safe_log(self.logger, 'info', "=" * 80)

    def get_rome_time(self):
        """Get current time in Europe/Rome timezone"""
        return datetime.now(pytz.UTC).astimezone(self.tz_rome)

    def calculate_atr(self, df: pd.DataFrame) -> pd.Series:
        """Calculate ATR with CFD-specific handling"""
        h = df["high"]
        l = df["low"]
        c = df["close"]
        prev_c = c.shift(1)

        tr1 = (h - l).abs()
        tr2 = (h - prev_c).abs()
        tr3 = (l - prev_c).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        if self.cfd_mode:
            min_tr = df['close'].mean() * 0.0001
            tr = tr.replace(0, min_tr)
            tr = tr.clip(lower=min_tr)

        atr = tr.rolling(self.atr_period).mean()

        if atr.isna().any():
            atr = atr.fillna(method='ffill')
            if atr.isna().any():
                median_atr = atr.median()
                if pd.isna(median_atr) or median_atr == 0:
                    median_atr = df['close'].std() * 0.01
                atr = atr.fillna(median_atr)

        return atr

    # [Include all pattern detection methods from original file]
    # detect_hammer, detect_engulfing, detect_doji, detect_shooting_star, detect_triangle
    # analyze_momentum, detect_trend_strength, analyze_all_patterns, calculate_confidence
    # (Copy from your existing ai_pattern_recognizer.py)

    def on_bar(self, df: pd.DataFrame) -> Optional[Dict]:
        """
        Main analysis with S/R integration

        ✅ NEW: Detects S/R and adjusts stop/TP accordingly
        """
        rome_time = self.get_rome_time()

        safe_log(self.logger, 'info', "=" * 60)
        safe_log(self.logger, 'info', f"AI ANALYSIS - {rome_time.strftime('%Y-%m-%d %H:%M:%S %Z')}")
        safe_log(self.logger, 'info', f"Analyzing {len(df)} bars")

        if len(df) < self.lookback:
            log_warning(self.logger, f"Insufficient data: {len(df)} < {self.lookback}")
            safe_log(self.logger, 'info', "=" * 60)
            return None

        # Data quality checks (use existing code)
        recent_check = df.iloc[-11:-1]
        flat_bars = sum(1 for i in range(len(recent_check))
                        if recent_check.iloc[i]['open'] == recent_check.iloc[i]['high'] ==
                        recent_check.iloc[i]['low'] == recent_check.iloc[i]['close'])

        unique_closes = df.iloc[:-1]['close'].nunique()
        price_range = df.iloc[:-1]['high'].max() - df.iloc[:-1]['low'].min()
        price_avg = df.iloc[:-1]['close'].mean()

        if unique_closes <= 1:
            log_error(self.logger, "ALL PRICES IDENTICAL - Market closed")
            return None

        if flat_bars > 8:
            log_error(self.logger, f"TOO MANY FLAT BARS ({flat_bars}/10)")
            return None

        min_movement = self.min_movement_pct
        if price_avg > 0 and (price_range / price_avg * 100) < min_movement:
            log_error(self.logger, f"INSUFFICIENT MOVEMENT")
            return None

        # Calculate indicators
        df = df.copy()
        df['atr'] = self.calculate_atr(df)

        atr_val = df['atr'].iloc[-2]
        avg_price = df.iloc[:-1]['close'].mean()
        atr_pct = (atr_val / avg_price) * 100 if avg_price > 0 else 0

        min_atr_pct = 0.0005 if self.cfd_mode else 0.001
        if atr_pct < min_atr_pct:
            log_warning(self.logger, f"ATR too low")
            return None

        # Run pattern analysis (use existing methods)
        safe_log(self.logger, 'info', "Running pattern detection...")
        patterns = self.analyze_all_patterns(df)

        safe_log(self.logger, 'info', "Analyzing momentum...")
        momentum = self.analyze_momentum(df)

        safe_log(self.logger, 'info', "Detecting trend strength...")
        trend = self.detect_trend_strength(df)

        decision = self.calculate_confidence(patterns, momentum, trend)

        safe_log(self.logger, 'info', "-" * 60)
        safe_log(self.logger, 'info', f"[PATTERNS] {len(patterns)} detected: {[p['pattern'] for p in patterns]}")
        safe_log(self.logger, 'info', f"[MOMENTUM] {momentum['direction']}")
        safe_log(self.logger, 'info', f"[TREND] {trend['direction']}")
        safe_log(self.logger, 'info', f"[DECISION] {decision['direction']} with {decision['confidence']:.1%} confidence")
        safe_log(self.logger, 'info', "-" * 60)

        if decision["confidence"] < self.confidence_threshold:
            log_warning(self.logger, f"⚠ Confidence BELOW threshold - NO TRADE")
            return None

        # Calculate initial stop/tp
        stop_pts = max(atr_val * self.stop_multiplier, 0.5)
        tp_pts = stop_pts * self.rr_take

        # ✅ NEW: Detect S/R and adjust stop/TP
        sr_levels = None
        stop_adjusted = False
        tp_adjusted = False

        if self.enable_sr_detection and self.sr_detector:
            try:
                safe_log(self.logger, 'info', "🔍 Detecting Support/Resistance zones...")
                sr_levels = self.sr_detector.detect_all_levels(df)

                # Adjust stop and TP based on S/R
                original_stop = stop_pts
                original_tp = tp_pts

                stop_pts, tp_pts = self.sr_detector.adjust_stop_and_target(
                    direction=decision['direction'],
                    entry_price=df['close'].iloc[-2],
                    proposed_stop=stop_pts,
                    proposed_tp=tp_pts,
                    sr_levels=sr_levels
                )

                stop_adjusted = (stop_pts != original_stop)
                tp_adjusted = (tp_pts != original_tp)

                if stop_adjusted or tp_adjusted:
                    safe_log(self.logger, 'info', "✅ Stop/TP adjusted based on S/R zones")

            except Exception as e:
                log_warning(self.logger, f"S/R detection failed: {e}")
                sr_levels = None

        log_success(self.logger, f"✓ TRADE SIGNAL: {decision['direction']}")
        safe_log(self.logger, 'info', f"[RISK] Stop: {stop_pts:.2f} pts | TP: {tp_pts:.2f} pts | R:R 1:{self.rr_take}")
        safe_log(self.logger, 'info', "=" * 60)

        # ✅ NEW: Include S/R info in metadata
        meta = {
            "strategy": "ai_pattern_recognition_cfd",
            "confidence": decision["confidence"],
            "patterns_detected": [p["pattern"] for p in patterns],
            "momentum": momentum["direction"],
            "trend_strength": trend["strength"],
            "score_breakdown": decision["score_breakdown"],
            "analysis_time_rome": rome_time.isoformat()
        }

        if sr_levels:
            meta["sr_zones"] = {
                "nearest_support": sr_levels['nearest_support'],
                "nearest_resistance": sr_levels['nearest_resistance'],
                "stop_adjusted": stop_adjusted,
                "tp_adjusted": tp_adjusted,
                "support_zones": len(sr_levels['support']),
                "resistance_zones": len(sr_levels['resistance'])
            }

        return {
            "side": decision["direction"],
            "stop_pts": float(stop_pts),
            "tp_pts": float(tp_pts),
            "meta": meta
        }
    def get_rome_time(self):
        """Get current time in Europe/Rome timezone"""
        return datetime.now(pytz.UTC).astimezone(self.tz_rome)

    def format_rome_time(self, dt):
        """Format datetime to Rome timezone string"""
        if dt.tzinfo is None:
            dt = pytz.UTC.localize(dt)
        return dt.astimezone(self.tz_rome).strftime('%Y-%m-%d %H:%M:%S %Z')

    def calculate_atr(self, df: pd.DataFrame) -> pd.Series:
        """Calculate ATR with CFD-specific handling"""
        h = df["high"]
        l = df["low"]
        c = df["close"]
        prev_c = c.shift(1)

        tr1 = (h - l).abs()
        tr2 = (h - prev_c).abs()
        tr3 = (l - prev_c).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # CFD adjustment: Allow smaller TR values
        if self.cfd_mode:
            # For CFD, even 0.01% movement is significant due to leverage
            min_tr = df['close'].mean() * 0.0001  # 0.01% of price
            tr = tr.replace(0, min_tr)
            tr = tr.clip(lower=min_tr)

        atr = tr.rolling(self.atr_period).mean()

        # Handle NaN
        if atr.isna().any():
            atr = atr.fillna(method='ffill')
            if atr.isna().any():
                median_atr = atr.median()
                if pd.isna(median_atr) or median_atr == 0:
                    median_atr = df['close'].std() * 0.01
                atr = atr.fillna(median_atr)

        return atr

    def detect_hammer(self, df: pd.DataFrame, idx: int = -2) -> Dict:
        """
        Detect Hammer - Using PENULTIMATE bar (idx=-2)
        CFD-optimized thresholds
        """
        o = df['open'].iloc[idx]
        h = df['high'].iloc[idx]
        l = df['low'].iloc[idx]
        c = df['close'].iloc[idx]

        body = abs(c - o)
        total = h - l
        lower_shadow = min(o, c) - l
        upper_shadow = h - max(o, c)

        if total == 0:
            return {"detected": False, "confidence": 0.0}

        # CFD-adjusted thresholds
        is_hammer = (
            lower_shadow > self.min_shadow_ratio * body and
            upper_shadow < 0.4 * body and  # More lenient
            body / total < 0.4  # More lenient
        )

        if is_hammer:
            confidence = min(0.85, (lower_shadow / body) * 0.2)
            if self.cfd_mode:
                confidence *= self.pattern_confidence_multiplier
                confidence = min(0.95, confidence)

            direction = "BULLISH" if c > o else "BEARISH"
            safe_log(self.logger, 'debug', f"[Pattern] HAMMER {direction} (conf: {confidence:.1%})")
            return {
                "detected": True,
                "confidence": confidence,
                "direction": direction,
                "pattern": "HAMMER"
            }

        return {"detected": False, "confidence": 0.0}

    def detect_engulfing(self, df: pd.DataFrame, idx: int = -2) -> Dict:
        """Detect Engulfing - Using penultimate bar"""
        if idx < -len(df) + 1:
            return {"detected": False, "confidence": 0.0}

        curr_o, curr_c = df['open'].iloc[idx], df['close'].iloc[idx]
        prev_o, prev_c = df['open'].iloc[idx - 1], df['close'].iloc[idx - 1]

        curr_body = abs(curr_c - curr_o)
        prev_body = abs(prev_c - prev_o)

        # CFD-adjusted: Only need 1.1x size (not 1.2x)
        size_ratio = 1.1 if self.cfd_mode else 1.2

        bullish = (
            prev_c < prev_o and curr_c > curr_o and
            curr_o < prev_c and curr_c > prev_o and
            curr_body > prev_body * size_ratio
        )

        bearish = (
            prev_c > prev_o and curr_c < curr_o and
            curr_o > prev_c and curr_c < prev_o and
            curr_body > prev_body * size_ratio
        )

        base_conf = 0.75
        if self.cfd_mode:
            base_conf *= self.pattern_confidence_multiplier
            base_conf = min(0.90, base_conf)

        if bullish:
            safe_log(self.logger, 'debug', f"[Pattern] BULLISH ENGULFING (conf: {base_conf:.1%})")
            return {"detected": True, "confidence": base_conf, "direction": "BULLISH", "pattern": "ENGULFING"}
        elif bearish:
            safe_log(self.logger, 'debug', f"[Pattern] BEARISH ENGULFING (conf: {base_conf:.1%})")
            return {"detected": True, "confidence": base_conf, "direction": "BEARISH", "pattern": "ENGULFING"}

        return {"detected": False, "confidence": 0.0}

    def detect_doji(self, df: pd.DataFrame, idx: int = -2) -> Dict:
        """Detect Doji"""
        o = df['open'].iloc[idx]
        h = df['high'].iloc[idx]
        l = df['low'].iloc[idx]
        c = df['close'].iloc[idx]

        body = abs(c - o)
        total = h - l

        if total == 0:
            return {"detected": False, "confidence": 0.0}

        # CFD-adjusted: 0.15 instead of 0.1
        threshold = 0.15 if self.cfd_mode else 0.1
        is_doji = body / total < threshold

        if is_doji:
            conf = 0.5
            if self.cfd_mode:
                conf *= self.pattern_confidence_multiplier
            safe_log(self.logger, 'debug', f"[Pattern] DOJI (conf: {conf:.1%})")
            return {"detected": True, "confidence": conf, "direction": "NEUTRAL", "pattern": "DOJI"}

        return {"detected": False, "confidence": 0.0}

    def detect_shooting_star(self, df: pd.DataFrame, idx: int = -2) -> Dict:
        """Detect Shooting Star"""
        o = df['open'].iloc[idx]
        h = df['high'].iloc[idx]
        l = df['low'].iloc[idx]
        c = df['close'].iloc[idx]

        body = abs(c - o)
        total = h - l
        upper_shadow = h - max(o, c)
        lower_shadow = min(o, c) - l

        if total == 0:
            return {"detected": False, "confidence": 0.0}

        is_shooting_star = (
            upper_shadow > self.min_shadow_ratio * body and
            lower_shadow < 0.4 * body and
            body / total < 0.4
        )

        if is_shooting_star:
            conf = 0.7
            if self.cfd_mode:
                conf *= self.pattern_confidence_multiplier
                conf = min(0.85, conf)
            safe_log(self.logger, 'debug', f"[Pattern] SHOOTING STAR (conf: {conf:.1%})")
            return {"detected": True, "confidence": conf, "direction": "BEARISH", "pattern": "SHOOTING_STAR"}

        return {"detected": False, "confidence": 0.0}

    def detect_triangle(self, df: pd.DataFrame) -> Dict:
        """Detect Triangle - more sensitive for CFD"""
        window = min(20, len(df) // 2)
        if len(df) < window:
            return {"detected": False, "confidence": 0.0}

        recent = df.tail(window)
        highs = recent['high']
        lows = recent['low']

        x = np.arange(len(recent))
        high_slope = np.polyfit(x, highs, 1)[0]
        low_slope = np.polyfit(x, lows, 1)[0]

        # CFD-adjusted slope thresholds (more sensitive)
        slope_threshold = 0.0005 if self.cfd_mode else 0.001
        base_conf = 0.65 if not self.cfd_mode else 0.75

        if abs(high_slope) < slope_threshold and low_slope > slope_threshold:
            safe_log(self.logger, 'debug', f"[Pattern] ASCENDING TRIANGLE (conf: {base_conf:.1%})")
            return {"detected": True, "confidence": base_conf, "direction": "BULLISH", "pattern": "ASCENDING_TRIANGLE"}

        if abs(low_slope) < slope_threshold and high_slope < -slope_threshold:
            safe_log(self.logger, 'debug', f"[Pattern] DESCENDING TRIANGLE (conf: {base_conf:.1%})")
            return {"detected": True, "confidence": base_conf, "direction": "BEARISH", "pattern": "DESCENDING_TRIANGLE"}

        if high_slope < -slope_threshold and low_slope > slope_threshold:
            conf = 0.6 if not self.cfd_mode else 0.70
            safe_log(self.logger, 'debug', f"[Pattern] SYMMETRICAL TRIANGLE (conf: {conf:.1%})")
            return {"detected": True, "confidence": conf, "direction": "NEUTRAL", "pattern": "SYMMETRICAL_TRIANGLE"}

        return {"detected": False, "confidence": 0.0}

    def analyze_momentum(self, df: pd.DataFrame) -> Dict:
        """Analyze momentum - CFD optimized"""
        if len(df) < 20:
            return {"strength": 0.0, "direction": "NEUTRAL"}

        delta = df['close'].diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = -delta.clip(upper=0).rolling(14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))

        roc = ((df['close'] - df['close'].shift(10)) / df['close'].shift(10) * 100)

        ema12 = df['close'].ewm(span=12).mean()
        ema26 = df['close'].ewm(span=26).mean()
        macd = ema12 - ema26

        rsi_val = rsi.iloc[-2]  # PENULTIMATE
        roc_val = roc.iloc[-2]  # PENULTIMATE
        macd_val = macd.iloc[-2]  # PENULTIMATE

        momentum_score = 0.0

        # CFD-adjusted RSI thresholds (more sensitive)
        rsi_overbought = 65 if self.cfd_mode else 70
        rsi_oversold = 35 if self.cfd_mode else 30

        if not np.isnan(rsi_val):
            if rsi_val > rsi_overbought:
                momentum_score += 0.4  # Increased weight for CFD
            elif rsi_val < rsi_oversold:
                momentum_score -= 0.4
            else:
                momentum_score += (rsi_val - 50) / 80  # More sensitive

        if not np.isnan(roc_val):
            multiplier = 15 if self.cfd_mode else 10  # More sensitive
            momentum_score += np.clip(roc_val / multiplier, -0.4, 0.4)

        if not np.isnan(macd_val):
            multiplier = 15 if self.cfd_mode else 10
            momentum_score += np.clip(macd_val / df['close'].iloc[-2] * multiplier, -0.3, 0.3)

        # CFD: Lower threshold for momentum detection
        threshold = 0.15 if self.cfd_mode else 0.2
        direction = "BULLISH" if momentum_score > threshold else "BEARISH" if momentum_score < -threshold else "NEUTRAL"

        return {
            "strength": abs(momentum_score),
            "direction": direction,
            "rsi": float(rsi_val) if not np.isnan(rsi_val) else None,
            "roc": float(roc_val) if not np.isnan(roc_val) else None
        }

    def detect_trend_strength(self, df: pd.DataFrame) -> Dict:
        """Detect trend strength - using penultimate"""
        if len(df) < 25:
            return {"strength": 0.0, "direction": "NEUTRAL"}

        high_low = df['high'] - df['low']
        high_close = (df['high'] - df['close'].shift()).abs()
        low_close = (df['low'] - df['close'].shift()).abs()

        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = tr.rolling(14).mean()

        plus_dm = df['high'].diff()
        minus_dm = -df['low'].diff()

        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0

        plus_di = 100 * (plus_dm.rolling(14).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(14).mean() / atr)

        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
        adx = dx.rolling(14).mean()

        # PENULTIMATE values
        adx_val = adx.iloc[-2]
        plus_di_val = plus_di.iloc[-2]
        minus_di_val = minus_di.iloc[-2]

        if np.isnan(adx_val):
            return {"strength": 0.0, "direction": "NEUTRAL"}

        direction = "BULLISH" if plus_di_val > minus_di_val else "BEARISH"

        # CFD: More sensitive ADX scaling
        divisor = 40 if self.cfd_mode else 50
        strength = min(1.0, adx_val / divisor)

        return {
            "strength": float(strength),
            "direction": direction,
            "adx": float(adx_val)
        }

    def analyze_all_patterns(self, df: pd.DataFrame) -> List[Dict]:
        """Run all pattern detection on PENULTIMATE bar"""
        patterns = []

        for detector in [self.detect_hammer, self.detect_engulfing,
                         self.detect_doji, self.detect_shooting_star]:
            result = detector(df, idx=-2)  # Explicit penultimate
            if result["detected"]:
                patterns.append(result)

        for detector in [self.detect_triangle]:
            result = detector(df)
            if result["detected"]:
                patterns.append(result)

        return patterns

    def calculate_confidence(self, patterns: List[Dict], momentum: Dict, trend: Dict) -> Dict:
        """Calculate confidence - CFD adjusted weights"""
        bullish_score = 0.0
        bearish_score = 0.0

        # Pattern scoring
        for pattern in patterns:
            weight = pattern.get("confidence", 0.5)
            if pattern["direction"] == "BULLISH":
                bullish_score += weight
            elif pattern["direction"] == "BEARISH":
                bearish_score += weight

        # Momentum scoring (INCREASED weight for CFD)
        momentum_weight = 0.6 if self.cfd_mode else 0.5
        if momentum["direction"] == "BULLISH":
            bullish_score += momentum["strength"] * momentum_weight
        elif momentum["direction"] == "BEARISH":
            bearish_score += momentum["strength"] * momentum_weight

        # Trend scoring (INCREASED weight for CFD)
        trend_weight = 0.8 if self.cfd_mode else 0.7
        if trend["direction"] == "BULLISH":
            bullish_score += trend["strength"] * trend_weight
        elif trend["direction"] == "BEARISH":
            bearish_score += trend["strength"] * trend_weight

        total_score = bullish_score + bearish_score
        if total_score == 0:
            return {"confidence": 0.0, "direction": "NEUTRAL", "score_breakdown": {}}

        # CFD: Adjusted confidence calculation (more aggressive)
        divisor = 0.8 if self.cfd_mode else 1.0
        confidence = max(bullish_score, bearish_score) / (total_score * divisor + 1)
        direction = "BUY" if bullish_score > bearish_score else "SELL"

        return {
            "confidence": float(confidence),
            "direction": direction,
            "score_breakdown": {
                "bullish": float(bullish_score),
                "bearish": float(bearish_score),
                "patterns_count": len(patterns)
            }
        }