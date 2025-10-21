"""
FIXED: AI Pattern Recognizer with Working Logging
The issue: logger wasn't properly initialized in the class

Replace the on_bar method and add __init__ logging in:
strategy/ai_pattern_recognizer.py
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional

# CRITICAL: Import logging utilities
from core.logging_utils import log_success, log_error, log_warning, safe_log
import logging

from strategy.base import Strategy

# Create module-level logger
logger = logging.getLogger(__name__)


class AIPatternRecognizer(Strategy):
    """AI-Powered Pattern Recognition Strategy with comprehensive logging"""

    def __init__(self,
                 atr_period: int = 14,
                 stop_multiplier: float = 1.5,
                 rr_take: float = 2.0,
                 confidence_threshold: float = 0.65,
                 lookback_candles: int = 50):
        self.atr_period = atr_period
        self.stop_multiplier = stop_multiplier
        self.rr_take = rr_take
        self.confidence_threshold = confidence_threshold
        self.lookback = lookback_candles

        # Initialize logger for this instance
        self.logger = logging.getLogger(self.__class__.__name__)

        # Log initialization
        safe_log(self.logger, 'info', "=" * 80)
        safe_log(self.logger, 'info', "AI Pattern Recognizer Initialized")
        safe_log(self.logger, 'info', "=" * 80)
        safe_log(self.logger, 'info', f"Confidence threshold: {confidence_threshold:.1%}")
        safe_log(self.logger, 'info', f"Stop multiplier: {stop_multiplier}x ATR")
        safe_log(self.logger, 'info', f"Risk/Reward: 1:{rr_take}")
        safe_log(self.logger, 'info', f"Lookback candles: {lookback_candles}")
        safe_log(self.logger, 'info', "=" * 80)

    def calculate_atr(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate Average True Range with flat data handling
        Returns ATR series, handling cases with minimal price movement
        """
        h = df["high"]
        l = df["low"]
        c = df["close"]
        prev_c = c.shift(1)

        # Calculate True Range components
        tr1 = (h - l).abs()
        tr2 = (h - prev_c).abs()
        tr3 = (l - prev_c).abs()

        # Take maximum of the three
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # Check if TR is all zeros or near-zero (flat data)
        if (tr > 0).sum() < len(tr) * 0.3:  # Less than 30% have movement
            safe_log(self.logger, 'warning',
                     f"Flat data detected: {(tr == 0).sum()}/{len(tr)} bars have zero True Range")

            # For completely flat data, use a tiny default value
            # This allows ATR calculation but signals should be rejected later
            tr = tr.replace(0, 0.0001)  # Tiny value instead of zero

        # Calculate ATR
        atr = tr.rolling(self.atr_period).mean()

        # If still NaN at the end (shouldn't happen now), forward fill
        if atr.isna().any():
            # Forward fill NaN values
            atr = atr.fillna(method='ffill')

            # If still NaN at start (before rolling window), use median
            if atr.isna().any():
                median_atr = atr.median()
                if pd.isna(median_atr) or median_atr == 0:
                    median_atr = df['close'].std() * 0.01  # 1% of price std as fallback
                atr = atr.fillna(median_atr)

        # Final sanity check
        if (atr <= 0).any():
            safe_log(self.logger, 'warning',
                     "ATR contains zero/negative values, replacing with minimum positive value")
            min_positive = atr[atr > 0].min() if (atr > 0).any() else 0.0001
            atr = atr.replace(0, min_positive)
            atr = atr.clip(lower=min_positive)

        return atr

    def detect_hammer(self, df: pd.DataFrame, idx: int = -1) -> Dict:
        """Detect Hammer pattern"""
        o, h, l, c = df['open'].iloc[idx], df['high'].iloc[idx], \
            df['low'].iloc[idx], df['close'].iloc[idx]

        body = abs(c - o)
        total = h - l
        lower_shadow = min(o, c) - l
        upper_shadow = h - max(o, c)

        if total == 0:
            return {"detected": False, "confidence": 0.0}

        is_hammer = (
                lower_shadow > 2 * body and
                upper_shadow < 0.3 * body and
                body / total < 0.3
        )

        if is_hammer:
            confidence = min(0.8, (lower_shadow / body) * 0.15)
            direction = "BULLISH" if c > o else "BEARISH"
            safe_log(self.logger, 'debug', f"[Pattern] HAMMER {direction} (conf: {confidence:.1%})")
            return {
                "detected": True,
                "confidence": confidence,
                "direction": direction,
                "pattern": "HAMMER"
            }

        return {"detected": False, "confidence": 0.0}

    def detect_engulfing(self, df: pd.DataFrame, idx: int = -1) -> Dict:
        """Detect Engulfing pattern"""
        if idx < -len(df) + 1:
            return {"detected": False, "confidence": 0.0}

        curr_o, curr_c = df['open'].iloc[idx], df['close'].iloc[idx]
        prev_o, prev_c = df['open'].iloc[idx - 1], df['close'].iloc[idx - 1]

        curr_body = abs(curr_c - curr_o)
        prev_body = abs(prev_c - prev_o)

        bullish = (
                prev_c < prev_o and curr_c > curr_o and
                curr_o < prev_c and curr_c > prev_o and
                curr_body > prev_body * 1.2
        )

        bearish = (
                prev_c > prev_o and curr_c < curr_o and
                curr_o > prev_c and curr_c < prev_o and
                curr_body > prev_body * 1.2
        )

        if bullish:
            safe_log(self.logger, 'debug', "[Pattern] BULLISH ENGULFING (conf: 75%)")
            return {"detected": True, "confidence": 0.75, "direction": "BULLISH", "pattern": "ENGULFING"}
        elif bearish:
            safe_log(self.logger, 'debug', "[Pattern] BEARISH ENGULFING (conf: 75%)")
            return {"detected": True, "confidence": 0.75, "direction": "BEARISH", "pattern": "ENGULFING"}

        return {"detected": False, "confidence": 0.0}

    def detect_doji(self, df: pd.DataFrame, idx: int = -1) -> Dict:
        """Detect Doji pattern"""
        o, h, l, c = df['open'].iloc[idx], df['high'].iloc[idx], \
            df['low'].iloc[idx], df['close'].iloc[idx]

        body = abs(c - o)
        total = h - l

        if total == 0:
            return {"detected": False, "confidence": 0.0}

        is_doji = body / total < 0.1

        if is_doji:
            safe_log(self.logger, 'debug', "[Pattern] DOJI (indecision)")
            return {"detected": True, "confidence": 0.5, "direction": "NEUTRAL", "pattern": "DOJI"}

        return {"detected": False, "confidence": 0.0}

    def detect_shooting_star(self, df: pd.DataFrame, idx: int = -1) -> Dict:
        """Detect Shooting Star pattern"""
        o, h, l, c = df['open'].iloc[idx], df['high'].iloc[idx], \
            df['low'].iloc[idx], df['close'].iloc[idx]

        body = abs(c - o)
        total = h - l
        upper_shadow = h - max(o, c)
        lower_shadow = min(o, c) - l

        if total == 0:
            return {"detected": False, "confidence": 0.0}

        is_shooting_star = (
                upper_shadow > 2 * body and
                lower_shadow < 0.3 * body and
                body / total < 0.3
        )

        if is_shooting_star:
            safe_log(self.logger, 'debug', "[Pattern] SHOOTING STAR (conf: 70%)")
            return {"detected": True, "confidence": 0.7, "direction": "BEARISH", "pattern": "SHOOTING_STAR"}

        return {"detected": False, "confidence": 0.0}

    def detect_double_top_bottom(self, df: pd.DataFrame) -> Dict:
        """Detect Double Top/Bottom patterns"""
        window = min(20, len(df) // 3)
        if len(df) < window * 2:
            return {"detected": False, "confidence": 0.0}

        highs = df['high'].rolling(window=5).max()
        lows = df['low'].rolling(window=5).min()

        peaks = []
        troughs = []

        for i in range(window, len(df) - window):
            if highs.iloc[i] == df['high'].iloc[i - window:i + window].max():
                peaks.append((i, highs.iloc[i]))
            if lows.iloc[i] == df['low'].iloc[i - window:i + window].min():
                troughs.append((i, lows.iloc[i]))

        if len(peaks) >= 2:
            last_two = peaks[-2:]
            price_diff = abs(last_two[0][1] - last_two[1][1]) / last_two[0][1]
            time_diff = last_two[1][0] - last_two[0][0]

            if price_diff < 0.02 and window < time_diff < window * 2:
                safe_log(self.logger, 'debug', "[Pattern] DOUBLE TOP (conf: 70%)")
                return {"detected": True, "confidence": 0.7, "direction": "BEARISH", "pattern": "DOUBLE_TOP"}

        if len(troughs) >= 2:
            last_two = troughs[-2:]
            price_diff = abs(last_two[0][1] - last_two[1][1]) / last_two[0][1]
            time_diff = last_two[1][0] - last_two[0][0]

            if price_diff < 0.02 and window < time_diff < window * 2:
                safe_log(self.logger, 'debug', "[Pattern] DOUBLE BOTTOM (conf: 70%)")
                return {"detected": True, "confidence": 0.7, "direction": "BULLISH", "pattern": "DOUBLE_BOTTOM"}

        return {"detected": False, "confidence": 0.0}

    def detect_triangle(self, df: pd.DataFrame) -> Dict:
        """Detect Triangle patterns"""
        window = min(20, len(df) // 2)
        if len(df) < window:
            return {"detected": False, "confidence": 0.0}

        recent = df.tail(window)
        highs = recent['high']
        lows = recent['low']

        x = np.arange(len(recent))
        high_slope = np.polyfit(x, highs, 1)[0]
        low_slope = np.polyfit(x, lows, 1)[0]

        if abs(high_slope) < 0.001 and low_slope > 0.001:
            safe_log(self.logger, 'debug', "[Pattern] ASCENDING TRIANGLE")
            return {"detected": True, "confidence": 0.65, "direction": "BULLISH", "pattern": "ASCENDING_TRIANGLE"}

        if abs(low_slope) < 0.001 and high_slope < -0.001:
            safe_log(self.logger, 'debug', "[Pattern] DESCENDING TRIANGLE")
            return {"detected": True, "confidence": 0.65, "direction": "BEARISH", "pattern": "DESCENDING_TRIANGLE"}

        if high_slope < -0.001 and low_slope > 0.001:
            safe_log(self.logger, 'debug', "[Pattern] SYMMETRICAL TRIANGLE")
            return {"detected": True, "confidence": 0.5, "direction": "NEUTRAL", "pattern": "SYMMETRICAL_TRIANGLE"}

        return {"detected": False, "confidence": 0.0}

    def detect_head_shoulders(self, df: pd.DataFrame) -> Dict:
        """Detect Head & Shoulders pattern"""
        window = min(30, len(df) // 2)
        if len(df) < window:
            return {"detected": False, "confidence": 0.0}

        recent = df.tail(window)
        highs = recent['high'].values

        peaks = []
        for i in range(5, len(highs) - 5):
            if highs[i] == max(highs[i - 5:i + 6]):
                peaks.append((i, highs[i]))

        if len(peaks) >= 3:
            last_three = peaks[-3:]
            if (last_three[1][1] > last_three[0][1] and
                    last_three[1][1] > last_three[2][1] and
                    abs(last_three[0][1] - last_three[2][1]) / last_three[0][1] < 0.05):
                safe_log(self.logger, 'debug', "[Pattern] HEAD & SHOULDERS (conf: 75%)")
                return {"detected": True, "confidence": 0.75, "direction": "BEARISH", "pattern": "HEAD_SHOULDERS"}

        return {"detected": False, "confidence": 0.0}

    def analyze_momentum(self, df: pd.DataFrame) -> Dict:
        """Analyze momentum"""
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

        rsi_val = rsi.iloc[-1]
        roc_val = roc.iloc[-1]
        macd_val = macd.iloc[-1]

        momentum_score = 0.0

        if not np.isnan(rsi_val):
            if rsi_val > 70:
                momentum_score += 0.3
            elif rsi_val < 30:
                momentum_score -= 0.3
            else:
                momentum_score += (rsi_val - 50) / 100

        if not np.isnan(roc_val):
            momentum_score += np.clip(roc_val / 10, -0.3, 0.3)

        if not np.isnan(macd_val):
            momentum_score += np.clip(macd_val / df['close'].iloc[-1] * 10, -0.2, 0.2)

        direction = "BULLISH" if momentum_score > 0.2 else "BEARISH" if momentum_score < -0.2 else "NEUTRAL"

        return {
            "strength": abs(momentum_score),
            "direction": direction,
            "rsi": float(rsi_val) if not np.isnan(rsi_val) else None,
            "roc": float(roc_val) if not np.isnan(roc_val) else None
        }

    def detect_trend_strength(self, df: pd.DataFrame) -> Dict:
        """Detect trend strength"""
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

        adx_val = adx.iloc[-1]
        plus_di_val = plus_di.iloc[-1]
        minus_di_val = minus_di.iloc[-1]

        if np.isnan(adx_val):
            return {"strength": 0.0, "direction": "NEUTRAL"}

        direction = "BULLISH" if plus_di_val > minus_di_val else "BEARISH"
        strength = min(1.0, adx_val / 50)

        return {
            "strength": float(strength),
            "direction": direction,
            "adx": float(adx_val)
        }

    def analyze_all_patterns(self, df: pd.DataFrame) -> List[Dict]:
        """Run all pattern detection"""
        patterns = []

        for detector in [self.detect_hammer, self.detect_engulfing,
                         self.detect_doji, self.detect_shooting_star]:
            result = detector(df)
            if result["detected"]:
                patterns.append(result)

        for detector in [self.detect_double_top_bottom, self.detect_triangle,
                         self.detect_head_shoulders]:
            result = detector(df)
            if result["detected"]:
                patterns.append(result)

        return patterns

    def calculate_confidence(self, patterns: List[Dict], momentum: Dict, trend: Dict) -> Dict:
        """Calculate confidence"""
        if not patterns:
            return {"confidence": 0.0, "direction": "NEUTRAL", "score_breakdown": {}}

        bullish_score = 0.0
        bearish_score = 0.0

        for pattern in patterns:
            weight = pattern.get("confidence", 0.5)
            if pattern["direction"] == "BULLISH":
                bullish_score += weight
            elif pattern["direction"] == "BEARISH":
                bearish_score += weight

        if momentum["direction"] == "BULLISH":
            bullish_score += momentum["strength"] * 0.5
        elif momentum["direction"] == "BEARISH":
            bearish_score += momentum["strength"] * 0.5

        if trend["direction"] == "BULLISH":
            bullish_score += trend["strength"] * 0.7
        elif trend["direction"] == "BEARISH":
            bearish_score += trend["strength"] * 0.7

        total_score = bullish_score + bearish_score
        if total_score == 0:
            return {"confidence": 0.0, "direction": "NEUTRAL", "score_breakdown": {}}

        confidence = max(bullish_score, bearish_score) / (total_score + 1)
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

    def on_bar(self, df: pd.DataFrame) -> Optional[Dict]:
        """
        Main strategy entry point with flat data detection
        """
        safe_log(self.logger, 'info', "=" * 60)
        safe_log(self.logger, 'info', "AI PATTERN ANALYSIS STARTED")
        safe_log(self.logger, 'info', f"Analyzing {len(df)} bars")

        if len(df) < self.lookback:
            log_warning(self.logger, f"Insufficient data: {len(df)} < {self.lookback}")
            safe_log(self.logger, 'info', "=" * 60)
            return None

        # PRE-CHECK: Detect flat data before calculations
        recent_10 = df.tail(10)
        flat_bars = sum(1 for i in range(len(recent_10))
                        if recent_10.iloc[i]['open'] == recent_10.iloc[i]['high'] ==
                        recent_10.iloc[i]['low'] == recent_10.iloc[i]['close'])

        unique_closes = df['close'].nunique()
        price_range = df['high'].max() - df['low'].min()
        price_avg = df['close'].mean()

        safe_log(self.logger, 'debug', f"Flat bars (last 10): {flat_bars}/10")
        safe_log(self.logger, 'debug', f"Unique prices: {unique_closes}/{len(df)}")
        safe_log(self.logger, 'debug', f"Price range: {price_range:.6f} ({(price_range / price_avg * 100):.6f}%)")

        # REJECT if data is too flat
        if unique_closes <= 1:
            log_error(self.logger, "ALL PRICES IDENTICAL - Market closed or data error")
            safe_log(self.logger, 'info', "=" * 60)
            return None

        if flat_bars > 7:
            log_error(self.logger, f"TOO MANY FLAT BARS ({flat_bars}/10) - Insufficient activity")
            safe_log(self.logger, 'info', "=" * 60)
            return None

        if price_avg > 0 and (price_range / price_avg * 100) < 0.01:
            log_error(self.logger, f"INSUFFICIENT MOVEMENT ({(price_range / price_avg * 100):.6f}%)")
            safe_log(self.logger, 'info', "=" * 60)
            return None

        # Now calculate ATR
        df = df.copy()
        df['atr'] = self.calculate_atr(df)

        atr_val = df['atr'].iloc[-1]

        # Check if ATR is suspiciously low (indicates flat data)
        avg_price = df['close'].mean()
        atr_pct = (atr_val / avg_price) * 100 if avg_price > 0 else 0

        safe_log(self.logger, 'debug', f"ATR: {atr_val:.6f} ({atr_pct:.4f}% of price)")

        if atr_pct < 0.001:  # Less than 0.001% of price
            log_warning(self.logger,
                        f"ATR too low ({atr_pct:.6f}%) - Data appears flat, rejecting analysis")
            safe_log(self.logger, 'info', "=" * 60)
            return None

        # Continue with rest of analysis...

        # Run all analyses
        safe_log(self.logger, 'info', "Running pattern detection...")
        patterns = self.analyze_all_patterns(df)

        safe_log(self.logger, 'info', "Analyzing momentum...")
        momentum = self.analyze_momentum(df)

        safe_log(self.logger, 'info', "Detecting trend strength...")
        trend = self.detect_trend_strength(df)

        # Calculate trading confidence
        decision = self.calculate_confidence(patterns, momentum, trend)

        # ALWAYS log analysis results
        safe_log(self.logger, 'info', "-" * 60)
        safe_log(self.logger, 'info', f"[PATTERNS] {len(patterns)} detected: {[p['pattern'] for p in patterns]}")
        safe_log(self.logger, 'info',
                 f"[MOMENTUM] {momentum['direction']} (strength: {momentum['strength']:.2f}, RSI: {momentum.get('rsi', 'N/A')})")
        safe_log(self.logger, 'info',
                 f"[TREND] {trend['direction']} (strength: {trend['strength']:.2f}, ADX: {trend.get('adx', 'N/A')})")
        safe_log(self.logger, 'info',
                 f"[DECISION] {decision['direction']} with {decision['confidence']:.1%} confidence")
        safe_log(self.logger, 'info',
                 f"[SCORES] Bullish: {decision['score_breakdown'].get('bullish', 0):.2f} | Bearish: {decision['score_breakdown'].get('bearish', 0):.2f}")
        safe_log(self.logger, 'info', "-" * 60)

        # Check confidence threshold
        if decision["confidence"] < self.confidence_threshold:
            log_warning(self.logger,
                        f"Confidence {decision['confidence']:.1%} BELOW threshold {self.confidence_threshold:.1%} - NO TRADE")
            safe_log(self.logger, 'info', "=" * 60)
            return None

        # Calculate risk parameters
        atr_val = df['atr'].iloc[-1]
        stop_pts = max(atr_val * self.stop_multiplier, 0.5)
        tp_pts = stop_pts * self.rr_take

        log_success(self.logger, f"TRADE SIGNAL GENERATED: {decision['direction']}")
        safe_log(self.logger, 'info', f"[RISK] Stop: {stop_pts:.2f} pts | TP: {tp_pts:.2f} pts | R:R 1:{self.rr_take}")
        safe_log(self.logger, 'info', "=" * 60)

        return {
            "side": decision["direction"],
            "stop_pts": float(stop_pts),
            "tp_pts": float(tp_pts),
            "meta": {
                "strategy": "ai_pattern_recognition",
                "confidence": decision["confidence"],
                "patterns_detected": [p["pattern"] for p in patterns],
                "momentum": momentum["direction"],
                "trend_strength": trend["strength"],
                "score_breakdown": decision["score_breakdown"]
            }
        }