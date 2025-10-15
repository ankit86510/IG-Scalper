import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging

from strategy.base import Strategy

logger = logging.getLogger(__name__)


class AIPatternRecognizer(Strategy):
    """
    AI-Powered Pattern Recognition Strategy

    Analyzes OHLC data for multiple technical patterns and makes autonomous
    trading decisions based on pattern confluence and strength.

    Features:
    - Multi-pattern detection (candlestick, chart patterns, technical indicators)
    - Pattern confidence scoring
    - Autonomous position management
    - Risk-adjusted position sizing
    """

    def __init__(self,
                 atr_period: int = 14,
                 stop_multiplier: float = 1.5,
                 rr_take: float = 2.0,
                 confidence_threshold: float = 0.65,
                 lookback_candles: int = 50):
        """
        Initialize AI Pattern Recognizer

        Args:
            atr_period: Period for ATR calculation
            stop_multiplier: Stop loss multiplier on ATR
            rr_take: Risk-reward ratio for take profit
            confidence_threshold: Minimum confidence to trade (0-1)
            lookback_candles: Candles to analyze for patterns
        """
        self.atr_period = atr_period
        self.stop_multiplier = stop_multiplier
        self.rr_take = rr_take
        self.confidence_threshold = confidence_threshold
        self.lookback = lookback_candles

    def calculate_atr(self, df: pd.DataFrame) -> pd.Series:
        """Calculate Average True Range"""
        h, l, c = df["high"], df["low"], df["close"]
        prev_c = c.shift(1)
        tr = pd.concat([
            (h - l).abs(),
            (h - prev_c).abs(),
            (l - prev_c).abs()
        ], axis=1).max(axis=1)
        return tr.rolling(self.atr_period).mean()

    # ==================== CANDLESTICK PATTERNS ====================

    def detect_hammer(self, df: pd.DataFrame, idx: int = -1) -> Dict:
        """Detect Hammer/Hanging Man pattern"""
        o, h, l, c = df['open'].iloc[idx], df['high'].iloc[idx], \
            df['low'].iloc[idx], df['close'].iloc[idx]

        body = abs(c - o)
        total = h - l
        lower_shadow = min(o, c) - l
        upper_shadow = h - max(o, c)

        if total == 0:
            return {"detected": False, "confidence": 0.0}

        # Hammer criteria: small body, long lower shadow, little upper shadow
        is_hammer = (
                lower_shadow > 2 * body and
                upper_shadow < 0.3 * body and
                body / total < 0.3
        )

        if is_hammer:
            confidence = min(0.8, (lower_shadow / body) * 0.15)
            direction = "BULLISH" if c > o else "BEARISH"
            return {
                "detected": True,
                "confidence": confidence,
                "direction": direction,
                "pattern": "HAMMER"
            }

        return {"detected": False, "confidence": 0.0}

    def detect_engulfing(self, df: pd.DataFrame, idx: int = -1) -> Dict:
        """Detect Bullish/Bearish Engulfing pattern"""
        if idx < -len(df) + 1:
            return {"detected": False, "confidence": 0.0}

        curr_o, curr_c = df['open'].iloc[idx], df['close'].iloc[idx]
        prev_o, prev_c = df['open'].iloc[idx - 1], df['close'].iloc[idx - 1]

        curr_body = abs(curr_c - curr_o)
        prev_body = abs(prev_c - prev_o)

        # Bullish engulfing
        bullish = (
                prev_c < prev_o and  # Previous bearish
                curr_c > curr_o and  # Current bullish
                curr_o < prev_c and  # Opens below previous close
                curr_c > prev_o and  # Closes above previous open
                curr_body > prev_body * 1.2  # Significantly larger
        )

        # Bearish engulfing
        bearish = (
                prev_c > prev_o and
                curr_c < curr_o and
                curr_o > prev_c and
                curr_c < prev_o and
                curr_body > prev_body * 1.2
        )

        if bullish:
            return {
                "detected": True,
                "confidence": 0.75,
                "direction": "BULLISH",
                "pattern": "ENGULFING"
            }
        elif bearish:
            return {
                "detected": True,
                "confidence": 0.75,
                "direction": "BEARISH",
                "pattern": "ENGULFING"
            }

        return {"detected": False, "confidence": 0.0}

    def detect_doji(self, df: pd.DataFrame, idx: int = -1) -> Dict:
        """Detect Doji pattern (indecision)"""
        o, h, l, c = df['open'].iloc[idx], df['high'].iloc[idx], \
            df['low'].iloc[idx], df['close'].iloc[idx]

        body = abs(c - o)
        total = h - l

        if total == 0:
            return {"detected": False, "confidence": 0.0}

        # Doji: very small body relative to range
        is_doji = body / total < 0.1

        if is_doji:
            return {
                "detected": True,
                "confidence": 0.5,
                "direction": "NEUTRAL",
                "pattern": "DOJI"
            }

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

        # Shooting star: long upper shadow, small body, little lower shadow
        is_shooting_star = (
                upper_shadow > 2 * body and
                lower_shadow < 0.3 * body and
                body / total < 0.3
        )

        if is_shooting_star:
            return {
                "detected": True,
                "confidence": 0.7,
                "direction": "BEARISH",
                "pattern": "SHOOTING_STAR"
            }

        return {"detected": False, "confidence": 0.0}

    # ==================== CHART PATTERNS ====================

    def detect_double_top_bottom(self, df: pd.DataFrame) -> Dict:
        """Detect Double Top/Bottom patterns"""
        window = min(20, len(df) // 3)
        if len(df) < window * 2:
            return {"detected": False, "confidence": 0.0}

        highs = df['high'].rolling(window=5).max()
        lows = df['low'].rolling(window=5).min()

        # Find peaks and troughs
        peaks = []
        troughs = []

        for i in range(window, len(df) - window):
            if highs.iloc[i] == df['high'].iloc[i - window:i + window].max():
                peaks.append((i, highs.iloc[i]))
            if lows.iloc[i] == df['low'].iloc[i - window:i + window].min():
                troughs.append((i, lows.iloc[i]))

        # Check for double top
        if len(peaks) >= 2:
            last_two = peaks[-2:]
            price_diff = abs(last_two[0][1] - last_two[1][1]) / last_two[0][1]
            time_diff = last_two[1][0] - last_two[0][0]

            if price_diff < 0.02 and window < time_diff < window * 2:
                return {
                    "detected": True,
                    "confidence": 0.7,
                    "direction": "BEARISH",
                    "pattern": "DOUBLE_TOP"
                }

        # Check for double bottom
        if len(troughs) >= 2:
            last_two = troughs[-2:]
            price_diff = abs(last_two[0][1] - last_two[1][1]) / last_two[0][1]
            time_diff = last_two[1][0] - last_two[0][0]

            if price_diff < 0.02 and window < time_diff < window * 2:
                return {
                    "detected": True,
                    "confidence": 0.7,
                    "direction": "BULLISH",
                    "pattern": "DOUBLE_BOTTOM"
                }

        return {"detected": False, "confidence": 0.0}

    def detect_triangle(self, df: pd.DataFrame) -> Dict:
        """Detect Triangle consolidation patterns"""
        window = min(20, len(df) // 2)
        if len(df) < window:
            return {"detected": False, "confidence": 0.0}

        recent = df.tail(window)
        highs = recent['high']
        lows = recent['low']

        # Fit trendlines
        x = np.arange(len(recent))
        high_slope = np.polyfit(x, highs, 1)[0]
        low_slope = np.polyfit(x, lows, 1)[0]

        # Ascending triangle: flat top, rising bottom
        if abs(high_slope) < 0.001 and low_slope > 0.001:
            return {
                "detected": True,
                "confidence": 0.65,
                "direction": "BULLISH",
                "pattern": "ASCENDING_TRIANGLE"
            }

        # Descending triangle: flat bottom, falling top
        if abs(low_slope) < 0.001 and high_slope < -0.001:
            return {
                "detected": True,
                "confidence": 0.65,
                "direction": "BEARISH",
                "pattern": "DESCENDING_TRIANGLE"
            }

        # Symmetrical triangle: converging trendlines
        if high_slope < -0.001 and low_slope > 0.001:
            # Direction depends on breakout
            volatility = recent['close'].pct_change().std()
            return {
                "detected": True,
                "confidence": 0.5,
                "direction": "NEUTRAL",
                "pattern": "SYMMETRICAL_TRIANGLE",
                "volatility": volatility
            }

        return {"detected": False, "confidence": 0.0}

    def detect_head_shoulders(self, df: pd.DataFrame) -> Dict:
        """Detect Head and Shoulders pattern"""
        window = min(30, len(df) // 2)
        if len(df) < window:
            return {"detected": False, "confidence": 0.0}

        recent = df.tail(window)
        highs = recent['high'].values

        # Find three peaks
        peaks = []
        for i in range(5, len(highs) - 5):
            if highs[i] == max(highs[i - 5:i + 6]):
                peaks.append((i, highs[i]))

        if len(peaks) >= 3:
            # Check if middle peak is highest (head)
            last_three = peaks[-3:]
            if (last_three[1][1] > last_three[0][1] and
                    last_three[1][1] > last_three[2][1] and
                    abs(last_three[0][1] - last_three[2][1]) / last_three[0][1] < 0.05):
                return {
                    "detected": True,
                    "confidence": 0.75,
                    "direction": "BEARISH",
                    "pattern": "HEAD_SHOULDERS"
                }

        return {"detected": False, "confidence": 0.0}

    # ==================== MOMENTUM & TREND ====================

    def analyze_momentum(self, df: pd.DataFrame) -> Dict:
        """Analyze price momentum using multiple indicators"""
        if len(df) < 20:
            return {"strength": 0.0, "direction": "NEUTRAL"}

        # RSI
        delta = df['close'].diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = -delta.clip(upper=0).rolling(14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))

        # Rate of Change
        roc = ((df['close'] - df['close'].shift(10)) / df['close'].shift(10) * 100)

        # Moving Average Convergence
        ema12 = df['close'].ewm(span=12).mean()
        ema26 = df['close'].ewm(span=26).mean()
        macd = ema12 - ema26

        # Aggregate momentum
        rsi_val = rsi.iloc[-1]
        roc_val = roc.iloc[-1]
        macd_val = macd.iloc[-1]

        # Score momentum (-1 to 1)
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
        """Detect trend strength using ADX-like calculation"""
        if len(df) < 25:
            return {"strength": 0.0, "direction": "NEUTRAL"}

        # Simple ADX approximation
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

        # Strong trend: ADX > 25
        # Direction: which DI is higher
        direction = "BULLISH" if plus_di_val > minus_di_val else "BEARISH"
        strength = min(1.0, adx_val / 50)  # Normalize

        return {
            "strength": float(strength),
            "direction": direction,
            "adx": float(adx_val)
        }

    # ==================== AI DECISION ENGINE ====================

    def analyze_all_patterns(self, df: pd.DataFrame) -> List[Dict]:
        """Run all pattern detection methods"""
        patterns = []

        # Candlestick patterns
        hammer = self.detect_hammer(df)
        if hammer["detected"]:
            patterns.append(hammer)

        engulfing = self.detect_engulfing(df)
        if engulfing["detected"]:
            patterns.append(engulfing)

        doji = self.detect_doji(df)
        if doji["detected"]:
            patterns.append(doji)

        shooting_star = self.detect_shooting_star(df)
        if shooting_star["detected"]:
            patterns.append(shooting_star)

        # Chart patterns
        double = self.detect_double_top_bottom(df)
        if double["detected"]:
            patterns.append(double)

        triangle = self.detect_triangle(df)
        if triangle["detected"]:
            patterns.append(triangle)

        head_shoulders = self.detect_head_shoulders(df)
        if head_shoulders["detected"]:
            patterns.append(head_shoulders)

        return patterns

    def calculate_confidence(self, patterns: List[Dict], momentum: Dict, trend: Dict) -> Dict:
        """Calculate overall trading confidence and direction"""
        if not patterns:
            return {"confidence": 0.0, "direction": "NEUTRAL", "score_breakdown": {}}

        # Weight patterns by confidence
        bullish_score = 0.0
        bearish_score = 0.0

        for pattern in patterns:
            weight = pattern.get("confidence", 0.5)
            if pattern["direction"] == "BULLISH":
                bullish_score += weight
            elif pattern["direction"] == "BEARISH":
                bearish_score += weight

        # Add momentum influence
        if momentum["direction"] == "BULLISH":
            bullish_score += momentum["strength"] * 0.5
        elif momentum["direction"] == "BEARISH":
            bearish_score += momentum["strength"] * 0.5

        # Add trend influence
        if trend["direction"] == "BULLISH":
            bullish_score += trend["strength"] * 0.7
        elif trend["direction"] == "BEARISH":
            bearish_score += trend["strength"] * 0.7

        # Normalize scores
        total_score = bullish_score + bearish_score
        if total_score == 0:
            return {"confidence": 0.0, "direction": "NEUTRAL", "score_breakdown": {}}

        confidence = max(bullish_score, bearish_score) / (total_score + 1)  # +1 to prevent overconfidence
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
        Main strategy entry point - analyzes OHLC data and makes trading decisions
        """
        if len(df) < self.lookback:
            logger.debug(f"Insufficient data: {len(df)} < {self.lookback}")
            return None

        df = df.copy()
        df['atr'] = self.calculate_atr(df)

        if df['atr'].isna().any():
            return None

        # Run all analyses
        patterns = self.analyze_all_patterns(df)
        momentum = self.analyze_momentum(df)
        trend = self.detect_trend_strength(df)

        # Calculate trading confidence
        decision = self.calculate_confidence(patterns, momentum, trend)

        logger.info(f"AI Analysis: {len(patterns)} patterns detected")
        logger.info(f"Momentum: {momentum['direction']} ({momentum['strength']:.2f})")
        logger.info(f"Trend: {trend['direction']} ({trend['strength']:.2f})")
        logger.info(f"Decision: {decision['direction']} with {decision['confidence']:.1%} confidence")

        # Only trade if confidence exceeds threshold
        if decision["confidence"] < self.confidence_threshold:
            logger.info(f"Confidence {decision['confidence']:.1%} below threshold {self.confidence_threshold:.1%}")
            return None

        # Calculate risk parameters
        atr_val = df['atr'].iloc[-1]
        stop_pts = max(atr_val * self.stop_multiplier, 0.5)
        tp_pts = stop_pts * self.rr_take

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