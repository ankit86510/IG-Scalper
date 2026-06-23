"""Signal generation module for multi-timeframe FVG analysis.

Produces trade signals from 5min FVGs aligned with higher-timeframe bias.
Pure calculation logic — no I/O, no side effects.
"""

import logging
from typing import List, Optional
from zoneinfo import ZoneInfo

from strategy.fvg_detector import Bias, FVG

logger = logging.getLogger("ig-scalper")

ROME_TZ = ZoneInfo("Europe/Rome")


class SignalGenerator:
    """Produces trade signals from 5min FVGs aligned with bias.

    Selects the most recent unfilled 5min FVG that matches the bias
    direction, calculates entry/stop/TP levels, and returns the signal
    in the standard format or None if conditions aren't met.
    """

    def __init__(self, stop_buffer: float, min_confidence: float):
        """Initialize SignalGenerator.

        Args:
            stop_buffer: Points to add beyond opposite FVG boundary for stop loss.
            min_confidence: Minimum bias confidence required to generate a signal.
        """
        self.stop_buffer = stop_buffer
        self.min_confidence = min_confidence

    def generate(
        self,
        fvgs_5min: List[FVG],
        bias: Bias,
        fvgs_higher_tf: List[FVG],
    ) -> Optional[dict]:
        """Generate a trade signal from aligned 5min FVGs and bias.

        Returns {"side", "stop_pts", "tp_pts", "meta"} or None.

        Logic:
        - Selects most recent unfilled 5min FVG matching bias direction.
        - Entry at FVG zone boundary (zone_upper for BUY, zone_lower for SELL).
        - Stop beyond opposite boundary + buffer.
        - TP at nearest HTF zone boundary in profit direction, or zone_size
          as default target distance.
        - Discards if TP distance <= SL distance (unfavorable R:R).

        Args:
            fvgs_5min: List of FVG objects from the 5min timeframe.
            bias: Current Bias (direction + confidence) from higher TFs.
            fvgs_higher_tf: List of FVG objects from 60min and 15min timeframes.

        Returns:
            Signal dict or None if no valid signal can be generated.
        """
        # Req 4.7: If bias is neutral, no signal
        if bias.direction == "neutral":
            logger.debug("SignalGenerator: no signal — bias is neutral")
            return None

        # Req 4.8: If bias confidence < min_confidence, no signal
        if bias.confidence < self.min_confidence:
            logger.debug(
                f"SignalGenerator: no signal — confidence {bias.confidence:.2f} "
                f"< threshold {self.min_confidence:.2f}"
            )
            return None

        # Determine required FVG type based on bias
        required_type = "bullish" if bias.direction == "bullish" else "bearish"

        # Filter to unfilled/partial 5min FVGs matching bias direction
        matching_fvgs = [
            fvg for fvg in fvgs_5min
            if fvg.type == required_type
            and fvg.fill_status in ("unfilled", "partial")
        ]

        # No matching FVGs → no signal
        if not matching_fvgs:
            logger.debug(
                f"SignalGenerator: no signal — no unfilled {required_type} 5min FVGs"
            )
            return None

        # Select most recent (by formation_ts)
        trigger_fvg = max(matching_fvgs, key=lambda f: f.formation_ts)

        # Calculate entry, stop, and TP based on direction
        if bias.direction == "bullish":
            return self._generate_buy_signal(trigger_fvg, bias, fvgs_higher_tf, fvgs_5min)
        else:
            return self._generate_sell_signal(trigger_fvg, bias, fvgs_higher_tf, fvgs_5min)

    def _generate_buy_signal(
        self,
        trigger_fvg: FVG,
        bias: Bias,
        fvgs_higher_tf: List[FVG],
        fvgs_5min: List[FVG],
    ) -> Optional[dict]:
        """Generate a BUY signal from a bullish 5min FVG.

        Entry at zone_upper (price retraces down to top of gap, we buy).
        Stop below zone_lower - buffer.
        TP at nearest HTF resistance above entry, or zone_size as default.
        """
        entry = trigger_fvg.zone_upper
        zone_lower = trigger_fvg.zone_lower

        # Req 4.3: Stop loss below zone_lower - buffer
        stop_pts = (entry - zone_lower) + self.stop_buffer

        # Req 4.4: TP at nearest HTF zone boundary above entry
        tp_pts = self._find_tp_distance_buy(entry, trigger_fvg, fvgs_higher_tf)

        # Req 4.5: Discard if TP <= SL distance
        if tp_pts <= stop_pts:
            logger.debug(
                f"SignalGenerator: BUY discarded — tp_pts={tp_pts:.4f} <= stop_pts={stop_pts:.4f} (unfavorable R:R)"
            )
            return None

        # Req 8.3: Log signal details with entry zone, stop, TP, alignment rationale
        rome_ts = trigger_fvg.formation_ts.astimezone(ROME_TZ) if trigger_fvg.formation_ts.tzinfo else trigger_fvg.formation_ts
        logger.info(
            f"SignalGenerator BUY: entry_zone=[{trigger_fvg.zone_lower:.2f}, {trigger_fvg.zone_upper:.2f}] | "
            f"entry={entry:.2f} | stop_pts={stop_pts:.4f} | tp_pts={tp_pts:.4f} | "
            f"R:R=1:{tp_pts/stop_pts:.2f} | "
            f"bias={bias.direction}@{bias.confidence:.2f} | "
            f"trigger_fvg_formed={rome_ts.strftime('%Y-%m-%d %H:%M:%S %Z') if trigger_fvg.formation_ts.tzinfo else rome_ts.strftime('%Y-%m-%d %H:%M:%S')}"
        )

        # Req 4.6: Return standard signal format
        return {
            "side": "BUY",
            "stop_pts": round(stop_pts, 4),
            "tp_pts": round(tp_pts, 4),
            "meta": self._build_meta(trigger_fvg, bias, fvgs_higher_tf, fvgs_5min),
        }

    def _generate_sell_signal(
        self,
        trigger_fvg: FVG,
        bias: Bias,
        fvgs_higher_tf: List[FVG],
        fvgs_5min: List[FVG],
    ) -> Optional[dict]:
        """Generate a SELL signal from a bearish 5min FVG.

        Entry at zone_lower (price retraces up to bottom of gap, we sell).
        Stop above zone_upper + buffer.
        TP at nearest HTF support below entry, or zone_size as default.
        """
        entry = trigger_fvg.zone_lower
        zone_upper = trigger_fvg.zone_upper

        # Req 4.3: Stop loss above zone_upper + buffer
        stop_pts = (zone_upper - entry) + self.stop_buffer

        # Req 4.4: TP at nearest HTF zone boundary below entry
        tp_pts = self._find_tp_distance_sell(entry, trigger_fvg, fvgs_higher_tf)

        # Req 4.5: Discard if TP <= SL distance
        if tp_pts <= stop_pts:
            logger.debug(
                f"SignalGenerator: SELL discarded — tp_pts={tp_pts:.4f} <= stop_pts={stop_pts:.4f} (unfavorable R:R)"
            )
            return None

        # Req 8.3: Log signal details with entry zone, stop, TP, alignment rationale
        rome_ts = trigger_fvg.formation_ts.astimezone(ROME_TZ) if trigger_fvg.formation_ts.tzinfo else trigger_fvg.formation_ts
        logger.info(
            f"SignalGenerator SELL: entry_zone=[{trigger_fvg.zone_lower:.2f}, {trigger_fvg.zone_upper:.2f}] | "
            f"entry={entry:.2f} | stop_pts={stop_pts:.4f} | tp_pts={tp_pts:.4f} | "
            f"R:R=1:{tp_pts/stop_pts:.2f} | "
            f"bias={bias.direction}@{bias.confidence:.2f} | "
            f"trigger_fvg_formed={rome_ts.strftime('%Y-%m-%d %H:%M:%S %Z') if trigger_fvg.formation_ts.tzinfo else rome_ts.strftime('%Y-%m-%d %H:%M:%S')}"
        )

        # Req 4.6: Return standard signal format
        return {
            "side": "SELL",
            "stop_pts": round(stop_pts, 4),
            "tp_pts": round(tp_pts, 4),
            "meta": self._build_meta(trigger_fvg, bias, fvgs_higher_tf, fvgs_5min),
        }

    def _find_tp_distance_buy(
        self, entry: float, trigger_fvg: FVG, fvgs_higher_tf: List[FVG]
    ) -> float:
        """Find TP distance for a BUY signal.

        Looks for the nearest unfilled HTF zone boundary ABOVE entry that
        acts as resistance (bearish HTF FVG zone_lower above entry, or
        bullish HTF FVG zone_upper above entry). Uses zone_size as
        fallback if no HTF levels found.

        Per Req 4.4: Use the nearest HTF boundary as TP if available
        (shortest distance from entry). Otherwise fall back to zone_size.
        """
        zone_size = trigger_fvg.zone_upper - trigger_fvg.zone_lower

        # Find HTF zone boundaries above entry (potential resistance)
        htf_targets: List[float] = []
        for fvg in fvgs_higher_tf:
            if fvg.fill_status == "filled":
                continue
            # Bearish HTF FVG zone_lower above entry = resistance
            if fvg.type == "bearish" and fvg.zone_lower > entry:
                htf_targets.append(fvg.zone_lower)
            # Bullish HTF FVG zone_upper above entry = potential resistance area
            if fvg.type == "bullish" and fvg.zone_upper > entry:
                htf_targets.append(fvg.zone_upper)

        if htf_targets:
            # Nearest HTF target above entry (shortest distance)
            nearest_htf = min(htf_targets)
            return nearest_htf - entry

        # Default TP is zone_size distance above entry
        return zone_size

    def _find_tp_distance_sell(
        self, entry: float, trigger_fvg: FVG, fvgs_higher_tf: List[FVG]
    ) -> float:
        """Find TP distance for a SELL signal.

        Looks for the nearest unfilled HTF zone boundary BELOW entry that
        acts as support (bullish HTF FVG zone_upper below entry, or
        bearish HTF FVG zone_lower below entry). Uses zone_size as
        fallback if no HTF levels found.

        Per Req 4.4: Use the nearest HTF boundary as TP if available
        (shortest distance from entry). Otherwise fall back to zone_size.
        """
        zone_size = trigger_fvg.zone_upper - trigger_fvg.zone_lower

        # Find HTF zone boundaries below entry (potential support)
        htf_targets: List[float] = []
        for fvg in fvgs_higher_tf:
            if fvg.fill_status == "filled":
                continue
            # Bullish HTF FVG zone_upper below entry = support
            if fvg.type == "bullish" and fvg.zone_upper < entry:
                htf_targets.append(fvg.zone_upper)
            # Bearish HTF FVG zone_lower below entry = potential support area
            if fvg.type == "bearish" and fvg.zone_lower < entry:
                htf_targets.append(fvg.zone_lower)

        if htf_targets:
            # Nearest HTF target below entry (highest value below = shortest distance)
            nearest_htf = max(htf_targets)
            return entry - nearest_htf

        # Default TP is zone_size distance below entry
        return zone_size

    def _build_meta(
        self,
        trigger_fvg: FVG,
        bias: Bias,
        fvgs_higher_tf: List[FVG],
        fvgs_5min: List[FVG],
    ) -> dict:
        """Build the meta field for the signal output.

        Req 4.9: Includes source FVGs, bias direction/confidence,
        and triggering 5min FVG zone boundaries.
        """
        # Separate HTF FVGs by timeframe
        fvgs_60min = [f.to_dict() for f in fvgs_higher_tf if f.source_tf == "60min"]
        fvgs_15min = [f.to_dict() for f in fvgs_higher_tf if f.source_tf == "15min"]
        fvgs_5min_dicts = [f.to_dict() for f in fvgs_5min]

        return {
            "bias_direction": bias.direction,
            "bias_confidence": bias.confidence,
            "trigger_fvg": {
                "type": trigger_fvg.type,
                "zone_upper": trigger_fvg.zone_upper,
                "zone_lower": trigger_fvg.zone_lower,
                "source_tf": trigger_fvg.source_tf,
            },
            "fvgs_60min": fvgs_60min,
            "fvgs_15min": fvgs_15min,
            "fvgs_5min": fvgs_5min_dicts,
            "entry_zone": (trigger_fvg.zone_lower, trigger_fvg.zone_upper),
        }
