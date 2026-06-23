"""Bias calculation module for multi-timeframe FVG analysis.

Derives directional bias from higher-timeframe FVGs (60min, 15min)
to filter 5min trade signals. Pure, stateless calculations.
"""

import logging
from typing import List
from zoneinfo import ZoneInfo

from strategy.fvg_detector import Bias, FVG

logger = logging.getLogger("ig-scalper")

ROME_TZ = ZoneInfo("Europe/Rome")


class BiasCalculator:
    """Derives directional bias from higher-timeframe FVGs."""

    def calculate_60min_bias(self, fvgs_60min: List[FVG]) -> Bias:
        """Calculate directional bias from 60min unfilled FVGs.

        Counts unfilled/partial bullish vs bearish FVGs to determine
        overall market bias direction and confidence level.

        Args:
            fvgs_60min: List of FVG objects from the 60min timeframe.

        Returns:
            Bias with direction and confidence score (0.0 to 1.0).
            Returns neutral with confidence 0.0 if no unfilled FVGs exist.
        """
        # Filter to only unfilled and partial FVGs (exclude filled)
        active_fvgs = [
            fvg for fvg in fvgs_60min
            if fvg.fill_status in ("unfilled", "partial")
        ]

        # No unfilled FVGs → neutral bias with confidence 0.0
        if not active_fvgs:
            return Bias(direction="neutral", confidence=0.0)

        bull_count = sum(1 for fvg in active_fvgs if fvg.type == "bullish")
        bear_count = sum(1 for fvg in active_fvgs if fvg.type == "bearish")

        total = bull_count + bear_count

        # If total is 0 (shouldn't happen given active_fvgs check, but safe)
        if total == 0:
            return Bias(direction="neutral", confidence=0.0)

        # Determine direction
        if bull_count > bear_count:
            direction = "bullish"
        elif bear_count > bull_count:
            direction = "bearish"
        else:
            direction = "neutral"

        # Confidence = abs(bull - bear) / (bull + bear)
        confidence = abs(bull_count - bear_count) / total

        bias = Bias(direction=direction, confidence=confidence)

        # Req 8.2: Log bias determination with contributing FVGs
        logger.info(
            f"BiasCalculator 60min: direction={direction} | confidence={confidence:.2f} | "
            f"contributing FVGs: bullish={bull_count}, bearish={bear_count} (total active={total})"
        )

        return bias

    def adjust_with_15min(self, bias: Bias, fvgs_15min: List[FVG]) -> Bias:
        """Adjust bias confidence using 15min FVG confirmation.

        If the 15min majority direction matches the 60min bias direction,
        confidence increases by 0.2 (capped at 1.0). If it opposes,
        confidence decreases by 0.3 (floored at 0.0). Direction unchanged.

        Args:
            bias: The 60min bias to adjust.
            fvgs_15min: List of FVG objects from the 15min timeframe.

        Returns:
            Adjusted Bias with updated confidence. Direction remains the same.
        """
        # Filter to only unfilled and partial FVGs
        active_fvgs = [
            fvg for fvg in fvgs_15min
            if fvg.fill_status in ("unfilled", "partial")
        ]

        # No active 15min FVGs → no adjustment
        if not active_fvgs:
            return Bias(direction=bias.direction, confidence=bias.confidence)

        bull_count = sum(1 for fvg in active_fvgs if fvg.type == "bullish")
        bear_count = sum(1 for fvg in active_fvgs if fvg.type == "bearish")

        # Determine 15min majority direction
        if bull_count > bear_count:
            majority_15min = "bullish"
        elif bear_count > bull_count:
            majority_15min = "bearish"
        else:
            # Equal counts on 15min → no adjustment
            return Bias(direction=bias.direction, confidence=bias.confidence)

        # Adjust confidence based on alignment
        new_confidence = bias.confidence

        if majority_15min == bias.direction:
            # 15min confirms 60min bias → boost confidence
            new_confidence = min(bias.confidence + 0.2, 1.0)
        else:
            # 15min opposes 60min bias → reduce confidence
            new_confidence = max(bias.confidence - 0.3, 0.0)

        # Req 8.2: Log 15min adjustment details
        logger.info(
            f"BiasCalculator 15min adjustment: 15min_majority={majority_15min} | "
            f"alignment={'confirms' if majority_15min == bias.direction else 'opposes'} | "
            f"confidence {bias.confidence:.2f} -> {new_confidence:.2f} | "
            f"15min FVGs: bullish={bull_count}, bearish={bear_count}"
        )

        return Bias(direction=bias.direction, confidence=new_confidence)
