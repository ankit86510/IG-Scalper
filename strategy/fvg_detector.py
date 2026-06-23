"""FVG (Fair Value Gap) detection module.

Pure, stateless detection of Fair Value Gaps in OHLC price data.
No I/O, no side effects — ideal for property-based testing.
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import List
from zoneinfo import ZoneInfo

import pandas as pd

logger = logging.getLogger("ig-scalper")

ROME_TZ = ZoneInfo("Europe/Rome")


@dataclass
class FVG:
    """Represents a Fair Value Gap detected in OHLC candle data.

    A 3-candle pattern where the wick of candle 1 and candle 3 do not
    overlap, leaving a gap in price coverage (imbalance zone).
    """

    type: str  # "bullish" or "bearish"
    zone_upper: float  # Upper boundary of the gap zone
    zone_lower: float  # Lower boundary of the gap zone
    formation_ts: datetime  # Timestamp of the middle candle (candle[i+1])
    source_tf: str  # e.g. "60min", "15min", "5min"
    fill_status: str = "unfilled"  # "unfilled", "partial", "filled"
    age_bars: int = 0  # Bars elapsed since formation

    def to_dict(self) -> dict:
        """Serialize to JSON-safe dict."""
        return {
            "type": self.type,
            "zone_upper": self.zone_upper,
            "zone_lower": self.zone_lower,
            "formation_ts": self.formation_ts.isoformat(),
            "source_tf": self.source_tf,
            "fill_status": self.fill_status,
            "age_bars": self.age_bars,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "FVG":
        """Deserialize from dict."""
        return cls(
            type=d["type"],
            zone_upper=d["zone_upper"],
            zone_lower=d["zone_lower"],
            formation_ts=datetime.fromisoformat(d["formation_ts"]),
            source_tf=d["source_tf"],
            fill_status=d.get("fill_status", "unfilled"),
            age_bars=d.get("age_bars", 0),
        )


@dataclass
class Bias:
    """Directional bias derived from higher-timeframe FVG analysis.

    Used to filter 5min trade signals — only signals aligned with the
    higher-timeframe bias are considered valid.
    """

    direction: str  # "bullish", "bearish", "neutral"
    confidence: float  # 0.0 to 1.0


class FVGDetector:
    """Pure FVG detection on OHLC DataFrames. No I/O, no state."""

    def detect(self, df: pd.DataFrame, timeframe: str) -> List[FVG]:
        """Scan all 3-candle windows up through penultimate bar (iloc[-2]).

        Returns list of detected FVGs. Excludes the last bar (still forming).

        Args:
            df: DataFrame with OHLC columns (open, high, low, close) and a
                datetime index.
            timeframe: Source timeframe label, e.g. "60min", "15min", "5min".

        Returns:
            List of FVG objects detected in the data.
        """
        if df is None or len(df) < 3:
            return []

        fvgs: List[FVG] = []
        ohlc_cols = ["open", "high", "low", "close"]

        # Scan windows from index 0 through penultimate completed bar.
        # The last bar (iloc[-1]) is still forming, so the last valid
        # window's candle[i+2] is iloc[-2].
        # That means i+2 <= len(df)-2, so i <= len(df)-4.
        max_i = len(df) - 3  # last i where we can form a 3-candle window ending at iloc[-2]

        for i in range(max_i):
            # Get the three candles in this window
            c1 = df.iloc[i]
            c2 = df.iloc[i + 1]
            c3 = df.iloc[i + 2]

            # Skip window if any of the 3 candles have NaN in OHLC columns
            if (
                c1[ohlc_cols].isna().any()
                or c2[ohlc_cols].isna().any()
                or c3[ohlc_cols].isna().any()
            ):
                continue

            # Bullish FVG: candle[i].high < candle[i+2].low
            if c1["high"] < c3["low"]:
                ts = c2.name if isinstance(c2.name, datetime) else datetime.now()
                fvg = FVG(
                    type="bullish",
                    zone_upper=float(c3["low"]),
                    zone_lower=float(c1["high"]),
                    formation_ts=ts,
                    source_tf=timeframe,
                )
                fvgs.append(fvg)
                # Req 8.1: Log each detected FVG at DEBUG with Rome timezone
                rome_ts = ts.astimezone(ROME_TZ) if ts.tzinfo else ts
                logger.debug(
                    f"FVG detected: type=bullish | zone=[{fvg.zone_lower:.2f}, {fvg.zone_upper:.2f}] | "
                    f"tf={timeframe} | formed={rome_ts.strftime('%Y-%m-%d %H:%M:%S %Z') if ts.tzinfo else rome_ts.strftime('%Y-%m-%d %H:%M:%S')}"
                )

            # Bearish FVG: candle[i].low > candle[i+2].high
            elif c1["low"] > c3["high"]:
                ts = c2.name if isinstance(c2.name, datetime) else datetime.now()
                fvg = FVG(
                    type="bearish",
                    zone_upper=float(c1["low"]),
                    zone_lower=float(c3["high"]),
                    formation_ts=ts,
                    source_tf=timeframe,
                )
                fvgs.append(fvg)
                # Req 8.1: Log each detected FVG at DEBUG with Rome timezone
                rome_ts = ts.astimezone(ROME_TZ) if ts.tzinfo else ts
                logger.debug(
                    f"FVG detected: type=bearish | zone=[{fvg.zone_lower:.2f}, {fvg.zone_upper:.2f}] | "
                    f"tf={timeframe} | formed={rome_ts.strftime('%Y-%m-%d %H:%M:%S %Z') if ts.tzinfo else rome_ts.strftime('%Y-%m-%d %H:%M:%S')}"
                )

        # Req 8.1: Log summary count at INFO level
        if fvgs:
            bull_count = sum(1 for f in fvgs if f.type == "bullish")
            bear_count = sum(1 for f in fvgs if f.type == "bearish")
            logger.info(
                f"FVGDetector [{timeframe}]: {len(fvgs)} FVGs detected "
                f"(bullish={bull_count}, bearish={bear_count})"
            )

        return fvgs

    def update_fill_status(self, fvgs: List[FVG], df: pd.DataFrame, max_age: int = 50) -> List[FVG]:
        """Process bars chronologically, update fill status, expire old FVGs.

        For each FVG, only processes bars AFTER the FVG's formation timestamp.
        This ensures age_bars reflects actual bars elapsed since formation.

        For each relevant bar:
          - Increment age_bars for FVGs formed before that bar
          - For bullish FVGs: check if bar.high fills the zone
          - For bearish FVGs: check if bar.low fills the zone
          - Remove fully-filled and expired FVGs

        Args:
            fvgs: List of FVG objects to update (modified in-place).
            df: DataFrame with OHLC columns processed chronologically.
            max_age: Maximum age in bars before an FVG expires (default 50).

        Returns:
            List of remaining unfilled/partial FVGs.
        """
        if not fvgs or df is None or len(df) == 0:
            return fvgs

        active = list(fvgs)

        for idx, bar in df.iterrows():
            bar_high = bar["high"]
            bar_low = bar["low"]

            # Skip bars with NaN in high/low
            if pd.isna(bar_high) or pd.isna(bar_low):
                continue

            # Get bar timestamp for age comparison
            bar_ts = idx if isinstance(idx, datetime) else None

            to_remove = []

            for fvg in active:
                # Only process bars that come AFTER this FVG formed
                if bar_ts is not None and fvg.formation_ts is not None:
                    # Compare timestamps (handle timezone-aware vs naive)
                    fvg_ts = fvg.formation_ts
                    cmp_bar_ts = bar_ts

                    # Normalize both to naive for comparison if mixed
                    if hasattr(fvg_ts, 'tzinfo') and fvg_ts.tzinfo is not None:
                        if hasattr(cmp_bar_ts, 'tzinfo') and cmp_bar_ts.tzinfo is None:
                            fvg_ts = fvg_ts.replace(tzinfo=None)
                    elif hasattr(cmp_bar_ts, 'tzinfo') and cmp_bar_ts.tzinfo is not None:
                        cmp_bar_ts = cmp_bar_ts.replace(tzinfo=None)

                    if cmp_bar_ts <= fvg_ts:
                        continue  # Skip bars at or before formation

                # Increment age for each bar processed after formation
                fvg.age_bars += 1

                # Check fill status based on FVG type
                if fvg.type == "bullish":
                    if bar_high >= fvg.zone_upper:
                        fvg.fill_status = "filled"
                        to_remove.append(fvg)
                    elif bar_high >= fvg.zone_lower:
                        fvg.fill_status = "partial"
                        fvg.zone_lower = float(bar_high)
                elif fvg.type == "bearish":
                    if bar_low <= fvg.zone_lower:
                        fvg.fill_status = "filled"
                        to_remove.append(fvg)
                    elif bar_low <= fvg.zone_upper:
                        fvg.fill_status = "partial"
                        fvg.zone_upper = float(bar_low)

                # Check age expiry
                if fvg.age_bars > max_age and fvg not in to_remove:
                    to_remove.append(fvg)

            # Remove filled and expired FVGs from active set
            for fvg in to_remove:
                active.remove(fvg)

        return active
