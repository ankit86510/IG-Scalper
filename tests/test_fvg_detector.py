"""Unit tests for FVGDetector.detect() method."""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from strategy.fvg_detector import FVG, FVGDetector


def _make_df(rows, start_ts=None):
    """Helper: build a DataFrame from list of (open, high, low, close) tuples."""
    if start_ts is None:
        start_ts = datetime(2024, 1, 1, 9, 0)
    index = [start_ts + timedelta(minutes=5 * i) for i in range(len(rows))]
    df = pd.DataFrame(rows, columns=["open", "high", "low", "close"], index=index)
    return df


class TestDetectBullishFVG:
    """Tests for bullish FVG detection."""

    def test_basic_bullish_fvg(self):
        """A clear bullish gap: candle[0].high < candle[2].low."""
        # candle 0: high=100, candle 2: low=105 → gap between 100 and 105
        rows = [
            (95, 100, 90, 98),   # candle 0
            (99, 110, 99, 108),  # candle 1 (middle - big move up)
            (107, 112, 105, 110),  # candle 2
            (110, 115, 109, 114),  # candle 3 (still forming - excluded)
        ]
        df = _make_df(rows)
        detector = FVGDetector()
        fvgs = detector.detect(df, "5min")

        assert len(fvgs) == 1
        fvg = fvgs[0]
        assert fvg.type == "bullish"
        assert fvg.zone_lower == 100.0  # candle[0].high
        assert fvg.zone_upper == 105.0  # candle[2].low
        assert fvg.source_tf == "5min"
        assert fvg.formation_ts == df.index[1]  # middle candle timestamp

    def test_bullish_fvg_zone_validity(self):
        """Zone upper must be greater than zone lower."""
        rows = [
            (90, 95, 85, 93),
            (94, 115, 94, 112),
            (110, 120, 100, 118),
            (118, 125, 117, 123),
        ]
        df = _make_df(rows)
        detector = FVGDetector()
        fvgs = detector.detect(df, "15min")

        for fvg in fvgs:
            if fvg.type == "bullish":
                assert fvg.zone_upper > fvg.zone_lower


class TestDetectBearishFVG:
    """Tests for bearish FVG detection."""

    def test_basic_bearish_fvg(self):
        """A clear bearish gap: candle[0].low > candle[2].high."""
        # candle 0: low=110, candle 2: high=105 → gap between 105 and 110
        rows = [
            (115, 120, 110, 112),  # candle 0
            (111, 111, 100, 102),  # candle 1 (middle - big move down)
            (103, 105, 98, 100),   # candle 2
            (100, 102, 96, 97),    # candle 3 (still forming)
        ]
        df = _make_df(rows)
        detector = FVGDetector()
        fvgs = detector.detect(df, "60min")

        assert len(fvgs) == 1
        fvg = fvgs[0]
        assert fvg.type == "bearish"
        assert fvg.zone_upper == 110.0  # candle[0].low
        assert fvg.zone_lower == 105.0  # candle[2].high
        assert fvg.source_tf == "60min"
        assert fvg.formation_ts == df.index[1]

    def test_bearish_fvg_zone_validity(self):
        """Zone upper must be greater than zone lower for bearish FVGs."""
        rows = [
            (120, 125, 115, 118),
            (117, 117, 100, 102),
            (103, 108, 95, 100),
            (100, 103, 94, 96),
        ]
        df = _make_df(rows)
        detector = FVGDetector()
        fvgs = detector.detect(df, "5min")

        for fvg in fvgs:
            if fvg.type == "bearish":
                assert fvg.zone_upper > fvg.zone_lower


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_less_than_3_bars_returns_empty(self):
        """Fewer than 3 bars should return empty list without exception."""
        detector = FVGDetector()

        # 0 bars
        df_empty = pd.DataFrame(columns=["open", "high", "low", "close"])
        assert detector.detect(df_empty, "5min") == []

        # 1 bar
        df_one = _make_df([(100, 105, 95, 102)])
        assert detector.detect(df_one, "5min") == []

        # 2 bars
        df_two = _make_df([(100, 105, 95, 102), (103, 108, 98, 106)])
        assert detector.detect(df_two, "5min") == []

    def test_exactly_3_bars_no_fvg(self):
        """3 bars but no gap — returns empty list (last bar is still forming)."""
        # With exactly 3 bars, the last valid window would need candle[i+2]
        # to be iloc[-2] which is index 1. So i=0, i+2=2 but iloc[-2]=index 1.
        # Actually with 3 bars, max_i = 3-3 = 0, so range(0) is empty.
        rows = [
            (100, 105, 95, 102),
            (102, 107, 100, 106),
            (106, 110, 104, 108),
        ]
        df = _make_df(rows)
        detector = FVGDetector()
        fvgs = detector.detect(df, "5min")
        assert fvgs == []

    def test_exactly_4_bars_with_fvg(self):
        """4 bars: one valid window (i=0, checking candles 0,1,2 which ends at iloc[-2])."""
        rows = [
            (95, 100, 90, 98),     # candle 0
            (99, 110, 99, 108),    # candle 1
            (107, 112, 105, 110),  # candle 2 = iloc[-2]
            (110, 115, 109, 114),  # candle 3 = iloc[-1] (forming)
        ]
        df = _make_df(rows)
        detector = FVGDetector()
        fvgs = detector.detect(df, "5min")
        assert len(fvgs) == 1
        assert fvgs[0].type == "bullish"

    def test_nan_in_ohlc_skips_window(self):
        """Windows containing NaN values in OHLC should be skipped."""
        rows = [
            (95, 100, 90, 98),
            (99, float("nan"), 99, 108),  # NaN in high
            (107, 112, 105, 110),
            (110, 115, 105, 112),
            (110, 115, 109, 114),
        ]
        df = _make_df(rows)
        detector = FVGDetector()
        fvgs = detector.detect(df, "5min")

        # Windows involving the NaN row (index 1) should be skipped:
        # Window (0,1,2) - candle 1 has NaN → skipped
        # Window (1,2,3) - candle 1 (iloc[1]) has NaN → skipped
        # Only window (2,3,4) is valid but index 4 is iloc[-1] (forming) so
        # max_i = 5-3 = 2, range(2) gives i=0,1
        # i=0: window (0,1,2) → skipped (NaN)
        # i=1: window (1,2,3) → skipped (NaN)
        assert fvgs == []

    def test_nan_not_in_window_still_detects(self):
        """FVGs in valid windows should still be detected even if other rows have NaN."""
        rows = [
            (95, float("nan"), 90, 98),   # candle 0 - NaN
            (95, 100, 90, 98),            # candle 1
            (99, 110, 99, 108),           # candle 2
            (107, 112, 105, 110),         # candle 3
            (110, 115, 109, 114),         # candle 4 (forming)
        ]
        df = _make_df(rows)
        detector = FVGDetector()
        fvgs = detector.detect(df, "5min")

        # max_i = 5-3 = 2, range(2) → i=0, i=1
        # i=0: window (0,1,2) → skipped (candle 0 has NaN)
        # i=1: window (1,2,3) → candle[1].high=100 < candle[3].low=105 → bullish FVG
        assert len(fvgs) == 1
        assert fvgs[0].type == "bullish"

    def test_no_fvg_when_candles_overlap(self):
        """Normal overlapping candles produce no FVG."""
        rows = [
            (100, 105, 95, 102),
            (102, 107, 100, 106),
            (106, 110, 104, 108),
            (108, 112, 106, 110),
            (110, 114, 108, 112),
        ]
        df = _make_df(rows)
        detector = FVGDetector()
        fvgs = detector.detect(df, "5min")
        assert fvgs == []

    def test_multiple_fvgs_detected(self):
        """Multiple FVGs in a single DataFrame."""
        rows = [
            (95, 100, 90, 98),     # candle 0
            (99, 110, 99, 108),    # candle 1 (middle of bullish)
            (107, 112, 105, 110),  # candle 2 - bullish gap (100 < 105)
            (112, 115, 110, 113),  # candle 3
            (113, 113, 100, 101),  # candle 4 (middle of bearish)
            (100, 105, 95, 98),    # candle 5 - bearish gap (110 > 105)
            (98, 102, 94, 96),     # candle 6 (forming)
        ]
        df = _make_df(rows)
        detector = FVGDetector()
        fvgs = detector.detect(df, "5min")

        # max_i = 7-3 = 4, range(4) → i=0,1,2,3
        # i=0: 100 < 105 → bullish
        # i=1: check 99 < 110 and 110 > 105 — nope, high=110 not < low=110
        # i=2: high=112, low=110 — check if 112 < 110 (no) or 105 > 115 (no)
        # i=3: low=110, high=105 — check if 115 < 95 (no) or 110 > 105 (yes!) → bearish
        bullish = [f for f in fvgs if f.type == "bullish"]
        bearish = [f for f in fvgs if f.type == "bearish"]
        assert len(bullish) >= 1
        assert len(bearish) >= 1

    def test_last_bar_excluded(self):
        """The last bar (still forming) should never be part of a detected FVG window."""
        # Create a scenario where a FVG exists only if the last bar is included
        rows = [
            (100, 105, 95, 102),   # candle 0
            (102, 107, 100, 106),  # candle 1
            (106, 110, 104, 108),  # candle 2
            (108, 112, 106, 110),  # candle 3
            # Gap only with candles 2,3,4: high[2]=110 vs low[4]=115 (110<115 → bullish)
            (113, 118, 115, 116),  # candle 4 (forming - should be excluded)
        ]
        df = _make_df(rows)
        detector = FVGDetector()
        fvgs = detector.detect(df, "5min")

        # max_i = 5-3 = 2, range(2) → i=0, i=1
        # i=0: window (0,1,2): high[0]=105 vs low[2]=104 → 105 < 104? No
        # i=1: window (1,2,3): high[1]=107 vs low[3]=106 → 107 < 106? No
        # The FVG with candle 4 is not detected since it's the forming bar
        assert fvgs == []


class TestFormationTimestamp:
    """Tests for formation_ts assignment."""

    def test_formation_ts_is_middle_candle(self):
        """formation_ts should be the timestamp of candle[i+1] (middle candle)."""
        start = datetime(2024, 6, 15, 10, 0)
        rows = [
            (95, 100, 90, 98),
            (99, 110, 99, 108),
            (107, 112, 105, 110),
            (110, 115, 109, 114),
        ]
        df = _make_df(rows, start_ts=start)
        detector = FVGDetector()
        fvgs = detector.detect(df, "5min")

        assert len(fvgs) == 1
        expected_ts = start + timedelta(minutes=5)  # index 1
        assert fvgs[0].formation_ts == expected_ts


class TestSourceTimeframe:
    """Tests for source_tf field."""

    def test_source_tf_matches_input(self):
        """source_tf should match the timeframe parameter."""
        rows = [
            (95, 100, 90, 98),
            (99, 110, 99, 108),
            (107, 112, 105, 110),
            (110, 115, 109, 114),
        ]
        df = _make_df(rows)
        detector = FVGDetector()

        for tf in ["60min", "15min", "5min"]:
            fvgs = detector.detect(df, tf)
            assert len(fvgs) == 1
            assert fvgs[0].source_tf == tf


# ============================================================
# Tests for FVGDetector.update_fill_status()
# ============================================================


class TestUpdateFillStatusBullish:
    """Tests for bullish FVG fill tracking."""

    def test_bullish_partial_fill(self):
        """Bar high >= zone_lower but < zone_upper → partial fill, zone narrows."""
        fvg = FVG(
            type="bullish",
            zone_upper=110.0,
            zone_lower=100.0,
            formation_ts=datetime(2024, 1, 1, 9, 0),
            source_tf="5min",
        )
        # Bar high = 105 (between 100 and 110) → partial fill
        rows = [(98, 105, 95, 103)]
        df = _make_df(rows, start_ts=datetime(2024, 1, 1, 9, 5))
        detector = FVGDetector()

        result = detector.update_fill_status([fvg], df)
        assert len(result) == 1
        assert result[0].fill_status == "partial"
        assert result[0].zone_lower == 105.0  # narrowed to bar high
        assert result[0].zone_upper == 110.0  # unchanged

    def test_bullish_full_fill(self):
        """Bar high >= zone_upper → fully filled, removed from active set."""
        fvg = FVG(
            type="bullish",
            zone_upper=110.0,
            zone_lower=100.0,
            formation_ts=datetime(2024, 1, 1, 9, 0),
            source_tf="5min",
        )
        # Bar high = 112 (>= 110) → fully filled
        rows = [(105, 112, 102, 110)]
        df = _make_df(rows, start_ts=datetime(2024, 1, 1, 9, 5))
        detector = FVGDetector()

        result = detector.update_fill_status([fvg], df)
        assert len(result) == 0
        assert fvg.fill_status == "filled"

    def test_bullish_exact_upper_boundary_fills(self):
        """Bar high exactly at zone_upper → fully filled."""
        fvg = FVG(
            type="bullish",
            zone_upper=110.0,
            zone_lower=100.0,
            formation_ts=datetime(2024, 1, 1, 9, 0),
            source_tf="5min",
        )
        rows = [(105, 110, 102, 108)]
        df = _make_df(rows, start_ts=datetime(2024, 1, 1, 9, 5))
        detector = FVGDetector()

        result = detector.update_fill_status([fvg], df)
        assert len(result) == 0
        assert fvg.fill_status == "filled"

    def test_bullish_exact_lower_boundary_partial(self):
        """Bar high exactly at zone_lower → partial fill."""
        fvg = FVG(
            type="bullish",
            zone_upper=110.0,
            zone_lower=100.0,
            formation_ts=datetime(2024, 1, 1, 9, 0),
            source_tf="5min",
        )
        rows = [(95, 100, 90, 98)]
        df = _make_df(rows, start_ts=datetime(2024, 1, 1, 9, 5))
        detector = FVGDetector()

        result = detector.update_fill_status([fvg], df)
        assert len(result) == 1
        assert result[0].fill_status == "partial"
        assert result[0].zone_lower == 100.0

    def test_bullish_no_fill(self):
        """Bar high < zone_lower → no fill, FVG unchanged."""
        fvg = FVG(
            type="bullish",
            zone_upper=110.0,
            zone_lower=100.0,
            formation_ts=datetime(2024, 1, 1, 9, 0),
            source_tf="5min",
        )
        # Bar high = 95 (< 100) → no fill
        rows = [(90, 95, 85, 93)]
        df = _make_df(rows, start_ts=datetime(2024, 1, 1, 9, 5))
        detector = FVGDetector()

        result = detector.update_fill_status([fvg], df)
        assert len(result) == 1
        assert result[0].fill_status == "unfilled"
        assert result[0].zone_lower == 100.0
        assert result[0].zone_upper == 110.0


class TestUpdateFillStatusBearish:
    """Tests for bearish FVG fill tracking."""

    def test_bearish_partial_fill(self):
        """Bar low <= zone_upper but > zone_lower → partial fill, zone narrows."""
        fvg = FVG(
            type="bearish",
            zone_upper=110.0,
            zone_lower=100.0,
            formation_ts=datetime(2024, 1, 1, 9, 0),
            source_tf="5min",
        )
        # Bar low = 105 (between 100 and 110) → partial fill
        rows = [(112, 115, 105, 108)]
        df = _make_df(rows, start_ts=datetime(2024, 1, 1, 9, 5))
        detector = FVGDetector()

        result = detector.update_fill_status([fvg], df)
        assert len(result) == 1
        assert result[0].fill_status == "partial"
        assert result[0].zone_upper == 105.0  # narrowed to bar low
        assert result[0].zone_lower == 100.0  # unchanged

    def test_bearish_full_fill(self):
        """Bar low <= zone_lower → fully filled, removed from active set."""
        fvg = FVG(
            type="bearish",
            zone_upper=110.0,
            zone_lower=100.0,
            formation_ts=datetime(2024, 1, 1, 9, 0),
            source_tf="5min",
        )
        # Bar low = 98 (<= 100) → fully filled
        rows = [(105, 108, 98, 100)]
        df = _make_df(rows, start_ts=datetime(2024, 1, 1, 9, 5))
        detector = FVGDetector()

        result = detector.update_fill_status([fvg], df)
        assert len(result) == 0
        assert fvg.fill_status == "filled"

    def test_bearish_exact_lower_boundary_fills(self):
        """Bar low exactly at zone_lower → fully filled."""
        fvg = FVG(
            type="bearish",
            zone_upper=110.0,
            zone_lower=100.0,
            formation_ts=datetime(2024, 1, 1, 9, 0),
            source_tf="5min",
        )
        rows = [(105, 108, 100, 103)]
        df = _make_df(rows, start_ts=datetime(2024, 1, 1, 9, 5))
        detector = FVGDetector()

        result = detector.update_fill_status([fvg], df)
        assert len(result) == 0
        assert fvg.fill_status == "filled"

    def test_bearish_exact_upper_boundary_partial(self):
        """Bar low exactly at zone_upper → partial fill."""
        fvg = FVG(
            type="bearish",
            zone_upper=110.0,
            zone_lower=100.0,
            formation_ts=datetime(2024, 1, 1, 9, 0),
            source_tf="5min",
        )
        rows = [(115, 118, 110, 112)]
        df = _make_df(rows, start_ts=datetime(2024, 1, 1, 9, 5))
        detector = FVGDetector()

        result = detector.update_fill_status([fvg], df)
        assert len(result) == 1
        assert result[0].fill_status == "partial"
        assert result[0].zone_upper == 110.0

    def test_bearish_no_fill(self):
        """Bar low > zone_upper → no fill, FVG unchanged."""
        fvg = FVG(
            type="bearish",
            zone_upper=110.0,
            zone_lower=100.0,
            formation_ts=datetime(2024, 1, 1, 9, 0),
            source_tf="5min",
        )
        # Bar low = 115 (> 110) → no fill
        rows = [(118, 120, 115, 117)]
        df = _make_df(rows, start_ts=datetime(2024, 1, 1, 9, 5))
        detector = FVGDetector()

        result = detector.update_fill_status([fvg], df)
        assert len(result) == 1
        assert result[0].fill_status == "unfilled"
        assert result[0].zone_upper == 110.0
        assert result[0].zone_lower == 100.0


class TestUpdateFillStatusExpiry:
    """Tests for FVG age expiry."""

    def test_fvg_expires_after_max_age(self):
        """FVG with age_bars exceeding max_age is removed."""
        fvg = FVG(
            type="bullish",
            zone_upper=110.0,
            zone_lower=100.0,
            formation_ts=datetime(2024, 1, 1, 9, 0),
            source_tf="5min",
            age_bars=48,  # Start near max
        )
        # Process 3 bars → age goes to 51 (> 50)
        rows = [
            (90, 95, 85, 93),
            (91, 94, 86, 92),
            (92, 96, 87, 94),
        ]
        df = _make_df(rows, start_ts=datetime(2024, 1, 1, 9, 5))
        detector = FVGDetector()

        result = detector.update_fill_status([fvg], df, max_age=50)
        assert len(result) == 0

    def test_fvg_at_exact_max_age_not_expired(self):
        """FVG at exactly max_age is still active (expires when > max_age)."""
        fvg = FVG(
            type="bullish",
            zone_upper=110.0,
            zone_lower=100.0,
            formation_ts=datetime(2024, 1, 1, 9, 0),
            source_tf="5min",
            age_bars=49,  # After 1 bar → age = 50 (== max_age, not expired)
        )
        rows = [(90, 95, 85, 93)]
        df = _make_df(rows, start_ts=datetime(2024, 1, 1, 9, 5))
        detector = FVGDetector()

        result = detector.update_fill_status([fvg], df, max_age=50)
        assert len(result) == 1
        assert result[0].age_bars == 50

    def test_custom_max_age(self):
        """Custom max_age value is respected."""
        fvg = FVG(
            type="bearish",
            zone_upper=110.0,
            zone_lower=100.0,
            formation_ts=datetime(2024, 1, 1, 9, 0),
            source_tf="5min",
            age_bars=8,
        )
        # 3 bars → age = 11 (> 10)
        rows = [
            (115, 118, 112, 116),
            (116, 119, 113, 117),
            (117, 120, 114, 118),
        ]
        df = _make_df(rows, start_ts=datetime(2024, 1, 1, 9, 5))
        detector = FVGDetector()

        result = detector.update_fill_status([fvg], df, max_age=10)
        assert len(result) == 0


class TestUpdateFillStatusChronological:
    """Tests for chronological processing and incremental updates."""

    def test_incremental_partial_then_filled(self):
        """Multiple bars: first partially fills, then fully fills."""
        fvg = FVG(
            type="bullish",
            zone_upper=110.0,
            zone_lower=100.0,
            formation_ts=datetime(2024, 1, 1, 9, 0),
            source_tf="5min",
        )
        rows = [
            (98, 105, 95, 103),   # bar 1: high=105, partial fill (zone_lower → 105)
            (103, 112, 100, 110), # bar 2: high=112 >= zone_upper → filled
        ]
        df = _make_df(rows, start_ts=datetime(2024, 1, 1, 9, 5))
        detector = FVGDetector()

        result = detector.update_fill_status([fvg], df)
        assert len(result) == 0
        assert fvg.fill_status == "filled"

    def test_multiple_partial_fills_narrow_zone(self):
        """Multiple bars each partially fill, narrowing zone progressively."""
        fvg = FVG(
            type="bullish",
            zone_upper=110.0,
            zone_lower=100.0,
            formation_ts=datetime(2024, 1, 1, 9, 0),
            source_tf="5min",
        )
        rows = [
            (98, 103, 95, 101),  # high=103 → zone_lower = 103
            (100, 106, 98, 104), # high=106 → zone_lower = 106
            (104, 108, 102, 107), # high=108 → zone_lower = 108
        ]
        df = _make_df(rows, start_ts=datetime(2024, 1, 1, 9, 5))
        detector = FVGDetector()

        result = detector.update_fill_status([fvg], df)
        assert len(result) == 1
        assert result[0].fill_status == "partial"
        assert result[0].zone_lower == 108.0
        assert result[0].zone_upper == 110.0

    def test_age_increments_per_bar(self):
        """Age is incremented for each bar processed."""
        fvg = FVG(
            type="bullish",
            zone_upper=110.0,
            zone_lower=100.0,
            formation_ts=datetime(2024, 1, 1, 9, 0),
            source_tf="5min",
            age_bars=0,
        )
        rows = [
            (90, 95, 85, 93),
            (91, 94, 86, 92),
            (92, 96, 87, 94),
            (93, 97, 88, 95),
            (94, 98, 89, 96),
        ]
        df = _make_df(rows, start_ts=datetime(2024, 1, 1, 9, 5))
        detector = FVGDetector()

        result = detector.update_fill_status([fvg], df)
        assert len(result) == 1
        assert result[0].age_bars == 5

    def test_multiple_fvgs_independent_tracking(self):
        """Multiple FVGs are tracked independently."""
        bullish_fvg = FVG(
            type="bullish",
            zone_upper=110.0,
            zone_lower=100.0,
            formation_ts=datetime(2024, 1, 1, 9, 0),
            source_tf="5min",
        )
        bearish_fvg = FVG(
            type="bearish",
            zone_upper=90.0,
            zone_lower=80.0,
            formation_ts=datetime(2024, 1, 1, 9, 0),
            source_tf="5min",
        )
        # Bar: high=105 partially fills bullish, low=85 partially fills bearish
        rows = [(95, 105, 85, 98)]
        df = _make_df(rows, start_ts=datetime(2024, 1, 1, 9, 5))
        detector = FVGDetector()

        result = detector.update_fill_status([bullish_fvg, bearish_fvg], df)
        assert len(result) == 2
        assert bullish_fvg.fill_status == "partial"
        assert bullish_fvg.zone_lower == 105.0
        assert bearish_fvg.fill_status == "partial"
        assert bearish_fvg.zone_upper == 85.0

    def test_filled_fvg_removed_immediately(self):
        """Fully filled FVG is removed and not processed in subsequent bars."""
        fvg = FVG(
            type="bullish",
            zone_upper=110.0,
            zone_lower=100.0,
            formation_ts=datetime(2024, 1, 1, 9, 0),
            source_tf="5min",
        )
        rows = [
            (105, 115, 100, 112),  # bar 1: high=115 >= 110 → filled, removed
            (110, 120, 108, 118),  # bar 2: should not be processed for this FVG
        ]
        df = _make_df(rows, start_ts=datetime(2024, 1, 1, 9, 5))
        detector = FVGDetector()

        result = detector.update_fill_status([fvg], df)
        assert len(result) == 0
        assert fvg.fill_status == "filled"
        assert fvg.age_bars == 1  # Only processed 1 bar before removal


class TestUpdateFillStatusEdgeCases:
    """Edge case tests for update_fill_status."""

    def test_empty_fvg_list(self):
        """Empty FVG list returns empty list."""
        rows = [(100, 105, 95, 102)]
        df = _make_df(rows)
        detector = FVGDetector()

        result = detector.update_fill_status([], df)
        assert result == []

    def test_empty_dataframe(self):
        """Empty DataFrame returns FVGs unchanged."""
        fvg = FVG(
            type="bullish",
            zone_upper=110.0,
            zone_lower=100.0,
            formation_ts=datetime(2024, 1, 1, 9, 0),
            source_tf="5min",
        )
        df = pd.DataFrame(columns=["open", "high", "low", "close"])
        detector = FVGDetector()

        result = detector.update_fill_status([fvg], df)
        assert len(result) == 1
        assert result[0].fill_status == "unfilled"

    def test_none_dataframe(self):
        """None DataFrame returns FVGs unchanged."""
        fvg = FVG(
            type="bullish",
            zone_upper=110.0,
            zone_lower=100.0,
            formation_ts=datetime(2024, 1, 1, 9, 0),
            source_tf="5min",
        )
        detector = FVGDetector()

        result = detector.update_fill_status([fvg], None)
        assert len(result) == 1
        assert result[0].fill_status == "unfilled"

    def test_nan_bars_skipped(self):
        """Bars with NaN in high/low are skipped without affecting FVGs."""
        fvg = FVG(
            type="bullish",
            zone_upper=110.0,
            zone_lower=100.0,
            formation_ts=datetime(2024, 1, 1, 9, 0),
            source_tf="5min",
        )
        rows = [
            (98, float("nan"), 95, 97),  # NaN high → skipped
            (98, 105, 95, 103),          # valid → partial fill
        ]
        df = _make_df(rows, start_ts=datetime(2024, 1, 1, 9, 5))
        detector = FVGDetector()

        result = detector.update_fill_status([fvg], df)
        assert len(result) == 1
        assert result[0].fill_status == "partial"
        assert result[0].zone_lower == 105.0
        # Age only incremented for the valid bar
        assert result[0].age_bars == 1

    def test_default_max_age_is_50(self):
        """Default max_age parameter is 50."""
        fvg = FVG(
            type="bullish",
            zone_upper=110.0,
            zone_lower=100.0,
            formation_ts=datetime(2024, 1, 1, 9, 0),
            source_tf="5min",
            age_bars=50,
        )
        # 1 bar → age = 51 (> 50 default)
        rows = [(90, 95, 85, 93)]
        df = _make_df(rows, start_ts=datetime(2024, 1, 1, 9, 5))
        detector = FVGDetector()

        result = detector.update_fill_status([fvg], df)
        assert len(result) == 0
