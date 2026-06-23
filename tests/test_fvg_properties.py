"""Property-based tests for FVG data model using Hypothesis.

Validates correctness properties defined in the design document.
"""

import copy
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from hypothesis import assume, given, settings
from hypothesis.strategies import (
    composite,
    datetimes,
    floats,
    integers,
    lists,
    sampled_from,
)

from strategy.fvg_bias import BiasCalculator
from strategy.fvg_detector import Bias, FVG, FVGDetector
from strategy.fvg_signal import SignalGenerator


# ---------------------------------------------------------------------------
# Strategies (generators)
# ---------------------------------------------------------------------------

@composite
def fvg_objects(draw):
    """Generate random valid FVG objects with realistic gold-price zones."""
    fvg_type = draw(sampled_from(["bullish", "bearish"]))

    # Generate two distinct floats for zone boundaries ensuring upper > lower > 0
    zone_a = draw(floats(min_value=3000.0, max_value=5000.0, allow_nan=False, allow_infinity=False))
    zone_b = draw(floats(min_value=3000.0, max_value=5000.0, allow_nan=False, allow_infinity=False))

    # Ensure they are distinct; if equal, nudge one up
    if zone_a == zone_b:
        zone_b = zone_a + 0.01

    zone_upper = max(zone_a, zone_b)
    zone_lower = min(zone_a, zone_b)

    formation_ts = draw(datetimes(
        min_value=datetime(2020, 1, 1),
        max_value=datetime(2030, 12, 31),
    ))
    source_tf = draw(sampled_from(["60min", "15min", "5min"]))
    fill_status = draw(sampled_from(["unfilled", "partial", "filled"]))
    age_bars = draw(integers(min_value=0, max_value=1000))

    return FVG(
        type=fvg_type,
        zone_upper=zone_upper,
        zone_lower=zone_lower,
        formation_ts=formation_ts,
        source_tf=source_tf,
        fill_status=fill_status,
        age_bars=age_bars,
    )


@composite
def ohlc_dataframes(draw, min_rows=3, max_rows=500):
    """Generate random OHLC DataFrames with realistic gold prices (3000-5000).

    Produces valid OHLC bars where high >= open, close and low <= open, close.
    Index is a DatetimeIndex with 5-minute frequency.
    """
    n_rows = draw(integers(min_value=min_rows, max_value=max_rows))

    rows = []
    for _ in range(n_rows):
        open_price = draw(floats(min_value=3000.0, max_value=5000.0, allow_nan=False, allow_infinity=False))
        close_price = draw(floats(min_value=3000.0, max_value=5000.0, allow_nan=False, allow_infinity=False))
        # High must be >= max(open, close), low must be <= min(open, close)
        high_price = draw(floats(
            min_value=max(open_price, close_price),
            max_value=5000.0,
            allow_nan=False, allow_infinity=False,
        ))
        low_price = draw(floats(
            min_value=3000.0,
            max_value=min(open_price, close_price),
            allow_nan=False, allow_infinity=False,
        ))
        rows.append((open_price, high_price, low_price, close_price))

    index = pd.date_range(start="2024-01-01", periods=n_rows, freq="5min")
    df = pd.DataFrame(rows, columns=["open", "high", "low", "close"], index=index)
    return df


# ---------------------------------------------------------------------------
# Property 2: Detection Completeness
# Validates: Requirements 1.1
# ---------------------------------------------------------------------------

class TestFVGDetectionCompleteness:
    """Property 2: For any OHLC DataFrame with N valid bars (N >= 3),
    the number of detected FVGs is at most N-2.

    **Validates: Requirements 1.1**
    """

    @given(df=ohlc_dataframes(min_rows=3, max_rows=500))
    @settings(max_examples=50)
    def test_fvg_count_upper_bound(self, df: pd.DataFrame):
        """Number of detected FVGs must be <= len(df) - 2 for all DataFrames."""
        detector = FVGDetector()
        fvgs = detector.detect(df, "5min")
        assert len(fvgs) <= len(df) - 2


# ---------------------------------------------------------------------------
# Property 8: Round-Trip Serialization
# Validates: Requirements 1.7
# ---------------------------------------------------------------------------

class TestFVGRoundTripSerialization:
    """Property 8: FVG.from_dict(f.to_dict()) == f for all valid FVG objects.

    **Validates: Requirements 1.7**
    """

    @given(fvg=fvg_objects())
    @settings(max_examples=50)
    def test_round_trip_preserves_equality(self, fvg: FVG):
        """Serializing then deserializing an FVG must produce an equal object."""
        serialized = fvg.to_dict()
        deserialized = FVG.from_dict(serialized)
        assert deserialized == fvg

    @given(fvg=fvg_objects())
    @settings(max_examples=30)
    def test_to_dict_returns_dict(self, fvg: FVG):
        """to_dict() must return a plain dict with expected keys."""
        d = fvg.to_dict()
        assert isinstance(d, dict)
        expected_keys = {"type", "zone_upper", "zone_lower", "formation_ts", "source_tf", "fill_status", "age_bars"}
        assert set(d.keys()) == expected_keys

    @given(fvg=fvg_objects())
    @settings(max_examples=30)
    def test_serialized_formation_ts_is_iso_string(self, fvg: FVG):
        """formation_ts in the dict must be an ISO-format string parseable by fromisoformat."""
        d = fvg.to_dict()
        assert isinstance(d["formation_ts"], str)
        # Verify it round-trips through fromisoformat
        parsed = datetime.fromisoformat(d["formation_ts"])
        assert parsed == fvg.formation_ts


# ---------------------------------------------------------------------------
# Property 1: FVG Zone Validity
# Validates: Requirements 1.2, 1.3
# ---------------------------------------------------------------------------

class TestFVGZoneValidity:
    """Property 1: All detected FVGs must have zone_upper > zone_lower > 0.

    **Validates: Requirements 1.2, 1.3**
    """

    @given(df=ohlc_dataframes(min_rows=100, max_rows=500))
    @settings(max_examples=20, deadline=None)
    def test_zone_upper_greater_than_zone_lower(self, df: pd.DataFrame):
        """For every detected FVG, zone_upper > zone_lower > 0."""
        detector = FVGDetector()
        fvgs = detector.detect(df, "5min")

        for fvg in fvgs:
            assert fvg.zone_upper > fvg.zone_lower, (
                f"FVG zone_upper ({fvg.zone_upper}) must be > zone_lower ({fvg.zone_lower})"
            )
            assert fvg.zone_lower > 0, (
                f"FVG zone_lower ({fvg.zone_lower}) must be > 0"
            )
            assert fvg.zone_upper > 0, (
                f"FVG zone_upper ({fvg.zone_upper}) must be > 0"
            )


# ---------------------------------------------------------------------------
# Property 3: Fill Monotonicity
# Validates: Requirements 2.1, 2.2
# ---------------------------------------------------------------------------

# Strategies for fill monotonicity tests

@composite
def unfilled_fvg_objects(draw):
    """Generate random unfilled FVG objects with realistic gold-price zones.

    These start as 'unfilled' to allow tracking status progression.
    """
    fvg_type = draw(sampled_from(["bullish", "bearish"]))

    # Generate two distinct floats for zone boundaries ensuring upper > lower > 0
    zone_a = draw(floats(min_value=3000.0, max_value=5000.0, allow_nan=False, allow_infinity=False))
    zone_b = draw(floats(min_value=3000.0, max_value=5000.0, allow_nan=False, allow_infinity=False))

    # Ensure they are distinct; if equal, nudge one up
    if zone_a == zone_b:
        zone_b = zone_a + 0.01

    zone_upper = max(zone_a, zone_b)
    zone_lower = min(zone_a, zone_b)

    formation_ts = draw(datetimes(
        min_value=datetime(2020, 1, 1),
        max_value=datetime(2030, 12, 31),
    ))
    source_tf = draw(sampled_from(["60min", "15min", "5min"]))

    return FVG(
        type=fvg_type,
        zone_upper=zone_upper,
        zone_lower=zone_lower,
        formation_ts=formation_ts,
        source_tf=source_tf,
        fill_status="unfilled",
        age_bars=0,
    )


@composite
def ohlc_bar_sequences(draw, min_bars=5, max_bars=50):
    """Generate a sequence of OHLC bars for fill tracking.

    Produces valid OHLC bars where high >= max(open, close)
    and low <= min(open, close), with realistic gold prices.
    """
    n_bars = draw(integers(min_value=min_bars, max_value=max_bars))

    rows = []
    for _ in range(n_bars):
        open_price = draw(floats(min_value=3000.0, max_value=5000.0, allow_nan=False, allow_infinity=False))
        close_price = draw(floats(min_value=3000.0, max_value=5000.0, allow_nan=False, allow_infinity=False))
        high_price = draw(floats(
            min_value=max(open_price, close_price),
            max_value=5000.0,
            allow_nan=False, allow_infinity=False,
        ))
        low_price = draw(floats(
            min_value=3000.0,
            max_value=min(open_price, close_price),
            allow_nan=False, allow_infinity=False,
        ))
        rows.append((open_price, high_price, low_price, close_price))

    index = pd.date_range(start="2024-01-01", periods=n_bars, freq="5min")
    df = pd.DataFrame(rows, columns=["open", "high", "low", "close"], index=index)
    return df


class TestFVGFillMonotonicity:
    """Property 3: Fill status only moves forward: unfilled → partial → filled.

    Once an FVG is marked "filled", it SHALL NOT revert to "unfilled" or "partial"
    on any subsequent bar update. Valid transitions:
      unfilled→unfilled, unfilled→partial, unfilled→filled,
      partial→partial, partial→filled, filled→filled.
    Never: partial→unfilled, filled→unfilled, filled→partial.

    **Validates: Requirements 2.1, 2.2**
    """

    # Ordered status levels for monotonicity check
    STATUS_ORDER = {"unfilled": 0, "partial": 1, "filled": 2}

    @given(fvg=unfilled_fvg_objects(), bars=ohlc_bar_sequences(min_bars=5, max_bars=50))
    @settings(max_examples=50, deadline=None)
    def test_fill_status_never_regresses(self, fvg: FVG, bars: pd.DataFrame):
        """Fill status must only advance forward through unfilled → partial → filled.

        Process bars one at a time and verify status never goes backward.
        """
        detector = FVGDetector()

        # Use a very large max_age so expiry doesn't interfere with fill tracking
        max_age = 10000

        # Track the highest status level reached
        highest_status = self.STATUS_ORDER[fvg.fill_status]

        for i in range(len(bars)):
            single_bar_df = bars.iloc[i : i + 1]

            # Process one bar at a time, keeping FVG in a list
            fvg_list = [fvg]
            remaining = detector.update_fill_status(fvg_list, single_bar_df, max_age=max_age)

            # Get current status: if FVG was removed (filled), it's "filled"
            if fvg not in remaining:
                current_status = "filled"
            else:
                current_status = fvg.fill_status

            current_level = self.STATUS_ORDER[current_status]

            # Assert monotonicity: status level must not decrease
            assert current_level >= highest_status, (
                f"Fill status regressed from level {highest_status} "
                f"({self._level_name(highest_status)}) to level {current_level} "
                f"({current_status}) at bar {i}"
            )

            highest_status = current_level

            # Once filled, stop processing (FVG is removed from active set)
            if current_status == "filled":
                break

    @given(fvg=unfilled_fvg_objects(), bars=ohlc_bar_sequences(min_bars=5, max_bars=50))
    @settings(max_examples=50, deadline=None)
    def test_fill_status_batch_monotonicity(self, fvg: FVG, bars: pd.DataFrame):
        """Fill status is monotonic even when processing bars in batch.

        Process the full bar sequence at once and verify the final status
        is at or above the starting status (unfilled).
        """
        detector = FVGDetector()
        max_age = 10000

        initial_status = self.STATUS_ORDER[fvg.fill_status]

        fvg_list = [fvg]
        remaining = detector.update_fill_status(fvg_list, bars, max_age=max_age)

        if fvg not in remaining:
            final_status = "filled"
        else:
            final_status = fvg.fill_status

        final_level = self.STATUS_ORDER[final_status]

        assert final_level >= initial_status, (
            f"Fill status regressed from {self._level_name(initial_status)} "
            f"to {final_status} after batch processing {len(bars)} bars"
        )

    def _level_name(self, level: int) -> str:
        """Convert status level back to name for error messages."""
        for name, lvl in self.STATUS_ORDER.items():
            if lvl == level:
                return name
        return "unknown"


# ---------------------------------------------------------------------------
# Property 9: Age Expiry
# Validates: Requirements 2.4, 2.5
# ---------------------------------------------------------------------------

@composite
def non_filling_ohlc_dataframes(draw, min_rows=1, max_rows=100):
    """Generate OHLC DataFrames with prices far from typical FVG zones (3000-5000).

    Uses a very low price range (100-200) so that bars never touch FVG zones
    in the 3000-5000 range. This isolates the age expiry behavior from fill logic.
    """
    n_rows = draw(integers(min_value=min_rows, max_value=max_rows))

    rows = []
    for _ in range(n_rows):
        # Use prices in a range far below FVG zones (3000-5000)
        open_price = draw(floats(min_value=100.0, max_value=200.0, allow_nan=False, allow_infinity=False))
        close_price = draw(floats(min_value=100.0, max_value=200.0, allow_nan=False, allow_infinity=False))
        high_price = draw(floats(
            min_value=max(open_price, close_price),
            max_value=200.0,
            allow_nan=False, allow_infinity=False,
        ))
        low_price = draw(floats(
            min_value=100.0,
            max_value=min(open_price, close_price),
            allow_nan=False, allow_infinity=False,
        ))
        rows.append((open_price, high_price, low_price, close_price))

    index = pd.date_range(start="2024-01-01", periods=n_rows, freq="5min")
    df = pd.DataFrame(rows, columns=["open", "high", "low", "close"], index=index)
    return df


@composite
def fvg_list_with_ages(draw, min_size=1, max_size=10, min_age=0, max_age=100):
    """Generate a list of distinct FVG objects with configurable initial age_bars.

    Zones are in the 3000-5000 range so non-filling bars (100-200) won't touch them.
    Each FVG gets a unique formation timestamp to ensure distinctness.
    """
    n = draw(integers(min_value=min_size, max_value=max_size))
    fvgs = []
    base_ts = datetime(2024, 1, 1)

    for i in range(n):
        fvg_type = draw(sampled_from(["bullish", "bearish"]))

        zone_a = draw(floats(min_value=3000.0, max_value=5000.0, allow_nan=False, allow_infinity=False))
        zone_b = draw(floats(min_value=3000.0, max_value=5000.0, allow_nan=False, allow_infinity=False))

        if zone_a == zone_b:
            zone_b = zone_a + 0.01

        zone_upper = max(zone_a, zone_b)
        zone_lower = min(zone_a, zone_b)

        source_tf = draw(sampled_from(["60min", "15min", "5min"]))
        age_bars = draw(integers(min_value=min_age, max_value=max_age))

        # Use unique formation_ts per FVG to ensure list elements are distinct
        formation_ts = datetime(2024, 1, 1, i // 60, i % 60, 0)

        fvgs.append(FVG(
            type=fvg_type,
            zone_upper=zone_upper,
            zone_lower=zone_lower,
            formation_ts=formation_ts,
            source_tf=source_tf,
            fill_status="unfilled",
            age_bars=age_bars,
        ))

    return fvgs


class TestFVGAgeExpiry:
    """Property 9: No FVG with age_bars > max_age shall appear in the active set.

    **Validates: Requirements 2.4, 2.5**
    """

    @given(
        fvg_list=fvg_list_with_ages(min_size=1, max_size=10, min_age=0, max_age=100),
        df=non_filling_ohlc_dataframes(min_rows=1, max_rows=100),
        max_age=integers(min_value=5, max_value=100),
    )
    @settings(max_examples=50, deadline=None)
    def test_no_fvg_exceeds_max_age_in_active_set(self, fvg_list, df, max_age):
        """After update_fill_status, no active FVG should have age_bars > max_age."""
        detector = FVGDetector()

        # Deep copy to avoid mutating Hypothesis-generated data across examples
        fvgs = copy.deepcopy(fvg_list)

        active = detector.update_fill_status(fvgs, df, max_age=max_age)

        for fvg in active:
            assert fvg.age_bars <= max_age, (
                f"FVG with age_bars={fvg.age_bars} exceeds max_age={max_age} "
                f"but is still in active set"
            )


# ---------------------------------------------------------------------------
# Property 4: Bias Confidence Bounds
# Validates: Requirements 3.3, 3.4, 3.5
# ---------------------------------------------------------------------------

class TestBiasConfidenceBounds:
    """Property 4: For all bias calculations, 0.0 <= confidence <= 1.0.

    Both calculate_60min_bias and adjust_with_15min must always produce
    a confidence score within the [0.0, 1.0] range regardless of input.

    **Validates: Requirements 3.3, 3.4, 3.5**
    """

    @given(fvgs_60min=lists(fvg_objects(), min_size=0, max_size=20))
    @settings(max_examples=50, deadline=None)
    def test_60min_bias_confidence_bounded(self, fvgs_60min: list):
        """calculate_60min_bias must return confidence in [0.0, 1.0]."""
        calc = BiasCalculator()
        bias = calc.calculate_60min_bias(fvgs_60min)

        assert 0.0 <= bias.confidence <= 1.0, (
            f"60min bias confidence {bias.confidence} is out of bounds [0.0, 1.0] "
            f"for {len(fvgs_60min)} FVGs"
        )

    @given(
        fvgs_60min=lists(fvg_objects(), min_size=0, max_size=20),
        fvgs_15min=lists(fvg_objects(), min_size=0, max_size=20),
    )
    @settings(max_examples=50, deadline=None)
    def test_adjusted_bias_confidence_bounded(self, fvgs_60min: list, fvgs_15min: list):
        """After adjust_with_15min, confidence must still be in [0.0, 1.0]."""
        calc = BiasCalculator()

        # First calculate 60min bias
        bias = calc.calculate_60min_bias(fvgs_60min)
        assert 0.0 <= bias.confidence <= 1.0, (
            f"Initial 60min bias confidence {bias.confidence} out of bounds"
        )

        # Then adjust with 15min FVGs
        adjusted = calc.adjust_with_15min(bias, fvgs_15min)
        assert 0.0 <= adjusted.confidence <= 1.0, (
            f"Adjusted bias confidence {adjusted.confidence} is out of bounds [0.0, 1.0] "
            f"after adjusting with {len(fvgs_15min)} 15min FVGs "
            f"(initial confidence was {bias.confidence})"
        )


# ---------------------------------------------------------------------------
# Property 5: Signal-Bias Alignment
# Validates: Requirements 4.1, 4.2, 4.7
# ---------------------------------------------------------------------------

@composite
def bias_objects(draw):
    """Generate random Bias objects with direction and confidence."""
    direction = draw(sampled_from(["bullish", "bearish", "neutral"]))
    confidence = draw(floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False))
    return Bias(direction=direction, confidence=confidence)


@composite
def fvg_5min_objects(draw):
    """Generate random valid FVG objects specifically from 5min timeframe.

    These are used as trigger FVGs for signal generation.
    """
    fvg_type = draw(sampled_from(["bullish", "bearish"]))

    zone_a = draw(floats(min_value=3000.0, max_value=5000.0, allow_nan=False, allow_infinity=False))
    zone_b = draw(floats(min_value=3000.0, max_value=5000.0, allow_nan=False, allow_infinity=False))

    if zone_a == zone_b:
        zone_b = zone_a + 0.01

    zone_upper = max(zone_a, zone_b)
    zone_lower = min(zone_a, zone_b)

    formation_ts = draw(datetimes(
        min_value=datetime(2020, 1, 1),
        max_value=datetime(2030, 12, 31),
    ))
    fill_status = draw(sampled_from(["unfilled", "partial"]))
    age_bars = draw(integers(min_value=0, max_value=50))

    return FVG(
        type=fvg_type,
        zone_upper=zone_upper,
        zone_lower=zone_lower,
        formation_ts=formation_ts,
        source_tf="5min",
        fill_status=fill_status,
        age_bars=age_bars,
    )


@composite
def htf_fvg_objects(draw):
    """Generate random valid FVG objects from higher timeframes (15min/60min)."""
    fvg_type = draw(sampled_from(["bullish", "bearish"]))

    zone_a = draw(floats(min_value=3000.0, max_value=5000.0, allow_nan=False, allow_infinity=False))
    zone_b = draw(floats(min_value=3000.0, max_value=5000.0, allow_nan=False, allow_infinity=False))

    if zone_a == zone_b:
        zone_b = zone_a + 0.01

    zone_upper = max(zone_a, zone_b)
    zone_lower = min(zone_a, zone_b)

    formation_ts = draw(datetimes(
        min_value=datetime(2020, 1, 1),
        max_value=datetime(2030, 12, 31),
    ))
    source_tf = draw(sampled_from(["60min", "15min"]))
    fill_status = draw(sampled_from(["unfilled", "partial", "filled"]))
    age_bars = draw(integers(min_value=0, max_value=200))

    return FVG(
        type=fvg_type,
        zone_upper=zone_upper,
        zone_lower=zone_lower,
        formation_ts=formation_ts,
        source_tf=source_tf,
        fill_status=fill_status,
        age_bars=age_bars,
    )


class TestSignalBiasAlignment:
    """Property 5: Signal-Bias Alignment.

    If a signal is produced with side="BUY", then bias.direction == "bullish".
    If side="SELL", then bias.direction == "bearish".
    No signal is produced when bias is "neutral".

    **Validates: Requirements 4.1, 4.2, 4.7**
    """

    @given(
        fvgs_5min=lists(fvg_5min_objects(), min_size=0, max_size=10),
        bias=bias_objects(),
        fvgs_htf=lists(htf_fvg_objects(), min_size=0, max_size=10),
    )
    @settings(max_examples=50, deadline=None)
    def test_buy_signal_only_with_bullish_bias(self, fvgs_5min, bias, fvgs_htf):
        """If signal side is BUY, then bias direction must be bullish."""
        # Use min_confidence=0.0 to remove the confidence filter
        gen = SignalGenerator(stop_buffer=2.0, min_confidence=0.0)
        signal = gen.generate(fvgs_5min, bias, fvgs_htf)

        if signal is not None and signal["side"] == "BUY":
            assert bias.direction == "bullish", (
                f"BUY signal produced but bias.direction is '{bias.direction}', "
                f"expected 'bullish'"
            )

    @given(
        fvgs_5min=lists(fvg_5min_objects(), min_size=0, max_size=10),
        bias=bias_objects(),
        fvgs_htf=lists(htf_fvg_objects(), min_size=0, max_size=10),
    )
    @settings(max_examples=50, deadline=None)
    def test_sell_signal_only_with_bearish_bias(self, fvgs_5min, bias, fvgs_htf):
        """If signal side is SELL, then bias direction must be bearish."""
        gen = SignalGenerator(stop_buffer=2.0, min_confidence=0.0)
        signal = gen.generate(fvgs_5min, bias, fvgs_htf)

        if signal is not None and signal["side"] == "SELL":
            assert bias.direction == "bearish", (
                f"SELL signal produced but bias.direction is '{bias.direction}', "
                f"expected 'bearish'"
            )

    @given(
        fvgs_5min=lists(fvg_5min_objects(), min_size=0, max_size=10),
        fvgs_htf=lists(htf_fvg_objects(), min_size=0, max_size=10),
        confidence=floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=50, deadline=None)
    def test_no_signal_with_neutral_bias(self, fvgs_5min, fvgs_htf, confidence):
        """When bias direction is neutral, no signal must be produced."""
        neutral_bias = Bias(direction="neutral", confidence=confidence)
        gen = SignalGenerator(stop_buffer=2.0, min_confidence=0.0)
        signal = gen.generate(fvgs_5min, neutral_bias, fvgs_htf)

        assert signal is None, (
            f"Signal was produced with neutral bias: {signal}"
        )


# ---------------------------------------------------------------------------
# Strategies for Risk-Reward Sanity tests
# ---------------------------------------------------------------------------

@composite
def unfilled_5min_fvg_objects(draw):
    """Generate random unfilled FVG objects on the 5min timeframe.

    Generates both bullish and bearish FVGs with realistic gold-price zones
    in the 3000-5000 range.
    """
    fvg_type = draw(sampled_from(["bullish", "bearish"]))

    zone_a = draw(floats(min_value=3000.0, max_value=5000.0, allow_nan=False, allow_infinity=False))
    zone_b = draw(floats(min_value=3000.0, max_value=5000.0, allow_nan=False, allow_infinity=False))

    # Ensure the zone has a meaningful non-zero width (at least 0.1 points)
    assume(abs(zone_a - zone_b) >= 0.1)

    zone_upper = max(zone_a, zone_b)
    zone_lower = min(zone_a, zone_b)

    formation_ts = draw(datetimes(
        min_value=datetime(2020, 1, 1),
        max_value=datetime(2030, 12, 31),
    ))

    return FVG(
        type=fvg_type,
        zone_upper=zone_upper,
        zone_lower=zone_lower,
        formation_ts=formation_ts,
        source_tf="5min",
        fill_status="unfilled",
        age_bars=draw(integers(min_value=0, max_value=40)),
    )


@composite
def htf_fvg_objects(draw):
    """Generate random FVG objects from higher timeframes (60min or 15min).

    Zones are in the 3000-5000 range, varied positions to provide TP targets.
    """
    fvg_type = draw(sampled_from(["bullish", "bearish"]))

    zone_a = draw(floats(min_value=3000.0, max_value=5000.0, allow_nan=False, allow_infinity=False))
    zone_b = draw(floats(min_value=3000.0, max_value=5000.0, allow_nan=False, allow_infinity=False))

    assume(abs(zone_a - zone_b) >= 0.1)

    zone_upper = max(zone_a, zone_b)
    zone_lower = min(zone_a, zone_b)

    formation_ts = draw(datetimes(
        min_value=datetime(2020, 1, 1),
        max_value=datetime(2030, 12, 31),
    ))
    source_tf = draw(sampled_from(["60min", "15min"]))
    fill_status = draw(sampled_from(["unfilled", "partial"]))

    return FVG(
        type=fvg_type,
        zone_upper=zone_upper,
        zone_lower=zone_lower,
        formation_ts=formation_ts,
        source_tf=source_tf,
        fill_status=fill_status,
        age_bars=draw(integers(min_value=0, max_value=40)),
    )


@composite
def matching_bias_for_fvg_list(draw, fvg_list):
    """Generate a Bias matching the direction of the majority of FVGs in the list.

    Confidence is drawn above 0.0 to allow signal generation when FVGs align.
    The min_confidence used in tests is 0.0, so any positive confidence will work.
    """
    if not fvg_list:
        # No FVGs — can't generate a matching bias that would produce signals.
        # Return a dummy non-neutral bias with low confidence (will still be filtered).
        direction = draw(sampled_from(["bullish", "bearish"]))
        confidence = draw(floats(min_value=0.01, max_value=1.0, allow_nan=False, allow_infinity=False))
        return Bias(direction=direction, confidence=confidence)

    # Count directions in fvg_list to decide the majority
    bull_count = sum(1 for f in fvg_list if f.type == "bullish")
    bear_count = len(fvg_list) - bull_count

    if bull_count >= bear_count:
        direction = "bullish"
    else:
        direction = "bearish"

    confidence = draw(floats(min_value=0.01, max_value=1.0, allow_nan=False, allow_infinity=False))
    return Bias(direction=direction, confidence=confidence)


# ---------------------------------------------------------------------------
# Property 6: Risk-Reward Sanity
# Validates: Requirements 4.5
# ---------------------------------------------------------------------------

class TestRiskRewardSanity:
    """Property 6: For all generated signals, tp_pts > stop_pts > 0.

    Signals with unfavorable R:R (tp_pts <= stop_pts) are discarded.
    Any signal returned must have both stop_pts > 0 and tp_pts > stop_pts.

    **Validates: Requirements 4.5**
    """

    @given(
        fvgs_5min=lists(unfilled_5min_fvg_objects(), min_size=1, max_size=10),
        fvgs_higher_tf=lists(htf_fvg_objects(), min_size=0, max_size=15),
        stop_buffer=floats(min_value=0.0, max_value=10.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=50, deadline=None)
    def test_signal_tp_greater_than_stop_greater_than_zero(
        self,
        fvgs_5min: list,
        fvgs_higher_tf: list,
        stop_buffer: float,
    ):
        """For any returned signal, tp_pts > stop_pts > 0 must hold.

        Uses min_confidence=0.0 so signals are generated whenever FVG alignment exists.
        The bias direction is chosen to match the FVGs in fvgs_5min.
        """
        # Build a bias aligned with the 5min FVGs to maximise signal generation
        bull_count = sum(1 for f in fvgs_5min if f.type == "bullish")
        bear_count = len(fvgs_5min) - bull_count

        direction = "bullish" if bull_count >= bear_count else "bearish"
        # Confidence above min_confidence=0.0 so the generator proceeds
        bias = Bias(direction=direction, confidence=0.5)

        # Use min_confidence=0.0 so signals are produced whenever alignment exists
        gen = SignalGenerator(stop_buffer=stop_buffer, min_confidence=0.0)
        signal = gen.generate(fvgs_5min, bias, fvgs_higher_tf)

        if signal is not None:
            assert signal["stop_pts"] > 0, (
                f"stop_pts={signal['stop_pts']} must be > 0"
            )
            assert signal["tp_pts"] > signal["stop_pts"], (
                f"tp_pts={signal['tp_pts']} must be > stop_pts={signal['stop_pts']}"
            )

    @given(
        fvgs_5min=lists(unfilled_5min_fvg_objects(), min_size=1, max_size=10),
        fvgs_higher_tf=lists(htf_fvg_objects(), min_size=0, max_size=15),
        stop_buffer=floats(min_value=0.0, max_value=10.0, allow_nan=False, allow_infinity=False),
        confidence=floats(min_value=0.01, max_value=1.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=50, deadline=None)
    def test_signal_rr_sanity_across_confidence_levels(
        self,
        fvgs_5min: list,
        fvgs_higher_tf: list,
        stop_buffer: float,
        confidence: float,
    ):
        """R:R sanity holds for any confidence level, not just 0.0.

        Varying confidence tests that the R:R check is not bypassed by
        confidence filtering — any signal produced (regardless of confidence
        threshold used) must satisfy tp_pts > stop_pts > 0.
        """
        bull_count = sum(1 for f in fvgs_5min if f.type == "bullish")
        bear_count = len(fvgs_5min) - bull_count
        direction = "bullish" if bull_count >= bear_count else "bearish"

        bias = Bias(direction=direction, confidence=confidence)

        # min_confidence=0.0 ensures confidence doesn't block signal; we vary bias.confidence
        gen = SignalGenerator(stop_buffer=stop_buffer, min_confidence=0.0)
        signal = gen.generate(fvgs_5min, bias, fvgs_higher_tf)

        if signal is not None:
            assert signal["stop_pts"] > 0, (
                f"stop_pts={signal['stop_pts']} must be > 0 "
                f"(stop_buffer={stop_buffer}, confidence={confidence})"
            )
            assert signal["tp_pts"] > signal["stop_pts"], (
                f"tp_pts={signal['tp_pts']} must be > stop_pts={signal['stop_pts']} "
                f"(stop_buffer={stop_buffer}, confidence={confidence})"
            )
