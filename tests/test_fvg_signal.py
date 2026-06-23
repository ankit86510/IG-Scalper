"""Unit tests for SignalGenerator in strategy/fvg_signal.py."""

from datetime import datetime, timedelta

from strategy.fvg_detector import Bias, FVG
from strategy.fvg_signal import SignalGenerator


def _make_fvg(
    type_="bullish",
    zone_upper=110.0,
    zone_lower=100.0,
    source_tf="5min",
    fill_status="unfilled",
    ts_offset_min=0,
):
    """Helper to create an FVG with sensible defaults."""
    return FVG(
        type=type_,
        zone_upper=zone_upper,
        zone_lower=zone_lower,
        formation_ts=datetime(2024, 6, 15, 10, 0) + timedelta(minutes=ts_offset_min),
        source_tf=source_tf,
        fill_status=fill_status,
    )


class TestBuySignalGeneration:
    """Tests for BUY signal generation (Req 4.1)."""

    def test_basic_buy_signal_with_htf_target(self):
        """BUY signal from bullish 5min FVG + bullish bias + HTF target."""
        sg = SignalGenerator(stop_buffer=2.0, min_confidence=0.6)
        # 5min bullish FVG: zone 3040-3050, entry at 3050
        fvg_5min = _make_fvg(type_="bullish", zone_upper=3050.0, zone_lower=3040.0)
        bias = Bias(direction="bullish", confidence=0.8)
        # HTF bearish FVG with zone_lower at 3065 (15 pts above entry)
        htf = _make_fvg(
            type_="bearish", zone_upper=3070.0, zone_lower=3065.0, source_tf="60min"
        )

        signal = sg.generate([fvg_5min], bias, [htf])

        assert signal is not None
        assert signal["side"] == "BUY"
        # stop_pts = (3050 - 3040) + 2 = 12
        assert signal["stop_pts"] == 12.0
        # tp_pts = 3065 - 3050 = 15
        assert signal["tp_pts"] == 15.0

    def test_buy_entry_at_zone_upper(self):
        """Entry for BUY is at zone_upper (candle 3 low). Verify stop calc."""
        sg = SignalGenerator(stop_buffer=3.0, min_confidence=0.5)
        fvg_5min = _make_fvg(type_="bullish", zone_upper=3100.0, zone_lower=3090.0)
        bias = Bias(direction="bullish", confidence=0.7)
        htf = _make_fvg(
            type_="bearish", zone_upper=3120.0, zone_lower=3115.0, source_tf="60min"
        )

        signal = sg.generate([fvg_5min], bias, [htf])
        assert signal is not None
        # Entry = 3100, stop at 3090 - 3 = 3087
        # stop_pts = (3100 - 3090) + 3 = 13
        assert signal["stop_pts"] == 13.0
        # tp_pts = 3115 - 3100 = 15
        assert signal["tp_pts"] == 15.0

    def test_buy_no_htf_targets_uses_zone_size(self):
        """Without HTF targets, TP defaults to zone_size (may be discarded)."""
        sg = SignalGenerator(stop_buffer=2.0, min_confidence=0.5)
        fvg_5min = _make_fvg(type_="bullish", zone_upper=3050.0, zone_lower=3040.0)
        bias = Bias(direction="bullish", confidence=0.7)

        signal = sg.generate([fvg_5min], bias, [])
        # stop_pts = 12, tp_pts = 10 → 10 <= 12 → discard
        assert signal is None

    def test_buy_no_htf_zero_buffer_zone_size_tp(self):
        """With buffer=0, tp=zone_size=stop → discard (TP must be > SL)."""
        sg = SignalGenerator(stop_buffer=0.0, min_confidence=0.5)
        fvg_5min = _make_fvg(type_="bullish", zone_upper=3050.0, zone_lower=3040.0)
        bias = Bias(direction="bullish", confidence=0.7)

        signal = sg.generate([fvg_5min], bias, [])
        # stop_pts = 10, tp_pts = 10 → 10 <= 10 → discard
        assert signal is None

    def test_buy_selects_nearest_htf_target(self):
        """When multiple HTF targets exist, picks the nearest one."""
        sg = SignalGenerator(stop_buffer=2.0, min_confidence=0.5)
        fvg_5min = _make_fvg(type_="bullish", zone_upper=3050.0, zone_lower=3040.0)
        bias = Bias(direction="bullish", confidence=0.7)
        htf_near = _make_fvg(
            type_="bearish", zone_upper=3070.0, zone_lower=3063.0, source_tf="60min"
        )
        htf_far = _make_fvg(
            type_="bearish", zone_upper=3100.0, zone_lower=3090.0, source_tf="15min"
        )

        signal = sg.generate([fvg_5min], bias, [htf_near, htf_far])
        # Nearest = 3063, distance = 3063 - 3050 = 13
        assert signal["tp_pts"] == 13.0

    def test_buy_selects_most_recent_fvg(self):
        """Most recent unfilled FVG is used when multiple exist."""
        sg = SignalGenerator(stop_buffer=2.0, min_confidence=0.5)
        older = _make_fvg(
            type_="bullish", zone_upper=3020.0, zone_lower=3010.0, ts_offset_min=0
        )
        newer = _make_fvg(
            type_="bullish", zone_upper=3050.0, zone_lower=3040.0, ts_offset_min=30
        )
        bias = Bias(direction="bullish", confidence=0.7)
        htf = _make_fvg(
            type_="bearish", zone_upper=3070.0, zone_lower=3065.0, source_tf="60min"
        )

        signal = sg.generate([older, newer], bias, [htf])
        assert signal is not None
        assert signal["meta"]["trigger_fvg"]["zone_upper"] == 3050.0

    def test_buy_ignores_filled_htf(self):
        """Filled HTF FVGs are not considered as TP targets."""
        sg = SignalGenerator(stop_buffer=2.0, min_confidence=0.5)
        fvg_5min = _make_fvg(type_="bullish", zone_upper=3050.0, zone_lower=3040.0)
        bias = Bias(direction="bullish", confidence=0.7)
        # This HTF is filled → should be ignored
        htf_filled = _make_fvg(
            type_="bearish", zone_upper=3070.0, zone_lower=3063.0,
            source_tf="60min", fill_status="filled"
        )

        signal = sg.generate([fvg_5min], bias, [htf_filled])
        # No valid HTF targets → tp = zone_size = 10 <= stop = 12 → discard
        assert signal is None


class TestSellSignalGeneration:
    """Tests for SELL signal generation (Req 4.2)."""

    def test_basic_sell_signal_with_htf_target(self):
        """SELL signal from bearish 5min FVG + bearish bias + HTF target."""
        sg = SignalGenerator(stop_buffer=2.0, min_confidence=0.6)
        # 5min bearish FVG: zone 3050-3060, entry at 3050
        fvg_5min = _make_fvg(type_="bearish", zone_upper=3060.0, zone_lower=3050.0)
        bias = Bias(direction="bearish", confidence=0.8)
        # HTF bullish FVG with zone_upper at 3035 (15 pts below entry)
        htf = _make_fvg(
            type_="bullish", zone_upper=3035.0, zone_lower=3025.0, source_tf="60min"
        )

        signal = sg.generate([fvg_5min], bias, [htf])

        assert signal is not None
        assert signal["side"] == "SELL"
        # stop_pts = (3060 - 3050) + 2 = 12
        assert signal["stop_pts"] == 12.0
        # tp_pts = 3050 - 3035 = 15
        assert signal["tp_pts"] == 15.0

    def test_sell_entry_at_zone_lower(self):
        """Entry for SELL is at zone_lower (candle 3 high). Verify stop calc."""
        sg = SignalGenerator(stop_buffer=3.0, min_confidence=0.5)
        fvg_5min = _make_fvg(type_="bearish", zone_upper=3100.0, zone_lower=3085.0)
        bias = Bias(direction="bearish", confidence=0.7)
        htf = _make_fvg(
            type_="bullish", zone_upper=3065.0, zone_lower=3055.0, source_tf="60min"
        )

        signal = sg.generate([fvg_5min], bias, [htf])
        assert signal is not None
        # Entry = 3085, stop at 3100 + 3 = 3103
        # stop_pts = (3100 - 3085) + 3 = 18
        assert signal["stop_pts"] == 18.0
        # tp_pts = 3085 - 3065 = 20
        assert signal["tp_pts"] == 20.0

    def test_sell_no_htf_targets_discards(self):
        """Without HTF targets, TP = zone_size which is < stop → discard."""
        sg = SignalGenerator(stop_buffer=2.0, min_confidence=0.5)
        fvg_5min = _make_fvg(type_="bearish", zone_upper=3060.0, zone_lower=3050.0)
        bias = Bias(direction="bearish", confidence=0.7)

        signal = sg.generate([fvg_5min], bias, [])
        # stop_pts = 12, tp_pts = 10 → discard
        assert signal is None

    def test_sell_selects_nearest_htf_target(self):
        """When multiple HTF targets below, picks the nearest (highest)."""
        sg = SignalGenerator(stop_buffer=2.0, min_confidence=0.5)
        fvg_5min = _make_fvg(type_="bearish", zone_upper=3060.0, zone_lower=3050.0)
        bias = Bias(direction="bearish", confidence=0.7)
        htf_near = _make_fvg(
            type_="bullish", zone_upper=3037.0, zone_lower=3030.0, source_tf="60min"
        )
        htf_far = _make_fvg(
            type_="bullish", zone_upper=3010.0, zone_lower=3000.0, source_tf="15min"
        )

        signal = sg.generate([fvg_5min], bias, [htf_near, htf_far])
        # Nearest below = 3037, distance = 3050 - 3037 = 13
        assert signal["tp_pts"] == 13.0

    def test_sell_selects_most_recent_fvg(self):
        """Most recent unfilled bearish FVG is used."""
        sg = SignalGenerator(stop_buffer=2.0, min_confidence=0.5)
        older = _make_fvg(
            type_="bearish", zone_upper=3080.0, zone_lower=3070.0, ts_offset_min=0
        )
        newer = _make_fvg(
            type_="bearish", zone_upper=3060.0, zone_lower=3050.0, ts_offset_min=30
        )
        bias = Bias(direction="bearish", confidence=0.7)
        htf = _make_fvg(
            type_="bullish", zone_upper=3035.0, zone_lower=3025.0, source_tf="60min"
        )

        signal = sg.generate([older, newer], bias, [htf])
        assert signal is not None
        assert signal["meta"]["trigger_fvg"]["zone_upper"] == 3060.0


class TestSignalDiscardConditions:
    """Tests for signal discard logic (Req 4.5, 4.7, 4.8)."""

    def test_discard_when_tp_less_than_sl(self):
        """Signal discarded if TP distance <= SL distance (Req 4.5)."""
        sg = SignalGenerator(stop_buffer=2.0, min_confidence=0.5)
        fvg_5min = _make_fvg(type_="bullish", zone_upper=3050.0, zone_lower=3040.0)
        bias = Bias(direction="bullish", confidence=0.7)
        # HTF at 3060 → tp = 10, stop = 12 → 10 <= 12 → discard
        htf = _make_fvg(
            type_="bearish", zone_upper=3065.0, zone_lower=3060.0, source_tf="60min"
        )

        signal = sg.generate([fvg_5min], bias, [htf])
        assert signal is None

    def test_discard_when_tp_equals_sl(self):
        """Signal discarded if TP distance == SL distance."""
        sg = SignalGenerator(stop_buffer=2.0, min_confidence=0.5)
        fvg_5min = _make_fvg(type_="bullish", zone_upper=3050.0, zone_lower=3040.0)
        bias = Bias(direction="bullish", confidence=0.7)
        # HTF at 3062 → tp = 12, stop = 12 → 12 <= 12 → discard
        htf = _make_fvg(
            type_="bearish", zone_upper=3070.0, zone_lower=3062.0, source_tf="60min"
        )

        signal = sg.generate([fvg_5min], bias, [htf])
        assert signal is None

    def test_discard_neutral_bias(self):
        """No signal when bias is neutral (Req 4.7)."""
        sg = SignalGenerator(stop_buffer=2.0, min_confidence=0.5)
        fvg_5min = _make_fvg(type_="bullish", zone_upper=3050.0, zone_lower=3040.0)
        bias = Bias(direction="neutral", confidence=0.0)
        htf = _make_fvg(
            type_="bearish", zone_upper=3080.0, zone_lower=3070.0, source_tf="60min"
        )

        signal = sg.generate([fvg_5min], bias, [htf])
        assert signal is None

    def test_discard_low_confidence(self):
        """No signal when confidence < min_confidence (Req 4.8)."""
        sg = SignalGenerator(stop_buffer=2.0, min_confidence=0.6)
        fvg_5min = _make_fvg(type_="bullish", zone_upper=3050.0, zone_lower=3040.0)
        bias = Bias(direction="bullish", confidence=0.5)  # below 0.6
        htf = _make_fvg(
            type_="bearish", zone_upper=3080.0, zone_lower=3070.0, source_tf="60min"
        )

        signal = sg.generate([fvg_5min], bias, [htf])
        assert signal is None

    def test_confidence_at_threshold_passes(self):
        """Confidence exactly at min_confidence passes the check (Req 4.8)."""
        sg = SignalGenerator(stop_buffer=2.0, min_confidence=0.6)
        fvg_5min = _make_fvg(type_="bullish", zone_upper=3050.0, zone_lower=3040.0)
        bias = Bias(direction="bullish", confidence=0.6)
        # HTF providing TP > SL
        htf = _make_fvg(
            type_="bearish", zone_upper=3070.0, zone_lower=3065.0, source_tf="60min"
        )

        signal = sg.generate([fvg_5min], bias, [htf])
        assert signal is not None
        assert signal["side"] == "BUY"

    def test_discard_fvg_direction_contradicts_bias(self):
        """No signal when 5min FVG direction contradicts bias (Req 4.7)."""
        sg = SignalGenerator(stop_buffer=2.0, min_confidence=0.5)
        # Bearish FVG but bullish bias
        fvg_5min = _make_fvg(type_="bearish", zone_upper=3060.0, zone_lower=3050.0)
        bias = Bias(direction="bullish", confidence=0.8)
        htf = _make_fvg(
            type_="bearish", zone_upper=3080.0, zone_lower=3070.0, source_tf="60min"
        )

        signal = sg.generate([fvg_5min], bias, [htf])
        assert signal is None

    def test_discard_bullish_fvg_with_bearish_bias(self):
        """No signal when bullish FVG but bearish bias."""
        sg = SignalGenerator(stop_buffer=2.0, min_confidence=0.5)
        fvg_5min = _make_fvg(type_="bullish", zone_upper=3050.0, zone_lower=3040.0)
        bias = Bias(direction="bearish", confidence=0.8)
        htf = _make_fvg(
            type_="bullish", zone_upper=3020.0, zone_lower=3010.0, source_tf="60min"
        )

        signal = sg.generate([fvg_5min], bias, [htf])
        assert signal is None


class TestSignalFormat:
    """Tests for signal output format (Req 4.6, 4.9)."""

    def test_signal_has_required_keys(self):
        """Signal dict contains side, stop_pts, tp_pts, meta (Req 4.6)."""
        sg = SignalGenerator(stop_buffer=2.0, min_confidence=0.5)
        fvg_5min = _make_fvg(type_="bullish", zone_upper=3050.0, zone_lower=3040.0)
        bias = Bias(direction="bullish", confidence=0.7)
        htf = _make_fvg(
            type_="bearish", zone_upper=3070.0, zone_lower=3065.0, source_tf="60min"
        )

        signal = sg.generate([fvg_5min], bias, [htf])
        assert signal is not None
        assert "side" in signal
        assert "stop_pts" in signal
        assert "tp_pts" in signal
        assert "meta" in signal

    def test_stop_pts_and_tp_pts_are_positive_floats(self):
        """stop_pts and tp_pts are positive float values."""
        sg = SignalGenerator(stop_buffer=2.0, min_confidence=0.5)
        fvg_5min = _make_fvg(type_="bullish", zone_upper=3050.0, zone_lower=3040.0)
        bias = Bias(direction="bullish", confidence=0.7)
        htf = _make_fvg(
            type_="bearish", zone_upper=3070.0, zone_lower=3065.0, source_tf="60min"
        )

        signal = sg.generate([fvg_5min], bias, [htf])
        assert signal is not None
        assert isinstance(signal["stop_pts"], float)
        assert isinstance(signal["tp_pts"], float)
        assert signal["stop_pts"] > 0
        assert signal["tp_pts"] > 0

    def test_meta_contains_required_fields(self):
        """Meta includes bias, trigger FVG, source FVGs (Req 4.9)."""
        sg = SignalGenerator(stop_buffer=2.0, min_confidence=0.5)
        fvg_5min = _make_fvg(type_="bullish", zone_upper=3050.0, zone_lower=3040.0)
        bias = Bias(direction="bullish", confidence=0.7)
        htf_60 = _make_fvg(
            type_="bearish", zone_upper=3070.0, zone_lower=3065.0, source_tf="60min"
        )
        htf_15 = _make_fvg(
            type_="bearish", zone_upper=3080.0, zone_lower=3075.0, source_tf="15min"
        )

        signal = sg.generate([fvg_5min], bias, [htf_60, htf_15])
        meta = signal["meta"]

        assert meta["bias_direction"] == "bullish"
        assert meta["bias_confidence"] == 0.7
        assert meta["trigger_fvg"]["type"] == "bullish"
        assert meta["trigger_fvg"]["zone_upper"] == 3050.0
        assert meta["trigger_fvg"]["zone_lower"] == 3040.0
        assert meta["trigger_fvg"]["source_tf"] == "5min"
        assert "fvgs_60min" in meta
        assert "fvgs_15min" in meta
        assert "fvgs_5min" in meta
        assert "entry_zone" in meta
        assert meta["entry_zone"] == (3040.0, 3050.0)

    def test_meta_separates_htf_by_timeframe(self):
        """Meta correctly separates HTF FVGs by their timeframe."""
        sg = SignalGenerator(stop_buffer=2.0, min_confidence=0.5)
        fvg_5min = _make_fvg(type_="bullish", zone_upper=3050.0, zone_lower=3040.0)
        bias = Bias(direction="bullish", confidence=0.7)
        htf_60 = _make_fvg(
            type_="bearish", zone_upper=3070.0, zone_lower=3065.0, source_tf="60min"
        )
        htf_15 = _make_fvg(
            type_="bearish", zone_upper=3080.0, zone_lower=3075.0, source_tf="15min"
        )

        signal = sg.generate([fvg_5min], bias, [htf_60, htf_15])
        meta = signal["meta"]

        assert len(meta["fvgs_60min"]) == 1
        assert len(meta["fvgs_15min"]) == 1
        assert meta["fvgs_60min"][0]["source_tf"] == "60min"
        assert meta["fvgs_15min"][0]["source_tf"] == "15min"


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_empty_5min_fvg_list(self):
        """No 5min FVGs → None."""
        sg = SignalGenerator(stop_buffer=2.0, min_confidence=0.5)
        bias = Bias(direction="bullish", confidence=0.8)
        htf = _make_fvg(
            type_="bearish", zone_upper=3070.0, zone_lower=3065.0, source_tf="60min"
        )

        signal = sg.generate([], bias, [htf])
        assert signal is None

    def test_only_filled_5min_fvgs(self):
        """Only filled 5min FVGs → None (no unfilled matching FVGs)."""
        sg = SignalGenerator(stop_buffer=2.0, min_confidence=0.5)
        fvg_filled = _make_fvg(
            type_="bullish", zone_upper=3050.0, zone_lower=3040.0, fill_status="filled"
        )
        bias = Bias(direction="bullish", confidence=0.8)
        htf = _make_fvg(
            type_="bearish", zone_upper=3070.0, zone_lower=3065.0, source_tf="60min"
        )

        signal = sg.generate([fvg_filled], bias, [htf])
        assert signal is None

    def test_partial_fill_fvg_still_valid(self):
        """Partially filled 5min FVGs are still valid for signal generation."""
        sg = SignalGenerator(stop_buffer=2.0, min_confidence=0.5)
        fvg_partial = _make_fvg(
            type_="bullish", zone_upper=3050.0, zone_lower=3045.0, fill_status="partial"
        )
        bias = Bias(direction="bullish", confidence=0.8)
        htf = _make_fvg(
            type_="bearish", zone_upper=3070.0, zone_lower=3060.0, source_tf="60min"
        )

        signal = sg.generate([fvg_partial], bias, [htf])
        assert signal is not None
        assert signal["side"] == "BUY"
        # stop_pts = (3050 - 3045) + 2 = 7
        assert signal["stop_pts"] == 7.0
        # tp_pts = 3060 - 3050 = 10
        assert signal["tp_pts"] == 10.0

    def test_empty_htf_list(self):
        """Empty HTF list falls back to zone_size TP (likely discarded)."""
        sg = SignalGenerator(stop_buffer=2.0, min_confidence=0.5)
        fvg_5min = _make_fvg(type_="bullish", zone_upper=3050.0, zone_lower=3040.0)
        bias = Bias(direction="bullish", confidence=0.7)

        signal = sg.generate([fvg_5min], bias, [])
        # tp = zone_size = 10, stop = 12 → discard
        assert signal is None

    def test_htf_target_below_entry_ignored_for_buy(self):
        """HTF targets below entry are ignored for BUY signals."""
        sg = SignalGenerator(stop_buffer=2.0, min_confidence=0.5)
        fvg_5min = _make_fvg(type_="bullish", zone_upper=3050.0, zone_lower=3040.0)
        bias = Bias(direction="bullish", confidence=0.7)
        # HTF below entry — not valid as resistance for a BUY TP
        htf_below = _make_fvg(
            type_="bullish", zone_upper=3030.0, zone_lower=3020.0, source_tf="60min"
        )

        signal = sg.generate([fvg_5min], bias, [htf_below])
        # No valid HTF above → tp = zone_size = 10, stop = 12 → discard
        assert signal is None

    def test_htf_target_above_entry_ignored_for_sell(self):
        """HTF targets above entry are ignored for SELL signals."""
        sg = SignalGenerator(stop_buffer=2.0, min_confidence=0.5)
        fvg_5min = _make_fvg(type_="bearish", zone_upper=3060.0, zone_lower=3050.0)
        bias = Bias(direction="bearish", confidence=0.7)
        # HTF above entry — not valid as support for a SELL TP
        htf_above = _make_fvg(
            type_="bearish", zone_upper=3080.0, zone_lower=3070.0, source_tf="60min"
        )

        signal = sg.generate([fvg_5min], bias, [htf_above])
        # No valid HTF below → tp = zone_size = 10, stop = 12 → discard
        assert signal is None

    def test_multiple_5min_types_only_matching_used(self):
        """Only FVGs matching bias direction are considered."""
        sg = SignalGenerator(stop_buffer=2.0, min_confidence=0.5)
        bullish_fvg = _make_fvg(
            type_="bullish", zone_upper=3050.0, zone_lower=3040.0, ts_offset_min=10
        )
        bearish_fvg = _make_fvg(
            type_="bearish", zone_upper=3060.0, zone_lower=3050.0, ts_offset_min=20
        )
        bias = Bias(direction="bullish", confidence=0.7)
        htf = _make_fvg(
            type_="bearish", zone_upper=3070.0, zone_lower=3065.0, source_tf="60min"
        )

        signal = sg.generate([bullish_fvg, bearish_fvg], bias, [htf])
        assert signal is not None
        assert signal["side"] == "BUY"
        assert signal["meta"]["trigger_fvg"]["type"] == "bullish"
