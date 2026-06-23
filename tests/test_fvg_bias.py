"""Unit tests for BiasCalculator in strategy/fvg_bias.py.

Tests various scenarios for 60min bias calculation and 15min adjustment.
"""

from datetime import datetime

import pytest

from strategy.fvg_bias import BiasCalculator
from strategy.fvg_detector import Bias, FVG


def _make_fvg(fvg_type: str, fill_status: str = "unfilled", source_tf: str = "60min") -> FVG:
    """Helper to create FVG objects for testing."""
    return FVG(
        type=fvg_type,
        zone_upper=3050.0 if fvg_type == "bullish" else 3100.0,
        zone_lower=3040.0 if fvg_type == "bullish" else 3090.0,
        formation_ts=datetime(2024, 1, 15, 10, 0, 0),
        source_tf=source_tf,
        fill_status=fill_status,
    )


class TestCalculate60minBias:
    """Tests for BiasCalculator.calculate_60min_bias()."""

    def setup_method(self):
        self.calc = BiasCalculator()

    def test_no_fvgs_returns_neutral(self):
        """No FVGs → neutral bias with confidence 0.0 (Req 3.6)."""
        result = self.calc.calculate_60min_bias([])
        assert result.direction == "neutral"
        assert result.confidence == 0.0

    def test_all_filled_fvgs_returns_neutral(self):
        """All filled FVGs treated as no unfilled → neutral (Req 3.6)."""
        fvgs = [
            _make_fvg("bullish", fill_status="filled"),
            _make_fvg("bearish", fill_status="filled"),
        ]
        result = self.calc.calculate_60min_bias(fvgs)
        assert result.direction == "neutral"
        assert result.confidence == 0.0

    def test_more_bullish_returns_bullish(self):
        """More bullish than bearish → bullish bias (Req 3.2)."""
        fvgs = [
            _make_fvg("bullish"),
            _make_fvg("bullish"),
            _make_fvg("bearish"),
        ]
        result = self.calc.calculate_60min_bias(fvgs)
        assert result.direction == "bullish"
        # confidence = abs(2 - 1) / (2 + 1) = 1/3
        assert result.confidence == pytest.approx(1 / 3)

    def test_more_bearish_returns_bearish(self):
        """More bearish than bullish → bearish bias (Req 3.2)."""
        fvgs = [
            _make_fvg("bullish"),
            _make_fvg("bearish"),
            _make_fvg("bearish"),
            _make_fvg("bearish"),
        ]
        result = self.calc.calculate_60min_bias(fvgs)
        assert result.direction == "bearish"
        # confidence = abs(1 - 3) / (1 + 3) = 2/4 = 0.5
        assert result.confidence == pytest.approx(0.5)

    def test_equal_counts_returns_neutral(self):
        """Equal bullish and bearish counts → neutral (Req 3.2)."""
        fvgs = [
            _make_fvg("bullish"),
            _make_fvg("bearish"),
        ]
        result = self.calc.calculate_60min_bias(fvgs)
        assert result.direction == "neutral"
        # confidence = abs(1 - 1) / (1 + 1) = 0.0
        assert result.confidence == 0.0

    def test_partial_fvgs_counted(self):
        """Partial-fill FVGs are counted as active (Req 3.2)."""
        fvgs = [
            _make_fvg("bullish", fill_status="partial"),
            _make_fvg("bullish", fill_status="unfilled"),
            _make_fvg("bearish", fill_status="filled"),  # excluded
        ]
        result = self.calc.calculate_60min_bias(fvgs)
        assert result.direction == "bullish"
        # Only 2 bullish, 0 bearish → confidence = 2/2 = 1.0
        assert result.confidence == pytest.approx(1.0)

    def test_all_bullish_confidence_one(self):
        """All unfilled bullish → confidence 1.0 (Req 3.3)."""
        fvgs = [_make_fvg("bullish") for _ in range(5)]
        result = self.calc.calculate_60min_bias(fvgs)
        assert result.direction == "bullish"
        assert result.confidence == pytest.approx(1.0)

    def test_confidence_formula(self):
        """Verify confidence = abs(bull - bear) / (bull + bear) (Req 3.3)."""
        # 3 bullish, 1 bearish → abs(3-1)/(3+1) = 0.5
        fvgs = [_make_fvg("bullish") for _ in range(3)] + [_make_fvg("bearish")]
        result = self.calc.calculate_60min_bias(fvgs)
        assert result.confidence == pytest.approx(0.5)


class TestAdjustWith15min:
    """Tests for BiasCalculator.adjust_with_15min()."""

    def setup_method(self):
        self.calc = BiasCalculator()

    def test_15min_confirms_bullish_increases_confidence(self):
        """15min majority matches bullish → +0.2 (Req 3.4)."""
        bias = Bias(direction="bullish", confidence=0.5)
        fvgs_15min = [
            _make_fvg("bullish", source_tf="15min"),
            _make_fvg("bullish", source_tf="15min"),
            _make_fvg("bearish", source_tf="15min"),
        ]
        result = self.calc.adjust_with_15min(bias, fvgs_15min)
        assert result.direction == "bullish"
        assert result.confidence == pytest.approx(0.7)

    def test_15min_confirms_bearish_increases_confidence(self):
        """15min majority matches bearish → +0.2 (Req 3.4)."""
        bias = Bias(direction="bearish", confidence=0.6)
        fvgs_15min = [
            _make_fvg("bearish", source_tf="15min"),
            _make_fvg("bearish", source_tf="15min"),
        ]
        result = self.calc.adjust_with_15min(bias, fvgs_15min)
        assert result.direction == "bearish"
        assert result.confidence == pytest.approx(0.8)

    def test_15min_opposes_reduces_confidence(self):
        """15min majority opposes → -0.3 (Req 3.5)."""
        bias = Bias(direction="bullish", confidence=0.6)
        fvgs_15min = [
            _make_fvg("bearish", source_tf="15min"),
            _make_fvg("bearish", source_tf="15min"),
            _make_fvg("bullish", source_tf="15min"),
        ]
        result = self.calc.adjust_with_15min(bias, fvgs_15min)
        assert result.direction == "bullish"  # direction unchanged
        assert result.confidence == pytest.approx(0.3)

    def test_confidence_capped_at_one(self):
        """Confidence cannot exceed 1.0 (Req 3.4)."""
        bias = Bias(direction="bullish", confidence=0.9)
        fvgs_15min = [_make_fvg("bullish", source_tf="15min")]
        result = self.calc.adjust_with_15min(bias, fvgs_15min)
        assert result.confidence == pytest.approx(1.0)

    def test_confidence_floored_at_zero(self):
        """Confidence cannot go below 0.0 (Req 3.5)."""
        bias = Bias(direction="bullish", confidence=0.1)
        fvgs_15min = [_make_fvg("bearish", source_tf="15min")]
        result = self.calc.adjust_with_15min(bias, fvgs_15min)
        assert result.confidence == pytest.approx(0.0)

    def test_no_15min_fvgs_no_adjustment(self):
        """No 15min FVGs → no change to bias."""
        bias = Bias(direction="bullish", confidence=0.5)
        result = self.calc.adjust_with_15min(bias, [])
        assert result.direction == "bullish"
        assert result.confidence == pytest.approx(0.5)

    def test_all_15min_filled_no_adjustment(self):
        """All filled 15min FVGs → no adjustment."""
        bias = Bias(direction="bearish", confidence=0.7)
        fvgs_15min = [
            _make_fvg("bullish", fill_status="filled", source_tf="15min"),
            _make_fvg("bearish", fill_status="filled", source_tf="15min"),
        ]
        result = self.calc.adjust_with_15min(bias, fvgs_15min)
        assert result.direction == "bearish"
        assert result.confidence == pytest.approx(0.7)

    def test_equal_15min_counts_no_adjustment(self):
        """Equal 15min bullish/bearish counts → no adjustment."""
        bias = Bias(direction="bullish", confidence=0.6)
        fvgs_15min = [
            _make_fvg("bullish", source_tf="15min"),
            _make_fvg("bearish", source_tf="15min"),
        ]
        result = self.calc.adjust_with_15min(bias, fvgs_15min)
        assert result.direction == "bullish"
        assert result.confidence == pytest.approx(0.6)

    def test_neutral_bias_direction_unchanged(self):
        """Neutral bias direction stays neutral even with 15min adjustment."""
        bias = Bias(direction="neutral", confidence=0.0)
        fvgs_15min = [
            _make_fvg("bullish", source_tf="15min"),
            _make_fvg("bullish", source_tf="15min"),
        ]
        result = self.calc.adjust_with_15min(bias, fvgs_15min)
        # Direction remains neutral, but 15min doesn't "match" neutral
        # so this is an oppose scenario (bullish != neutral) → -0.3
        assert result.direction == "neutral"
        # 0.0 - 0.3 floored at 0.0
        assert result.confidence == pytest.approx(0.0)

    def test_direction_never_changes(self):
        """15min adjustment never changes the bias direction (Req 3.5)."""
        bias = Bias(direction="bearish", confidence=0.8)
        fvgs_15min = [
            _make_fvg("bullish", source_tf="15min"),
            _make_fvg("bullish", source_tf="15min"),
            _make_fvg("bullish", source_tf="15min"),
        ]
        result = self.calc.adjust_with_15min(bias, fvgs_15min)
        assert result.direction == "bearish"  # unchanged
        assert result.confidence == pytest.approx(0.5)  # 0.8 - 0.3
