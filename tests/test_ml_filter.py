"""Property-based tests for ML Directional Filter using Hypothesis.

Validates correctness properties defined in the design document.
# Feature: ml-trading-improvements

Tests Properties 1-7 covering feature extraction, label generation,
z-score normalization, insufficient data handling, prediction bounds,
signal confirmation logic, and disabled filter pass-through.
"""

import numpy as np
import pandas as pd
from hypothesis import assume, given, settings, HealthCheck
from hypothesis.strategies import (
    booleans,
    composite,
    floats,
    integers,
    sampled_from,
)

from strategy.ml_filter import MLDirectionalFilter


# ---------------------------------------------------------------------------
# Strategies (generators)
# ---------------------------------------------------------------------------


@composite
def ohlc_dataframes(draw, min_rows=50, max_rows=250):
    """Generate random valid OHLC DataFrames with realistic gold prices.

    Produces valid OHLC bars where high >= max(open, close) and
    low <= min(open, close). Uses a random walk to create realistic sequences.
    Ensures high > low on each bar for valid ATR computation.
    """
    n_rows = draw(integers(min_value=min_rows, max_value=max_rows))

    # Start with a base price and random walk
    base_price = draw(floats(min_value=3000.0, max_value=4000.0, allow_nan=False, allow_infinity=False))

    rows = []
    current_price = base_price

    for _ in range(n_rows):
        # Random walk with bounded drift
        change_pct = draw(floats(min_value=-0.005, max_value=0.005, allow_nan=False, allow_infinity=False))
        current_price = current_price * (1.0 + change_pct)

        # Clamp to reasonable gold price range
        current_price = max(2500.0, min(5500.0, current_price))

        open_price = current_price
        close_price = current_price * (1.0 + draw(floats(
            min_value=-0.003, max_value=0.003, allow_nan=False, allow_infinity=False
        )))
        close_price = max(2500.0, min(5500.0, close_price))

        # High >= max(open, close), Low <= min(open, close)
        # Ensure minimum spread for ATR validity (real markets always have some spread)
        high_extra = draw(floats(min_value=0.0001, max_value=0.002, allow_nan=False, allow_infinity=False))
        low_extra = draw(floats(min_value=0.0001, max_value=0.002, allow_nan=False, allow_infinity=False))

        high_price = max(open_price, close_price) * (1.0 + high_extra)
        low_price = min(open_price, close_price) * (1.0 - low_extra)

        # Ensure low > 0
        low_price = max(1.0, low_price)

        rows.append((open_price, high_price, low_price, close_price))

        current_price = close_price

    index = pd.date_range(start="2024-01-01", periods=n_rows, freq="5min")
    df = pd.DataFrame(rows, columns=["open", "high", "low", "close"], index=index)
    return df


@composite
def small_ohlc_dataframes(draw, min_rows=5, max_rows=99):
    """Generate valid OHLC DataFrames with fewer than 100 rows.

    Used for testing the insufficient data behavior (Property 4).
    Uses a random walk similar to ohlc_dataframes but with fewer bars.
    """
    n_rows = draw(integers(min_value=min_rows, max_value=max_rows))

    base_price = draw(floats(min_value=3000.0, max_value=4000.0, allow_nan=False, allow_infinity=False))

    rows = []
    current_price = base_price

    for _ in range(n_rows):
        change_pct = draw(floats(min_value=-0.005, max_value=0.005, allow_nan=False, allow_infinity=False))
        current_price = current_price * (1.0 + change_pct)
        current_price = max(2500.0, min(5500.0, current_price))

        open_price = current_price
        close_price = current_price * (1.0 + draw(floats(
            min_value=-0.003, max_value=0.003, allow_nan=False, allow_infinity=False
        )))
        close_price = max(2500.0, min(5500.0, close_price))

        high_extra = draw(floats(min_value=0.0001, max_value=0.002, allow_nan=False, allow_infinity=False))
        low_extra = draw(floats(min_value=0.0001, max_value=0.002, allow_nan=False, allow_infinity=False))

        high_price = max(open_price, close_price) * (1.0 + high_extra)
        low_price = min(open_price, close_price) * (1.0 - low_extra)
        low_price = max(1.0, low_price)

        rows.append((open_price, high_price, low_price, close_price))
        current_price = close_price

    index = pd.date_range(start="2024-01-01", periods=n_rows, freq="5min")
    df = pd.DataFrame(rows, columns=["open", "high", "low", "close"], index=index)
    return df


@composite
def feature_matrices(draw, min_rows=2, max_rows=50, n_cols=6):
    """Generate feature matrices with at least one column having non-zero variance.

    Used for testing z-score normalization (Property 3).
    """
    n_rows = draw(integers(min_value=min_rows, max_value=max_rows))

    # Generate a matrix with variation in each column
    matrix = np.zeros((n_rows, n_cols))
    for col in range(n_cols):
        for row in range(n_rows):
            matrix[row, col] = draw(floats(
                min_value=-100.0, max_value=100.0,
                allow_nan=False, allow_infinity=False,
            ))

    # Ensure at least one column has non-zero variance
    col_stds = matrix.std(axis=0)
    if np.all(col_stds == 0):
        # Force first column to have variance
        matrix[0, 0] = 1.0
        matrix[1, 0] = -1.0

    return matrix


@composite
def training_dataframes(draw):
    """Generate OHLC DataFrames large enough for training (>= 100 valid samples).

    Feature extraction drops ~50 warmup rows (SMA(50) + lagged returns).
    Training also drops the last row. So we need min_rows >= 160 to
    reliably get >= 100 valid training samples after warmup removal.
    """
    return draw(ohlc_dataframes(min_rows=170, max_rows=250))


@composite
def signal_dicts(draw):
    """Generate random valid signal dictionaries with BUY or SELL side."""
    side = draw(sampled_from(["BUY", "SELL"]))
    stop_pts = draw(floats(min_value=1.0, max_value=50.0, allow_nan=False, allow_infinity=False))
    tp_pts = draw(floats(min_value=2.0, max_value=100.0, allow_nan=False, allow_infinity=False))

    return {
        "side": side,
        "stop_pts": stop_pts,
        "tp_pts": tp_pts,
        "meta": {"source": "test"},
    }


# ---------------------------------------------------------------------------
# Helper: Train a filter and return it ready for prediction
# ---------------------------------------------------------------------------


def _train_filter(df: pd.DataFrame, threshold: float = 0.55) -> MLDirectionalFilter:
    """Create and train an ML filter on the given DataFrame."""
    config = {
        "enabled": True,
        "probability_threshold": threshold,
        "rolling_window_bars": 500,
        "retrain_interval_hours": 168,
        "model_path": "/tmp/test_ml_model.joblib",
        "model_type": "logistic_regression",
    }
    ml_filter = MLDirectionalFilter(config)
    success = ml_filter.train(df)
    assume(success)  # Skip examples where training fails
    assume(ml_filter.is_enabled)
    return ml_filter


# ---------------------------------------------------------------------------
# Property 1: Feature extraction produces correct dimensions and valid ranges
# Feature: ml-trading-improvements, Property 1: Feature extraction dimensions and ranges
# Validates: Requirements 1.2
# ---------------------------------------------------------------------------


class TestFeatureExtractionDimensionsAndRanges:
    """Property 1: For any valid OHLC DataFrame with at least 50 rows,
    extract_features() SHALL return a matrix with exactly 6 columns where:
    RSI values are in [0, 100], ATR_Ratio values are > 0,
    SMA_Ratio values are > 0, and return values are finite floats.

    **Validates: Requirements 1.2**
    """

    @given(df=ohlc_dataframes(min_rows=55, max_rows=250))
    @settings(max_examples=100, deadline=None)
    def test_feature_matrix_has_6_columns(self, df: pd.DataFrame):
        """# Feature: ml-trading-improvements, Property 1: Feature extraction dimensions and ranges

        extract_features() must return a matrix with exactly 6 columns.
        """
        ml_filter = MLDirectionalFilter({"enabled": True})
        features = ml_filter.extract_features(df)

        assert features.ndim == 2, f"Features must be 2D, got {features.ndim}D"
        assert features.shape[1] == 6, (
            f"Features must have 6 columns, got {features.shape[1]}"
        )

    @given(df=ohlc_dataframes(min_rows=55, max_rows=250))
    @settings(max_examples=100, deadline=None)
    def test_rsi_values_in_valid_range(self, df: pd.DataFrame):
        """# Feature: ml-trading-improvements, Property 1: Feature extraction dimensions and ranges

        RSI column (index 0) values must be in [0, 100].
        """
        ml_filter = MLDirectionalFilter({"enabled": True})
        features = ml_filter.extract_features(df)

        if features.shape[0] == 0:
            return  # No valid rows after NaN removal

        rsi_values = features[:, 0]
        assert np.all(rsi_values >= 0), (
            f"RSI min={rsi_values.min()}, expected >= 0"
        )
        assert np.all(rsi_values <= 100), (
            f"RSI max={rsi_values.max()}, expected <= 100"
        )

    @given(df=ohlc_dataframes(min_rows=55, max_rows=250))
    @settings(max_examples=100, deadline=None)
    def test_atr_ratio_values_positive(self, df: pd.DataFrame):
        """# Feature: ml-trading-improvements, Property 1: Feature extraction dimensions and ranges

        ATR_Ratio column (index 1) values must be > 0.
        """
        ml_filter = MLDirectionalFilter({"enabled": True})
        features = ml_filter.extract_features(df)

        if features.shape[0] == 0:
            return

        atr_ratio_values = features[:, 1]
        assert np.all(atr_ratio_values > 0), (
            f"ATR_Ratio min={atr_ratio_values.min()}, expected > 0"
        )

    @given(df=ohlc_dataframes(min_rows=55, max_rows=250))
    @settings(max_examples=100, deadline=None)
    def test_sma_ratio_values_positive(self, df: pd.DataFrame):
        """# Feature: ml-trading-improvements, Property 1: Feature extraction dimensions and ranges

        SMA_Ratio column (index 2) values must be > 0.
        """
        ml_filter = MLDirectionalFilter({"enabled": True})
        features = ml_filter.extract_features(df)

        if features.shape[0] == 0:
            return

        sma_ratio_values = features[:, 2]
        assert np.all(sma_ratio_values > 0), (
            f"SMA_Ratio min={sma_ratio_values.min()}, expected > 0"
        )

    @given(df=ohlc_dataframes(min_rows=55, max_rows=250))
    @settings(max_examples=100, deadline=None)
    def test_return_values_are_finite(self, df: pd.DataFrame):
        """# Feature: ml-trading-improvements, Property 1: Feature extraction dimensions and ranges

        Return columns (indices 3, 4, 5) must all be finite floats.
        """
        ml_filter = MLDirectionalFilter({"enabled": True})
        features = ml_filter.extract_features(df)

        if features.shape[0] == 0:
            return

        # ret_1 (col 3), ret_3 (col 4), ret_5 (col 5)
        for col_idx in [3, 4, 5]:
            col_values = features[:, col_idx]
            assert np.all(np.isfinite(col_values)), (
                f"Return column {col_idx} has non-finite values"
            )


# ---------------------------------------------------------------------------
# Property 2: Label generation correctness
# Feature: ml-trading-improvements, Property 2: Label generation correctness
# Validates: Requirements 1.3
# ---------------------------------------------------------------------------


class TestLabelGenerationCorrectness:
    """Property 2: For any valid OHLC DataFrame, generate_labels() SHALL produce
    a binary array where label[i] == 1 if and only if close[i+1] > close[i],
    and label[i] == 0 otherwise.

    **Validates: Requirements 1.3**
    """

    @given(df=ohlc_dataframes(min_rows=55, max_rows=250))
    @settings(max_examples=100, deadline=None)
    def test_labels_are_binary(self, df: pd.DataFrame):
        """# Feature: ml-trading-improvements, Property 2: Label generation correctness

        All label values must be either 0 or 1.
        """
        ml_filter = MLDirectionalFilter({"enabled": True})
        labels = ml_filter.generate_labels(df)

        if len(labels) == 0:
            return

        unique_values = set(np.unique(labels))
        assert unique_values.issubset({0, 1}), (
            f"Labels contain non-binary values: {unique_values}"
        )

    @given(df=ohlc_dataframes(min_rows=55, max_rows=250))
    @settings(max_examples=100, deadline=None)
    def test_labels_match_next_close_comparison(self, df: pd.DataFrame):
        """# Feature: ml-trading-improvements, Property 2: Label generation correctness

        label[i] == 1 iff close[i+1] > close[i] for aligned indices.

        generate_labels() aligns with extract_features() (same NaN mask).
        We verify the label logic against the raw close data for valid rows.
        """
        ml_filter = MLDirectionalFilter({"enabled": True})

        close = df["close"].values.astype(float)
        n = len(close)

        # Compute raw labels for all rows (before alignment)
        raw_labels = np.zeros(n, dtype=int)
        raw_labels[:-1] = (close[1:] > close[:-1]).astype(int)

        # Compute the valid mask (same logic as extract_features)
        high = df["high"].values.astype(float)
        low = df["low"].values.astype(float)

        rsi = ml_filter._compute_rsi(close, period=14)
        atr = ml_filter._compute_atr(high, low, close, period=14)
        atr_ratio = atr / close
        sma50 = ml_filter._compute_sma(close, period=50)
        sma_ratio = close / sma50

        ret_1 = np.full(n, np.nan)
        ret_3 = np.full(n, np.nan)
        ret_5 = np.full(n, np.nan)
        ret_1[1:] = (close[1:] - close[:-1]) / close[:-1]
        ret_3[3:] = (close[3:] - close[:-3]) / close[:-3]
        ret_5[5:] = (close[5:] - close[:-5]) / close[:-5]

        features = np.column_stack([rsi, atr_ratio, sma_ratio, ret_1, ret_3, ret_5])
        valid_mask = ~np.isnan(features).any(axis=1)

        # Expected labels = raw_labels filtered by valid_mask
        expected_labels = raw_labels[valid_mask]

        # Actual labels from the method
        actual_labels = ml_filter.generate_labels(df)

        assert len(actual_labels) == len(expected_labels), (
            f"Label length mismatch: got {len(actual_labels)}, expected {len(expected_labels)}"
        )
        np.testing.assert_array_equal(actual_labels, expected_labels)


# ---------------------------------------------------------------------------
# Property 3: Z-score normalization produces zero-mean unit-variance columns
# Feature: ml-trading-improvements, Property 3: Z-score normalization
# Validates: Requirements 1.4
# ---------------------------------------------------------------------------


class TestZScoreNormalization:
    """Property 3: For any feature matrix with more than 1 row where at least
    one column has non-zero variance, after z-score normalization each column
    SHALL have mean approx 0 (within 1e-7) and standard deviation approx 1
    (within 1e-7).

    **Validates: Requirements 1.4**
    """

    @given(matrix=feature_matrices(min_rows=2, max_rows=50, n_cols=6))
    @settings(max_examples=100, deadline=None)
    def test_normalized_columns_have_zero_mean(self, matrix: np.ndarray):
        """# Feature: ml-trading-improvements, Property 3: Z-score normalization

        After z-score normalization, columns with non-zero variance
        must have mean approx 0 (within 1e-7).
        """
        ml_filter = MLDirectionalFilter({"enabled": True})

        # Simulate training: store mean and std
        ml_filter._scaler_mean = matrix.mean(axis=0)
        ml_filter._scaler_std = matrix.std(axis=0)

        normalized = ml_filter.normalize(matrix)

        # Check each column that had non-zero variance
        for col in range(matrix.shape[1]):
            if ml_filter._scaler_std[col] > 0:
                col_mean = np.abs(normalized[:, col].mean())
                assert col_mean < 1e-7, (
                    f"Column {col} mean={col_mean}, expected approx 0 (within 1e-7)"
                )

    @given(matrix=feature_matrices(min_rows=2, max_rows=50, n_cols=6))
    @settings(max_examples=100, deadline=None)
    def test_normalized_columns_have_unit_variance(self, matrix: np.ndarray):
        """# Feature: ml-trading-improvements, Property 3: Z-score normalization

        After z-score normalization, columns with non-zero variance
        must have standard deviation approx 1 (within 1e-7).
        """
        ml_filter = MLDirectionalFilter({"enabled": True})

        # Simulate training: store mean and std
        ml_filter._scaler_mean = matrix.mean(axis=0)
        ml_filter._scaler_std = matrix.std(axis=0)

        normalized = ml_filter.normalize(matrix)

        # Check each column that had non-zero variance
        for col in range(matrix.shape[1]):
            if ml_filter._scaler_std[col] > 0:
                col_std = normalized[:, col].std()
                assert abs(col_std - 1.0) < 1e-7, (
                    f"Column {col} std={col_std}, expected approx 1 (within 1e-7)"
                )


# ---------------------------------------------------------------------------
# Property 4: Insufficient data disables ML filter
# Feature: ml-trading-improvements, Property 4: Insufficient data disables ML filter
# Validates: Requirements 1.5
# ---------------------------------------------------------------------------


class TestInsufficientDataDisablesFilter:
    """Property 4: For any DataFrame with fewer than 100 rows, after calling
    train(), the ML filter's is_enabled property SHALL be False.

    **Validates: Requirements 1.5**
    """

    @given(df=small_ohlc_dataframes(min_rows=5, max_rows=99))
    @settings(max_examples=100, deadline=None)
    def test_filter_disabled_with_insufficient_data(self, df: pd.DataFrame):
        """# Feature: ml-trading-improvements, Property 4: Insufficient data disables ML filter

        Training on fewer than 100 bars must leave is_enabled == False.
        """
        ml_filter = MLDirectionalFilter({"enabled": True})

        # Attempt training with insufficient data
        result = ml_filter.train(df)

        assert result is False, (
            f"train() should return False for {len(df)} bars (< 100)"
        )
        assert ml_filter.is_enabled is False, (
            f"is_enabled should be False after training with {len(df)} bars (< 100)"
        )


# ---------------------------------------------------------------------------
# Property 5: Prediction probability bounded to [0, 1]
# Feature: ml-trading-improvements, Property 5: Prediction probability bounded to [0, 1]
# Validates: Requirements 2.1
# ---------------------------------------------------------------------------


class TestPredictionProbabilityBounded:
    """Property 5: For any valid feature vector passed to a trained model,
    predict_probability() SHALL return a value in the closed interval [0.0, 1.0].

    **Validates: Requirements 2.1**
    """

    @given(df=training_dataframes())
    @settings(max_examples=100, deadline=None)
    def test_predict_probability_in_unit_interval(self, df: pd.DataFrame):
        """# Feature: ml-trading-improvements, Property 5: Prediction probability bounded to [0, 1]

        For any valid DataFrame used for training and then prediction,
        predict_probability() must return a value in [0.0, 1.0].
        """
        ml_filter = _train_filter(df)

        # Predict on the same data (which is valid since it was used for training)
        prob = ml_filter.predict_probability(df)

        assert isinstance(prob, float), (
            f"predict_probability() returned {type(prob)}, expected float"
        )
        assert 0.0 <= prob <= 1.0, (
            f"predict_probability() returned {prob}, expected value in [0.0, 1.0]"
        )

    @given(
        pred_df=ohlc_dataframes(min_rows=60, max_rows=150),
    )
    @settings(max_examples=100, deadline=None)
    def test_predict_probability_bounded_on_different_data(
        self, pred_df: pd.DataFrame
    ):
        """# Feature: ml-trading-improvements, Property 5: Prediction probability bounded to [0, 1]

        Training on one DataFrame and predicting on a different one
        must still return probability in [0.0, 1.0].
        """
        # Use a fixed training set (deterministic) — only prediction input varies
        np.random.seed(42)
        n = 200
        base = 3500.0
        prices = base + np.cumsum(np.random.randn(n) * 2.0)
        prices = np.clip(prices, 2500.0, 5500.0)
        train_df = pd.DataFrame({
            "open": prices,
            "high": prices + np.abs(np.random.randn(n)) * 3.0,
            "low": prices - np.abs(np.random.randn(n)) * 3.0,
            "close": prices + np.random.randn(n) * 1.5,
        }, index=pd.date_range("2024-01-01", periods=n, freq="5min"))
        train_df["high"] = train_df[["open", "high", "close"]].max(axis=1)
        train_df["low"] = train_df[["open", "low", "close"]].min(axis=1)

        ml_filter = _train_filter(train_df)

        prob = ml_filter.predict_probability(pred_df)

        assert isinstance(prob, float), (
            f"predict_probability() returned {type(prob)}, expected float"
        )
        assert 0.0 <= prob <= 1.0, (
            f"predict_probability() returned {prob}, expected value in [0.0, 1.0]"
        )


# ---------------------------------------------------------------------------
# Property 6: ML signal confirmation follows threshold rule
# Feature: ml-trading-improvements, Property 6: ML signal confirmation follows threshold rule
# Validates: Requirements 2.2, 2.3
# ---------------------------------------------------------------------------


class TestMLSignalConfirmationThresholdRule:
    """Property 6: For any signal with direction D, probability P, and threshold T:
    if D is BUY, the signal is confirmed iff P > T;
    if D is SELL, the signal is confirmed iff (1 - P) > T.

    **Validates: Requirements 2.2, 2.3**
    """

    @given(
        train_df=training_dataframes(),
        signal=signal_dicts(),
    )
    @settings(max_examples=100, deadline=None)
    def test_buy_signal_confirmed_iff_probability_exceeds_threshold(
        self, train_df: pd.DataFrame, signal: dict
    ):
        """# Feature: ml-trading-improvements, Property 6: ML signal confirmation follows threshold rule

        For BUY signals: confirmed iff P(bullish) > threshold.
        For SELL signals: confirmed iff (1 - P(bullish)) > threshold.
        """
        ml_filter = _train_filter(train_df, threshold=0.55)

        confirmed, metadata = ml_filter.confirm_signal(signal, train_df)

        # Get the actual probability used
        probability = metadata["probability"]
        threshold = metadata["threshold"]
        direction = signal["side"].upper()

        if direction == "BUY":
            expected_confirmed = probability > threshold
            assert confirmed == expected_confirmed, (
                f"BUY signal: P(bullish)={probability:.6f}, threshold={threshold}, "
                f"expected confirmed={expected_confirmed}, got confirmed={confirmed}"
            )
        elif direction == "SELL":
            p_bearish = 1.0 - probability
            expected_confirmed = p_bearish > threshold
            assert confirmed == expected_confirmed, (
                f"SELL signal: P(bearish)={p_bearish:.6f} (1 - {probability:.6f}), "
                f"threshold={threshold}, "
                f"expected confirmed={expected_confirmed}, got confirmed={confirmed}"
            )

    @given(
        train_df=training_dataframes(),
        threshold=floats(min_value=0.01, max_value=0.99, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=100, deadline=None)
    def test_threshold_variation_follows_rule(
        self, train_df: pd.DataFrame, threshold: float
    ):
        """# Feature: ml-trading-improvements, Property 6: ML signal confirmation follows threshold rule

        With varying thresholds, the confirmation rule must hold consistently.
        """
        ml_filter = _train_filter(train_df, threshold=threshold)

        # Test BUY signal
        buy_signal = {"side": "BUY", "stop_pts": 5.0, "tp_pts": 10.0, "meta": {}}
        confirmed_buy, meta_buy = ml_filter.confirm_signal(buy_signal, train_df)

        prob = meta_buy["probability"]
        expected_buy = prob > threshold
        assert confirmed_buy == expected_buy, (
            f"BUY with threshold={threshold:.4f}: P={prob:.6f}, "
            f"expected {expected_buy}, got {confirmed_buy}"
        )

        # Test SELL signal
        sell_signal = {"side": "SELL", "stop_pts": 5.0, "tp_pts": 10.0, "meta": {}}
        confirmed_sell, meta_sell = ml_filter.confirm_signal(sell_signal, train_df)

        prob_sell = meta_sell["probability"]
        p_bearish = 1.0 - prob_sell
        expected_sell = p_bearish > threshold
        assert confirmed_sell == expected_sell, (
            f"SELL with threshold={threshold:.4f}: P(bearish)={p_bearish:.6f}, "
            f"expected {expected_sell}, got {confirmed_sell}"
        )


# ---------------------------------------------------------------------------
# Property 7: Disabled ML filter passes all signals
# Feature: ml-trading-improvements, Property 7: Disabled ML filter passes all signals
# Validates: Requirements 2.5, 4.3
# ---------------------------------------------------------------------------


class TestDisabledMLFilterPassesAllSignals:
    """Property 7: For any signal, when the ML filter is disabled
    (either enabled=false in config or insufficient training data),
    confirm_signal() SHALL return (True, ...).

    **Validates: Requirements 2.5, 4.3**
    """

    @given(
        signal=signal_dicts(),
        df=ohlc_dataframes(min_rows=50, max_rows=200),
    )
    @settings(max_examples=100, deadline=None)
    def test_disabled_config_passes_all_signals(
        self, signal: dict, df: pd.DataFrame
    ):
        """# Feature: ml-trading-improvements, Property 7: Disabled ML filter passes all signals

        When enabled=False in config, all signals must pass through.
        """
        config = {
            "enabled": False,
            "probability_threshold": 0.55,
            "rolling_window_bars": 500,
            "retrain_interval_hours": 168,
            "model_path": "/tmp/test_disabled_model.joblib",
            "model_type": "logistic_regression",
        }
        ml_filter = MLDirectionalFilter(config)

        confirmed, metadata = ml_filter.confirm_signal(signal, df)

        assert confirmed is True, (
            f"Disabled ML filter should pass all signals, but rejected: "
            f"signal={signal['side']}, metadata={metadata}"
        )
        assert metadata.get("ml_enabled") is False, (
            f"metadata should indicate ml_enabled=False when disabled"
        )

    @given(
        signal=signal_dicts(),
        df=ohlc_dataframes(min_rows=10, max_rows=80),
    )
    @settings(max_examples=100, deadline=None)
    def test_insufficient_data_passes_all_signals(
        self, signal: dict, df: pd.DataFrame
    ):
        """# Feature: ml-trading-improvements, Property 7: Disabled ML filter passes all signals

        When training data is insufficient (< 100 bars), ML filter should be
        disabled and pass all signals through.
        """
        config = {
            "enabled": True,
            "probability_threshold": 0.55,
            "rolling_window_bars": 500,
            "retrain_interval_hours": 168,
            "model_path": "/tmp/test_insufficient_model.joblib",
            "model_type": "logistic_regression",
        }
        ml_filter = MLDirectionalFilter(config)

        # Train on insufficient data (< 100 bars) — should disable the filter
        ml_filter.train(df)

        # is_enabled should be False
        assert ml_filter.is_enabled is False, (
            f"ML filter should be disabled with {len(df)} bars (< 100), "
            f"but is_enabled={ml_filter.is_enabled}"
        )

        # All signals should pass through
        confirmed, metadata = ml_filter.confirm_signal(signal, df)

        assert confirmed is True, (
            f"ML filter with insufficient data should pass all signals, "
            f"but rejected: signal={signal['side']}, metadata={metadata}"
        )

    @given(signal=signal_dicts())
    @settings(max_examples=100, deadline=None)
    def test_untrained_filter_passes_all_signals(self, signal: dict):
        """# Feature: ml-trading-improvements, Property 7: Disabled ML filter passes all signals

        When no model has been trained (model is None), all signals pass through.
        """
        config = {
            "enabled": True,
            "probability_threshold": 0.55,
            "rolling_window_bars": 500,
            "retrain_interval_hours": 168,
            "model_path": "/tmp/nonexistent_model_path_xyz.joblib",
            "model_type": "logistic_regression",
        }
        ml_filter = MLDirectionalFilter(config)

        # No training performed, model should be None
        assert ml_filter.is_enabled is False, (
            "Untrained filter should be disabled"
        )

        # Create a minimal dataframe for the confirm_signal call
        df = pd.DataFrame(
            {"open": [3500.0], "high": [3501.0], "low": [3499.0], "close": [3500.5]},
            index=pd.date_range("2024-01-01", periods=1, freq="5min"),
        )

        confirmed, metadata = ml_filter.confirm_signal(signal, df)

        assert confirmed is True, (
            f"Untrained ML filter should pass all signals, but rejected: "
            f"signal={signal['side']}, metadata={metadata}"
        )


# ===========================================================================
# UNIT TESTS for ML Filter (Task 2.4)
# Tests: model save/load, retrain trigger, log messages on rejection,
#        config parsing with defaults.
# Validates: Requirements 1.6, 3.1, 3.2, 2.4, 4.1
# ===========================================================================

import logging
import os
import time
from unittest.mock import patch

import pytest


# ---------------------------------------------------------------------------
# Helper: Generate synthetic OHLC DataFrame
# ---------------------------------------------------------------------------

def _make_ohlc_df(n_bars: int = 200, seed: int = 42) -> pd.DataFrame:
    """Generate a synthetic OHLC DataFrame for unit testing."""
    rng = np.random.default_rng(seed)
    close = 100.0 + np.cumsum(rng.normal(0, 0.5, n_bars))
    high = close + rng.uniform(0.1, 1.0, n_bars)
    low = close - rng.uniform(0.1, 1.0, n_bars)
    open_ = close + rng.normal(0, 0.3, n_bars)

    return pd.DataFrame({
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
    })


# ---------------------------------------------------------------------------
# Test: Config parsing with default values (Requirement 4.1)
# ---------------------------------------------------------------------------


class TestConfigParsing:
    """Test that MLDirectionalFilter correctly parses config with defaults."""

    def test_empty_config_uses_defaults(self):
        """Empty dict produces all default config values."""
        ml = MLDirectionalFilter({})

        assert ml.probability_threshold == 0.55
        assert ml.rolling_window_bars == 500
        assert ml.retrain_interval_hours == 168
        assert ml.model_path == "data/ml_model.joblib"
        assert ml.model_type == "logistic_regression"

    def test_custom_config_overrides_defaults(self):
        """Provided config values override defaults."""
        config = {
            "enabled": True,
            "probability_threshold": 0.60,
            "rolling_window_bars": 300,
            "retrain_interval_hours": 24,
            "model_path": "/tmp/custom_model.joblib",
            "model_type": "random_forest",
        }
        ml = MLDirectionalFilter(config)

        assert ml.probability_threshold == 0.60
        assert ml.rolling_window_bars == 300
        assert ml.retrain_interval_hours == 24
        assert ml.model_path == "/tmp/custom_model.joblib"
        assert ml.model_type == "random_forest"

    def test_disabled_config(self):
        """When enabled=false, is_enabled is False even without model."""
        ml = MLDirectionalFilter({"enabled": False})
        assert ml.is_enabled is False

    def test_enabled_but_no_model_is_disabled(self):
        """When enabled=true but no model trained, is_enabled is False."""
        ml = MLDirectionalFilter({"enabled": True})
        assert ml.is_enabled is False


# ---------------------------------------------------------------------------
# Test: Model save/load round-trip with joblib (Requirements 1.6, 3.2)
# ---------------------------------------------------------------------------


class TestModelSaveLoad:
    """Test model persistence via joblib."""

    def test_save_and_load_produces_same_predictions(self, tmp_path):
        """Train → save → load in new instance → same predictions."""
        model_file = str(tmp_path / "test_model.joblib")
        df = _make_ohlc_df(200)

        # Train and save
        ml1 = MLDirectionalFilter({
            "enabled": True,
            "model_path": model_file,
        })
        success = ml1.train(df)
        assert success is True
        assert ml1.is_enabled is True

        # Get prediction from trained instance
        prob1 = ml1.predict_probability(df)

        # Load in a new instance
        ml2 = MLDirectionalFilter({
            "enabled": True,
            "model_path": model_file,
        })
        # The constructor should have loaded the model from disk
        assert ml2.is_enabled is True
        prob2 = ml2.predict_probability(df)

        # Both instances should produce the same prediction
        assert prob1 == pytest.approx(prob2, abs=1e-10)

    def test_save_creates_directory(self, tmp_path):
        """Save should create parent directories if they don't exist."""
        model_file = str(tmp_path / "nested" / "dir" / "model.joblib")
        df = _make_ohlc_df(200)

        ml = MLDirectionalFilter({
            "enabled": True,
            "model_path": model_file,
        })
        success = ml.train(df)
        assert success is True

        # Verify file exists
        assert os.path.exists(model_file)

    def test_load_nonexistent_model_stays_disabled(self, tmp_path):
        """Loading from non-existent path doesn't crash, filter stays disabled."""
        model_file = str(tmp_path / "nonexistent.joblib")

        ml = MLDirectionalFilter({
            "enabled": True,
            "model_path": model_file,
        })
        assert ml.is_enabled is False

    def test_metadata_preserved_after_save_load(self, tmp_path):
        """Model metadata (training_samples, last_trained_at) is preserved."""
        model_file = str(tmp_path / "meta_model.joblib")
        df = _make_ohlc_df(200)

        ml1 = MLDirectionalFilter({
            "enabled": True,
            "model_path": model_file,
        })
        ml1.train(df)

        # Load in new instance
        ml2 = MLDirectionalFilter({
            "enabled": True,
            "model_path": model_file,
        })

        assert ml2._training_samples == ml1._training_samples
        assert ml2._last_trained_at == ml1._last_trained_at


# ---------------------------------------------------------------------------
# Test: Retrain trigger after configured interval (Requirement 3.1)
# ---------------------------------------------------------------------------


class TestRetrainTrigger:
    """Test should_retrain() logic."""

    def test_never_trained_should_retrain(self):
        """If never trained, should_retrain() returns True."""
        ml = MLDirectionalFilter({"enabled": True})
        assert ml.should_retrain() is True

    def test_just_trained_should_not_retrain(self, tmp_path):
        """Immediately after training, should_retrain() returns False."""
        model_file = str(tmp_path / "model.joblib")
        df = _make_ohlc_df(200)

        ml = MLDirectionalFilter({
            "enabled": True,
            "model_path": model_file,
            "retrain_interval_hours": 168,
        })
        ml.train(df)
        assert ml.should_retrain() is False

    def test_retrain_after_interval_elapses(self, tmp_path):
        """After retrain interval elapses, should_retrain() returns True."""
        model_file = str(tmp_path / "model.joblib")
        df = _make_ohlc_df(200)

        ml = MLDirectionalFilter({
            "enabled": True,
            "model_path": model_file,
            "retrain_interval_hours": 1,  # 1 hour interval
        })
        ml.train(df)

        # Simulate time passing by manipulating _last_train_time
        ml._last_train_time = time.time() - 7200  # 2 hours ago
        assert ml.should_retrain() is True

    def test_retrain_disabled_filter_returns_false(self):
        """Disabled filter never triggers retrain."""
        ml = MLDirectionalFilter({"enabled": False})
        assert ml.should_retrain() is False

    def test_retrain_preserves_previous_model_on_failure(self, tmp_path):
        """If retraining fails, previous model is kept."""
        model_file = str(tmp_path / "model.joblib")
        df_good = _make_ohlc_df(200)
        df_bad = _make_ohlc_df(10)  # Too few bars

        ml = MLDirectionalFilter({
            "enabled": True,
            "model_path": model_file,
        })
        ml.train(df_good)
        assert ml.is_enabled is True

        # Store original prediction
        prob_before = ml.predict_probability(df_good)

        # Attempt retrain with insufficient data
        ml.retrain(df_bad)

        # Model should still be enabled and produce same predictions
        assert ml.is_enabled is True
        prob_after = ml.predict_probability(df_good)
        assert prob_before == pytest.approx(prob_after, abs=1e-10)


# ---------------------------------------------------------------------------
# Test: Log messages on rejection (Requirement 2.4)
# ---------------------------------------------------------------------------


class TestRejectionLogMessages:
    """Test that rejection logs contain direction, probability, and threshold."""

    @pytest.fixture
    def trained_filter(self, tmp_path):
        """Provide a trained ML filter instance with high threshold to force rejections."""
        model_file = str(tmp_path / "model.joblib")
        df = _make_ohlc_df(200)

        ml = MLDirectionalFilter({
            "enabled": True,
            "model_path": model_file,
            "probability_threshold": 0.99,  # Very high threshold to force rejections
        })
        ml.train(df)
        return ml, df

    def test_buy_rejection_log_contains_required_info(self, trained_filter, caplog):
        """BUY rejection log includes direction, probability, and threshold."""
        ml, df = trained_filter

        with caplog.at_level(logging.INFO, logger="ig-scalper.ml_filter"):
            confirmed, metadata = ml.confirm_signal({"side": "BUY"}, df)

        # With threshold=0.99, BUY is almost certainly rejected
        if not confirmed:
            # Check log message contains key information
            assert len(caplog.records) > 0
            log_msg = caplog.records[-1].message
            assert "BUY" in log_msg
            assert "REJECTED" in log_msg
            # Probability and threshold values should be in the reason
            assert metadata["direction"] == "BUY"
            assert metadata["threshold"] == 0.99
            assert isinstance(metadata["probability"], float)

    def test_sell_rejection_log_contains_required_info(self, trained_filter, caplog):
        """SELL rejection log includes direction, probability, and threshold."""
        ml, df = trained_filter

        with caplog.at_level(logging.INFO, logger="ig-scalper.ml_filter"):
            confirmed, metadata = ml.confirm_signal({"side": "SELL"}, df)

        # With threshold=0.99, SELL is almost certainly rejected
        if not confirmed:
            assert len(caplog.records) > 0
            log_msg = caplog.records[-1].message
            assert "SELL" in log_msg
            assert "REJECTED" in log_msg
            assert metadata["direction"] == "SELL"
            assert metadata["threshold"] == 0.99
            assert isinstance(metadata["probability"], float)

    def test_rejection_metadata_includes_probability_value(self, trained_filter):
        """Rejection metadata probability is a float in [0, 1]."""
        ml, df = trained_filter

        _, metadata = ml.confirm_signal({"side": "BUY"}, df)

        assert isinstance(metadata["probability"], float)
        assert 0.0 <= metadata["probability"] <= 1.0

    def test_confirmed_signal_no_rejection_log(self, tmp_path, caplog):
        """Confirmed signals should NOT produce rejection log messages."""
        model_file = str(tmp_path / "model.joblib")
        df = _make_ohlc_df(200)

        # Use very low threshold so signals pass
        ml = MLDirectionalFilter({
            "enabled": True,
            "model_path": model_file,
            "probability_threshold": 0.01,  # Extremely low, will always confirm
        })
        ml.train(df)

        with caplog.at_level(logging.INFO, logger="ig-scalper.ml_filter"):
            confirmed, metadata = ml.confirm_signal({"side": "BUY"}, df)

        if confirmed:
            # No "REJECTED" messages should appear
            rejection_logs = [r for r in caplog.records if "REJECTED" in r.message]
            assert len(rejection_logs) == 0
