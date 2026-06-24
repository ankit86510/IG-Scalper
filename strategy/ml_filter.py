"""
ML Directional Filter — Logistic Regression / Random Forest filter
that confirms or rejects trading signals based on predicted next-bar direction.

Uses technical features (RSI, ATR ratio, SMA ratio, lagged returns) to predict
whether the next bar will be bullish or bearish.

Fail-open philosophy: errors → pass signals through (return confirmed=True).
"""

import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import joblib
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier

    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False

log = logging.getLogger("ig-scalper.ml_filter")

# Minimum bars required for training
_MIN_TRAINING_BARS = 100

# Feature names in extraction order
FEATURE_NAMES = ["rsi_14", "atr_ratio", "sma_ratio", "ret_1", "ret_3", "ret_5"]


class MLDirectionalFilter:
    """Logistic regression filter that confirms/rejects signals based on
    predicted next-bar direction."""

    def __init__(self, config: dict):
        """
        config keys:
          enabled: bool
          probability_threshold: float (default 0.55)
          rolling_window_bars: int (default 500)
          retrain_interval_hours: int (default 168)
          model_path: str (default "data/ml_model.joblib")
          model_type: str ("logistic_regression" | "random_forest")
        """
        self._config_enabled = config.get("enabled", True)
        self.probability_threshold = config.get("probability_threshold", 0.55)
        self.rolling_window_bars = config.get("rolling_window_bars", 500)
        self.retrain_interval_hours = config.get("retrain_interval_hours", 168)
        self.model_path = config.get("model_path", "data/ml_model.joblib")
        self.model_type = config.get("model_type", "logistic_regression")

        # Internal state
        self._model = None
        self._scaler_mean: np.ndarray | None = None
        self._scaler_std: np.ndarray | None = None
        self._last_trained_at: str | None = None
        self._training_samples: int = 0
        self._last_train_time: float = 0.0  # time.time() of last training

        # Try loading existing model from disk
        if self._config_enabled:
            self._try_load_model()

    # ------------------------------------------------------------------
    # Public properties
    # ------------------------------------------------------------------

    @property
    def is_enabled(self) -> bool:
        """False if config disabled, sklearn unavailable, or no trained model."""
        if not self._config_enabled:
            return False
        if not _SKLEARN_AVAILABLE:
            return False
        if self._model is None:
            return False
        return True

    # ------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------

    def extract_features(self, df: pd.DataFrame) -> np.ndarray:
        """Extract feature matrix: RSI(14), ATR_Ratio, SMA_Ratio, ret_1, ret_3, ret_5.
        Returns shape (n_samples, 6).

        Requires at least 50 rows (for SMA(50) warmup).
        Rows with NaN features are dropped.
        """
        close = df["close"].values.astype(float)
        high = df["high"].values.astype(float)
        low = df["low"].values.astype(float)

        n = len(close)

        # RSI(14)
        rsi = self._compute_rsi(close, period=14)

        # ATR(14) / close
        atr = self._compute_atr(high, low, close, period=14)
        atr_ratio = atr / close

        # close / SMA(50)
        sma50 = self._compute_sma(close, period=50)
        sma_ratio = close / sma50

        # Lagged returns
        ret_1 = np.full(n, np.nan)
        ret_3 = np.full(n, np.nan)
        ret_5 = np.full(n, np.nan)

        ret_1[1:] = (close[1:] - close[:-1]) / close[:-1]
        ret_3[3:] = (close[3:] - close[:-3]) / close[:-3]
        ret_5[5:] = (close[5:] - close[:-5]) / close[:-5]

        # Stack features
        features = np.column_stack([rsi, atr_ratio, sma_ratio, ret_1, ret_3, ret_5])

        # Remove rows with any NaN
        valid_mask = ~np.isnan(features).any(axis=1)
        features = features[valid_mask]

        return features

    # ------------------------------------------------------------------
    # Label generation
    # ------------------------------------------------------------------

    def generate_labels(self, df: pd.DataFrame) -> np.ndarray:
        """Generate binary labels: 1 if next close > current close, else 0.
        Returns shape (n_samples,).

        The label for the last bar is not defined (no "next" bar), so it is excluded.
        The returned labels align with extract_features() output (after NaN removal).
        """
        close = df["close"].values.astype(float)
        n = len(close)

        # Label[i] = 1 if close[i+1] > close[i], else 0
        labels = np.zeros(n, dtype=int)
        labels[:-1] = (close[1:] > close[:-1]).astype(int)

        # We need to align with features (which drop NaN rows from warmup)
        # Remove the same warmup rows as extract_features does
        high = df["high"].values.astype(float)
        low = df["low"].values.astype(float)

        rsi = self._compute_rsi(close, period=14)
        atr = self._compute_atr(high, low, close, period=14)
        atr_ratio = atr / close
        sma50 = self._compute_sma(close, period=50)
        sma_ratio = close / sma50

        ret_1 = np.full(n, np.nan)
        ret_3 = np.full(n, np.nan)
        ret_5 = np.full(n, np.nan)
        ret_1[1:] = (close[1:] - close[:-1]) / close[:-1]
        ret_3[3:] = (close[3:] - close[:-3]) / close[:-3]
        ret_5[5:] = (close[5:] - close[:-5]) / close[:-5]

        features = np.column_stack([rsi, atr_ratio, sma_ratio, ret_1, ret_3, ret_5])
        valid_mask = ~np.isnan(features).any(axis=1)

        # Apply same mask to labels, then drop the last one (no future bar)
        labels = labels[valid_mask]
        # The last valid row's label is based on close[i+1] which may be the last bar
        # We drop it during training only (see train())
        return labels

    # ------------------------------------------------------------------
    # Normalization
    # ------------------------------------------------------------------

    def normalize(self, features: np.ndarray) -> np.ndarray:
        """Z-score normalize features using stored mean/std from training."""
        if self._scaler_mean is None or self._scaler_std is None:
            # If no scaler stats, return raw features (shouldn't happen in normal flow)
            return features

        # Avoid division by zero for constant columns
        std = self._scaler_std.copy()
        std[std == 0] = 1.0

        return (features - self._scaler_mean) / std

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, df: pd.DataFrame) -> bool:
        """Train model on DataFrame. Returns True if successful."""
        if not _SKLEARN_AVAILABLE:
            log.warning("⚠ ML Filter: scikit-learn not available, cannot train")
            return False

        n_bars = len(df)
        if n_bars < _MIN_TRAINING_BARS:
            log.warning(
                f"⚠ ML Filter: insufficient data ({n_bars} bars < {_MIN_TRAINING_BARS}), "
                "disabling until more data accumulates"
            )
            self._model = None
            return False

        try:
            features = self.extract_features(df)
            labels = self.generate_labels(df)

            # Align: labels has same length as features,
            # but last label is unreliable (last bar's "next" may not exist)
            # Drop the last row for training
            X = features[:-1]
            y = labels[:-1]

            if len(X) < _MIN_TRAINING_BARS:
                log.warning(
                    f"⚠ ML Filter: insufficient valid samples after feature extraction "
                    f"({len(X)} < {_MIN_TRAINING_BARS}), disabling"
                )
                self._model = None
                return False

            # Compute and store z-score stats
            self._scaler_mean = X.mean(axis=0)
            self._scaler_std = X.std(axis=0)

            # Normalize
            X_norm = self.normalize(X)

            # Fit model
            if self.model_type == "random_forest":
                model = RandomForestClassifier(
                    n_estimators=100, max_depth=5, random_state=42
                )
            else:
                model = LogisticRegression(max_iter=1000, random_state=42)

            model.fit(X_norm, y)
            self._model = model
            self._training_samples = len(X)
            self._last_trained_at = datetime.now(timezone.utc).isoformat()
            self._last_train_time = time.time()

            log.info(
                f"🧠 ML Filter trained: {self.model_type}, "
                f"{self._training_samples} samples, "
                f"threshold={self.probability_threshold}"
            )

            # Save to disk
            self._save_model()
            return True

        except Exception as e:
            log.error(f"✗ ML Filter training failed: {e}")
            self._model = None
            return False

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict_probability(self, df: pd.DataFrame) -> float:
        """Return P(bullish) for the current (penultimate) bar. Range [0.0, 1.0].

        Uses penultimate bar (iloc[-2]) consistent with the strategy convention.
        """
        if not self.is_enabled:
            return 0.5  # neutral when disabled

        try:
            features = self.extract_features(df)
            if len(features) == 0:
                return 0.5

            # Use the last valid feature row (which corresponds to penultimate bar
            # since the last bar is forming/incomplete)
            last_features = features[-1:].copy()
            last_norm = self.normalize(last_features)

            # Get probability of class 1 (bullish)
            proba = self._model.predict_proba(last_norm)[0]

            # sklearn returns [P(class0), P(class1)]
            # class 1 = bullish
            if len(proba) == 2:
                return float(proba[1])
            else:
                # Edge case: only one class seen during training
                return float(proba[0])

        except Exception as e:
            log.error(f"✗ ML Filter prediction failed: {e}")
            return 0.5  # fail-open: neutral probability

    # ------------------------------------------------------------------
    # Signal confirmation
    # ------------------------------------------------------------------

    def confirm_signal(self, signal: dict, df: pd.DataFrame) -> tuple[bool, dict]:
        """
        Returns (confirmed: bool, metadata: dict).
        metadata includes: probability, threshold, direction, reason.

        Fail-open: if ML filter is disabled or errors occur, signals pass through.
        """
        if not self.is_enabled:
            return True, {
                "probability": None,
                "threshold": self.probability_threshold,
                "direction": signal.get("side", "UNKNOWN"),
                "reason": "ML filter disabled — passing signal through",
                "ml_enabled": False,
            }

        try:
            probability = self.predict_probability(df)
            direction = signal.get("side", "UNKNOWN").upper()
            threshold = self.probability_threshold

            if direction == "BUY":
                confirmed = probability > threshold
                reason = (
                    f"BUY confirmed: P(bullish)={probability:.4f} > {threshold}"
                    if confirmed
                    else f"BUY rejected: P(bullish)={probability:.4f} <= {threshold}"
                )
            elif direction == "SELL":
                # For SELL, we want P(bearish) = 1 - P(bullish) > threshold
                p_bearish = 1.0 - probability
                confirmed = p_bearish > threshold
                reason = (
                    f"SELL confirmed: P(bearish)={p_bearish:.4f} > {threshold}"
                    if confirmed
                    else f"SELL rejected: P(bearish)={p_bearish:.4f} <= {threshold}"
                )
            else:
                # Unknown direction — pass through
                confirmed = True
                reason = f"Unknown direction '{direction}' — passing through"

            metadata = {
                "probability": probability,
                "threshold": threshold,
                "direction": direction,
                "reason": reason,
                "ml_enabled": True,
                "confirmed": confirmed,
            }

            if not confirmed:
                log.info(f"🧠 ML Filter REJECTED: {reason}")

            return confirmed, metadata

        except Exception as e:
            log.error(f"✗ ML Filter error in confirm_signal: {e}")
            # Fail-open
            return True, {
                "probability": None,
                "threshold": self.probability_threshold,
                "direction": signal.get("side", "UNKNOWN"),
                "reason": f"ML filter error: {e} — passing signal through",
                "ml_enabled": True,
                "error": str(e),
            }

    # ------------------------------------------------------------------
    # Retraining
    # ------------------------------------------------------------------

    def should_retrain(self) -> bool:
        """Check if retrain interval has elapsed."""
        if not self._config_enabled or not _SKLEARN_AVAILABLE:
            return False
        if self._last_train_time == 0.0:
            # Never trained — should train
            return True
        elapsed_hours = (time.time() - self._last_train_time) / 3600.0
        return elapsed_hours >= self.retrain_interval_hours

    def retrain(self, df: pd.DataFrame) -> bool:
        """Retrain on latest data, save to disk. Returns True if successful.

        If retraining fails, continues using the previous model.
        """
        previous_model = self._model
        previous_mean = self._scaler_mean
        previous_std = self._scaler_std

        success = self.train(df)

        if not success and previous_model is not None:
            # Restore previous model on failure
            log.warning("⚠ ML Filter: retraining failed, keeping previous model")
            self._model = previous_model
            self._scaler_mean = previous_mean
            self._scaler_std = previous_std

        return success

    # ------------------------------------------------------------------
    # Model persistence
    # ------------------------------------------------------------------

    def _save_model(self) -> None:
        """Save model and scaler stats to disk via joblib."""
        if not _SKLEARN_AVAILABLE or self._model is None:
            return

        try:
            # Ensure directory exists
            model_dir = os.path.dirname(self.model_path)
            if model_dir:
                os.makedirs(model_dir, exist_ok=True)

            metadata = {
                "model": self._model,
                "scaler_mean": self._scaler_mean,
                "scaler_std": self._scaler_std,
                "last_trained_at": self._last_trained_at,
                "training_samples": self._training_samples,
                "model_type": self.model_type,
                "feature_names": FEATURE_NAMES,
            }

            joblib.dump(metadata, self.model_path)
            log.info(f"🧠 ML Filter: model saved to {self.model_path}")

        except Exception as e:
            log.error(f"✗ ML Filter: failed to save model: {e}")

    def _try_load_model(self) -> None:
        """Try loading a pre-trained model from disk."""
        if not _SKLEARN_AVAILABLE:
            return

        if not os.path.exists(self.model_path):
            log.info("🧠 ML Filter: no pre-trained model found, will train on first data")
            return

        try:
            metadata = joblib.load(self.model_path)

            self._model = metadata["model"]
            self._scaler_mean = metadata["scaler_mean"]
            self._scaler_std = metadata["scaler_std"]
            self._last_trained_at = metadata.get("last_trained_at")
            self._training_samples = metadata.get("training_samples", 0)

            # Set last_train_time from stored timestamp
            if self._last_trained_at:
                try:
                    dt = datetime.fromisoformat(self._last_trained_at)
                    self._last_train_time = dt.timestamp()
                except (ValueError, TypeError):
                    self._last_train_time = time.time()
            else:
                self._last_train_time = time.time()

            log.info(
                f"🧠 ML Filter: loaded model from {self.model_path} "
                f"(trained on {self._training_samples} samples)"
            )

        except Exception as e:
            log.error(f"✗ ML Filter: failed to load model from {self.model_path}: {e}")
            self._model = None

    # ------------------------------------------------------------------
    # Technical indicator helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_rsi(close: np.ndarray, period: int = 14) -> np.ndarray:
        """Compute RSI(period). Returns array same length as close, with NaN for warmup."""
        n = len(close)
        rsi = np.full(n, np.nan)

        if n < period + 1:
            return rsi

        deltas = np.diff(close)
        gains = np.where(deltas > 0, deltas, 0.0)
        losses = np.where(deltas < 0, -deltas, 0.0)

        # Initial average (simple mean of first `period` values)
        avg_gain = gains[:period].mean()
        avg_loss = losses[:period].mean()

        if avg_loss == 0:
            rsi[period] = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi[period] = 100.0 - (100.0 / (1.0 + rs))

        # Smoothed (Wilder's method)
        for i in range(period, len(deltas)):
            avg_gain = (avg_gain * (period - 1) + gains[i]) / period
            avg_loss = (avg_loss * (period - 1) + losses[i]) / period

            if avg_loss == 0:
                rsi[i + 1] = 100.0
            else:
                rs = avg_gain / avg_loss
                rsi[i + 1] = 100.0 - (100.0 / (1.0 + rs))

        return rsi

    @staticmethod
    def _compute_atr(
        high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14
    ) -> np.ndarray:
        """Compute ATR(period). Returns array same length as input, with NaN for warmup."""
        n = len(close)
        atr = np.full(n, np.nan)

        if n < period + 1:
            return atr

        # True Range
        tr = np.full(n, np.nan)
        tr[0] = high[0] - low[0]
        for i in range(1, n):
            tr[i] = max(
                high[i] - low[i],
                abs(high[i] - close[i - 1]),
                abs(low[i] - close[i - 1]),
            )

        # Initial ATR (simple average)
        atr[period] = tr[1 : period + 1].mean()

        # Smoothed (Wilder's method)
        for i in range(period + 1, n):
            atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period

        return atr

    @staticmethod
    def _compute_sma(close: np.ndarray, period: int = 50) -> np.ndarray:
        """Compute SMA(period). Returns array same length as close, with NaN for warmup."""
        n = len(close)
        sma = np.full(n, np.nan)

        if n < period:
            return sma

        # Cumulative sum approach for efficiency
        cumsum = np.cumsum(close)
        sma[period - 1 :] = (cumsum[period - 1 :] - np.concatenate([[0], cumsum[:-period]])) / period

        return sma
