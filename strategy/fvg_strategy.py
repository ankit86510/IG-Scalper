"""FVG Multi-Timeframe Strategy.

Extends strategy.base.Strategy ABC. Orchestrates the analysis cycle
across 60min → 15min → 5min timeframes, derives directional bias from
higher timeframes, and generates trade signals on the 5min chart when
alignment exists.

Configuration is read from settings_ai.yaml under the `fvg_strategy` key.
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional
from zoneinfo import ZoneInfo

import pandas as pd

from strategy.base import Strategy
from strategy.fvg_bias import BiasCalculator
from strategy.fvg_detector import FVG, FVGDetector
from strategy.fvg_rate_budget import RateBudgetManager
from strategy.fvg_scheduler import CycleScheduler
from strategy.fvg_signal import SignalGenerator

logger = logging.getLogger("ig-scalper")

ROME_TZ = ZoneInfo("Europe/Rome")

# Default configuration values (Req 9.2)
_DEFAULTS: Dict = {
    "cycle_interval_seconds": 300,
    "timeframes": ["60min", "15min", "5min"],
    "fvg_max_age_bars": 50,
    "stop_buffer_points": 2.0,
    "min_bias_confidence": 0.6,
    "lookback_candles": 200,
}


def _resolve_config(config: dict) -> dict:
    """Merge user config with defaults. Logs warnings for missing keys (Req 9.3).

    Args:
        config: User-provided configuration dict.

    Returns:
        Resolved config dict with all required keys populated.
    """
    resolved = {}
    for key, default in _DEFAULTS.items():
        if key not in config:
            logger.warning(
                f"FVGStrategy: config key '{key}' missing, using default: {default}"
            )
            resolved[key] = default
        else:
            resolved[key] = config[key]
    return resolved


def _validate_config(config: dict) -> None:
    """Validate config values at initialization (Req 9.4).

    Raises:
        ValueError: If any config value is invalid.
    """
    interval = config["cycle_interval_seconds"]
    if not isinstance(interval, (int, float)) or interval <= 0:
        raise ValueError(
            f"cycle_interval_seconds must be > 0, got {interval}"
        )

    timeframes = config["timeframes"]
    if not isinstance(timeframes, list) or len(timeframes) == 0:
        raise ValueError(
            f"timeframes must be a non-empty list, got {timeframes}"
        )

    max_age = config["fvg_max_age_bars"]
    if not isinstance(max_age, (int, float)) or max_age <= 0:
        raise ValueError(
            f"fvg_max_age_bars must be > 0, got {max_age}"
        )

    stop_buffer = config["stop_buffer_points"]
    if not isinstance(stop_buffer, (int, float)) or stop_buffer < 0:
        raise ValueError(
            f"stop_buffer_points must be >= 0, got {stop_buffer}"
        )

    min_conf = config["min_bias_confidence"]
    if not isinstance(min_conf, (int, float)) or not (0 <= min_conf <= 1):
        raise ValueError(
            f"min_bias_confidence must be between 0 and 1, got {min_conf}"
        )

    lookback = config["lookback_candles"]
    if not isinstance(lookback, (int, float)) or lookback <= 0:
        raise ValueError(
            f"lookback_candles must be > 0, got {lookback}"
        )


class FVGStrategy(Strategy):
    """Multi-timeframe FVG strategy extending the Strategy ABC.

    Orchestrates the analysis cycle (60min → 15min → 5min) and caches
    signals between cycles. Implements on_bar(df) as required by the
    strategy architecture.
    """

    def __init__(
        self,
        config: dict,
        data_provider,
        symbol_epic: str = "CS.D.CFEGOLD.CEB.IP",
    ):
        """Initialize FVGStrategy with validated configuration.

        Args:
            config: Configuration dict (typically from settings_ai.yaml fvg_strategy key).
                    Missing keys are filled with defaults and logged as warnings.
            data_provider: Object with get_bars(symbol, timeframe, limit) method
                           (e.g., SmartDataAggregator or TwelveDataProvider).
            symbol_epic: IG epic for the instrument to trade.

        Raises:
            ValueError: If any config value is invalid (Req 9.4).
        """
        # Resolve defaults and validate (Req 9.2, 9.3, 9.4)
        self.config = _resolve_config(config)
        _validate_config(self.config)

        # Core components
        self.detector = FVGDetector()
        self.bias_calc = BiasCalculator()
        self.signal_gen = SignalGenerator(
            stop_buffer=self.config["stop_buffer_points"],
            min_confidence=self.config["min_bias_confidence"],
        )
        self.scheduler = CycleScheduler(
            interval_seconds=int(self.config["cycle_interval_seconds"]),
            timeframes=self.config["timeframes"],
        )
        self.data_provider = data_provider
        self.symbol_epic = symbol_epic

        # Rate budget manager (Req 6.1-6.6)
        # Only fetching 60min + 15min via provider; 5min comes from on_bar()
        self.rate_budget = RateBudgetManager(
            data_provider=data_provider,
            num_timeframes=len(self.config["timeframes"]) - 1,
            num_symbols=1,
        )

        # State
        self._cached_signal: Optional[dict] = None
        self._active_fvgs: Dict[str, List[FVG]] = {}

        logger.info(
            f"FVGStrategy initialized | cycle={self.config['cycle_interval_seconds']}s | "
            f"timeframes={self.config['timeframes']} | "
            f"max_age={self.config['fvg_max_age_bars']} | "
            f"stop_buffer={self.config['stop_buffer_points']} | "
            f"min_confidence={self.config['min_bias_confidence']} | "
            f"lookback={self.config['lookback_candles']}"
        )

    def on_bar(self, df: pd.DataFrame) -> Optional[dict]:
        """Called with 5min DataFrame. Triggers cycle if interval elapsed.

        If the cycle interval has not elapsed, returns the cached signal
        from the last completed analysis cycle (or None).

        Req 7.2: Triggers internal analysis cycle using provided 5min df.
        Req 7.5: Returns cached signal if interval not elapsed.

        Args:
            df: DataFrame with OHLC columns from the 5min timeframe.

        Returns:
            Signal dict {"side", "stop_pts", "tp_pts", "meta"} or None.
        """
        if not self.scheduler.should_run():
            # Req 7.5: Return cached signal when called more frequently than cycle interval
            return self._cached_signal

        # Run analysis cycle
        self.scheduler.mark_cycle_start()
        try:
            signal = self._run_analysis_cycle(df)
            self._cached_signal = signal
        except Exception as e:
            logger.error(f"FVGStrategy analysis cycle failed: {e}")
            self._cached_signal = None
        finally:
            self.scheduler.mark_cycle_complete()

        return self._cached_signal

    def _run_analysis_cycle(self, df_5min: pd.DataFrame) -> Optional[dict]:
        """Execute the full 60min → 15min → 5min cascade.

        Flow:
        1. Check rate budget (Req 6.2, 6.3)
        2. Fetch 60min bars via data_provider
        3. Detect 60min FVGs, update fill status
        4. Calculate 60min bias
        5. Fetch 15min bars via data_provider
        6. Detect 15min FVGs, update fill status
        7. Adjust bias with 15min
        8. Detect 5min FVGs from on_bar df, update fill status
        9. Generate signal from 5min FVGs + bias + HTF FVGs
        10. Return signal or None

        Args:
            df_5min: DataFrame with OHLC columns from the 5min timeframe.

        Returns:
            Signal dict or None.
        """
        lookback = int(self.config["lookback_candles"])
        max_age = int(self.config["fvg_max_age_bars"])
        timeframes = self.config["timeframes"]

        # --- Rate limit pre-check (Req 6.2, 6.3) ---
        if not self.rate_budget.should_proceed_with_cycle():
            now_rome = datetime.now(tz=ROME_TZ)
            logger.warning(
                f"FVGStrategy: Cycle skipped at {now_rome.strftime('%Y-%m-%d %H:%M:%S %Z')} — "
                f"insufficient rate budget"
            )
            return None

        self.rate_budget.record_cycle_start()

        # --- Step 1: Fetch and analyze 60min ---
        htf = timeframes[0] if len(timeframes) > 0 else "60min"

        self.rate_budget.record_request()
        df_60min = self.data_provider.get_bars(self.symbol_epic, htf, lookback)

        if df_60min is None or df_60min.empty:
            now_rome = datetime.now(tz=ROME_TZ)
            logger.warning(
                f"FVGStrategy: empty data for {htf} at {now_rome.strftime('%Y-%m-%d %H:%M:%S %Z')}, "
                f"aborting cycle"
            )
            self.rate_budget.log_cycle_consumption()
            return None

        # Detect 60min FVGs
        fvgs_60min = self.detector.detect(df_60min, htf)
        fvgs_60min = self.detector.update_fill_status(fvgs_60min, df_60min, max_age)
        self._active_fvgs[htf] = fvgs_60min

        logger.info(f"FVGStrategy [{htf}]: detected {len(fvgs_60min)} active FVGs")

        # Calculate 60min bias (Req 3.2, 3.3)
        bias = self.bias_calc.calculate_60min_bias(fvgs_60min)

        if bias.direction == "neutral" and bias.confidence == 0.0:
            now_rome = datetime.now(tz=ROME_TZ)
            logger.info(
                f"FVGStrategy: no active 60min FVGs at "
                f"{now_rome.strftime('%Y-%m-%d %H:%M:%S %Z')}, checking lower timeframes..."
            )

        # --- Step 2: Fetch and analyze 15min ---
        mtf = timeframes[1] if len(timeframes) > 1 else "15min"

        self.rate_budget.record_request()
        df_15min = self.data_provider.get_bars(self.symbol_epic, mtf, lookback)

        if df_15min is None or df_15min.empty:
            now_rome = datetime.now(tz=ROME_TZ)
            logger.warning(
                f"FVGStrategy: empty data for {mtf} at {now_rome.strftime('%Y-%m-%d %H:%M:%S %Z')}, "
                f"aborting cycle"
            )
            self.rate_budget.log_cycle_consumption()
            return None

        # Detect 15min FVGs
        fvgs_15min = self.detector.detect(df_15min, mtf)
        fvgs_15min = self.detector.update_fill_status(fvgs_15min, df_15min, max_age)
        self._active_fvgs[mtf] = fvgs_15min

        logger.info(f"FVGStrategy [{mtf}]: detected {len(fvgs_15min)} active FVGs")

        # Adjust bias with 15min (Req 3.4, 3.5)
        # If 60min bias is neutral, use 15min bias directly
        if bias.direction == "neutral" and bias.confidence == 0.0:
            bias = self.bias_calc.calculate_60min_bias(fvgs_15min)  # Reuse same logic for 15min
            if bias.direction != "neutral":
                logger.info(f"FVGStrategy: using 15min bias as primary: {bias.direction}@{bias.confidence:.2f}")
        else:
            bias = self.bias_calc.adjust_with_15min(bias, fvgs_15min)

        logger.info(
            f"FVGStrategy bias after 15min adjustment: "
            f"direction={bias.direction}, confidence={bias.confidence:.2f}"
        )

        # If still neutral after 15min, skip signal generation
        if bias.direction == "neutral" and bias.confidence == 0.0:
            now_rome = datetime.now(tz=ROME_TZ)
            logger.info(
                f"FVGStrategy: neutral bias (no active FVGs on 60min or 15min) at "
                f"{now_rome.strftime('%Y-%m-%d %H:%M:%S %Z')}, skipping signal generation"
            )
            self.rate_budget.log_cycle_consumption()
            return None

        # --- Step 3: Analyze 5min from on_bar DataFrame ---
        ltf = timeframes[2] if len(timeframes) > 2 else "5min"
        fvgs_5min = self.detector.detect(df_5min, ltf)
        fvgs_5min = self.detector.update_fill_status(fvgs_5min, df_5min, max_age)
        self._active_fvgs[ltf] = fvgs_5min

        logger.info(f"FVGStrategy [{ltf}]: detected {len(fvgs_5min)} active FVGs")

        # --- Step 4: Generate signal (Req 4) ---
        fvgs_higher_tf = fvgs_60min + fvgs_15min
        signal = self.signal_gen.generate(fvgs_5min, bias, fvgs_higher_tf)

        if signal:
            # Req 8.3: Log signal details with entry zone, stop, TP, alignment rationale
            now_rome = datetime.now(tz=ROME_TZ)
            meta = signal.get("meta", {})
            trigger = meta.get("trigger_fvg", {})
            logger.info(
                f"FVGStrategy SIGNAL at {now_rome.strftime('%Y-%m-%d %H:%M:%S %Z')}: "
                f"{signal['side']} | "
                f"entry_zone=[{trigger.get('zone_lower', 0):.2f}, {trigger.get('zone_upper', 0):.2f}] | "
                f"stop={signal['stop_pts']:.2f} | tp={signal['tp_pts']:.2f} | "
                f"R:R=1:{signal['tp_pts']/signal['stop_pts']:.2f} | "
                f"bias={bias.direction}@{bias.confidence:.2f} | "
                f"alignment: {signal['side']} aligned with {bias.direction} bias from "
                f"{len(fvgs_60min)} 60min + {len(fvgs_15min)} 15min FVGs"
            )
        else:
            logger.info("FVGStrategy: no signal generated this cycle")

        # --- Log budget consumption (Req 6.6) ---
        self.rate_budget.log_cycle_consumption()

        return signal
