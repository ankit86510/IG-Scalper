"""
Enhanced IG Price Bars Module with Standardized Logging
"""

import pandas as pd
from core.logging_utils import log_success, log_error, log_warning, safe_log
import logging

logger = logging.getLogger(__name__)


def bars_from_ig(prices_json, epic: str = "Unknown"):
    """Convert IG API price response to DataFrame with logging"""

    safe_log(logger, 'debug', f"[IG Parser] Processing price data for {epic}")

    if not prices_json:
        log_error(logger, f"IG Parser: Empty response for {epic}")
        return pd.DataFrame()

    if "prices" not in prices_json:
        log_error(logger, f"IG Parser: No 'prices' key in response for {epic}")
        return pd.DataFrame()

    prices_list = prices_json.get("prices", [])
    safe_log(logger, 'info', f"[IG Parser] Processing {len(prices_list)} price bars for {epic}")

    rows = []
    skipped = 0

    for i, b in enumerate(prices_list):
        try:
            rows.append({
                "ts": pd.to_datetime(b["snapshotTimeUTC"]),
                "open": b["openPrice"]["bid"],
                "high": b["highPrice"]["bid"],
                "low": b["lowPrice"]["bid"],
                "close": b["closePrice"]["bid"],
                "volume": b.get("lastTradedVolume", 0),
            })
        except (KeyError, Exception) as e:
            skipped += 1
            safe_log(logger, 'debug', f"[IG Parser] Bar {i} error: {e}")

    if skipped > 0:
        log_warning(logger, f"IG Parser: Skipped {skipped}/{len(prices_list)} bars")

    if not rows:
        log_error(logger, f"IG Parser: No valid bars parsed for {epic}")
        return pd.DataFrame()

    df = pd.DataFrame(rows).set_index("ts").sort_index()

    log_success(logger, f"IG Parser: Parsed {len(df)} bars for {epic}")
    safe_log(logger, 'debug', f"[IG Parser] Range: {df.index[0]} to {df.index[-1]}")

    if len(df) > 0:
        latest = df.iloc[-1]
        safe_log(logger, 'debug',
                 f"[IG Parser] Latest: O={latest['open']:.2f} H={latest['high']:.2f} "
                 f"L={latest['low']:.2f} C={latest['close']:.2f}")

    return df
