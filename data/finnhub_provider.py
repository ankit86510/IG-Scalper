"""
Finnhub WebSocket Real-Time Data Provider

Connects to Finnhub's free WebSocket to receive real-time forex ticks
and aggregates them into OHLCV bars (1min, 5min, etc.).

Free tier: Unlimited WebSocket connections, 60 REST API calls/min.
Symbols: OANDA:XAU_USD (Gold spot), OANDA:EUR_USD, etc.

Usage:
    provider = FinnhubRealtimeProvider(api_key="your_key")
    provider.subscribe("CS.D.CFEGOLD.CEB.IP")
    df = provider.get_bars("CS.D.CFEGOLD.CEB.IP", timeframe="5min", limit=288)
"""

import threading
import time
import json
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from collections import deque
from typing import Dict, Optional, List

logger = logging.getLogger(__name__)

# Map IG epics → Finnhub forex symbols
IG_TO_FINNHUB = {
    "CS.D.CFEGOLD.CEB.IP":   "OANDA:XAU_USD",
    "CS.D.CFESILVER.CEB.IP": "OANDA:XAG_USD",
    "CS.D.EURUSD.CFD.IP":    "OANDA:EUR_USD",
    "CS.D.GBPUSD.CFD.IP":    "OANDA:GBP_USD",
    "CS.D.USDJPY.CFD.IP":    "OANDA:USD_JPY",
    "CMD.USCrude.CFD.IP":    "OANDA:BCO_USD",
}

# Timeframe → seconds
TIMEFRAME_SECONDS = {
    "1min":  60,
    "3min":  180,
    "5min":  300,
    "15min": 900,
    "30min": 1800,
    "60min": 3600,
}


class BarAggregator:
    """
    Aggregates raw ticks into OHLCV bars of a given timeframe.
    Stores completed bars in a ring buffer.
    """

    def __init__(self, timeframe: str = "5min", max_bars: int = 500):
        self.interval_sec = TIMEFRAME_SECONDS.get(timeframe, 300)
        self.max_bars = max_bars
        self.bars: deque = deque(maxlen=max_bars)
        self._current_bar: Optional[Dict] = None
        self._current_bar_end: float = 0
        self._lock = threading.Lock()
        self.tick_count = 0

    def _bar_start_time(self, ts_sec: float) -> float:
        """Round timestamp down to bar boundary."""
        return (int(ts_sec) // self.interval_sec) * self.interval_sec

    def on_tick(self, price: float, timestamp_ms: int):
        """Process a single tick. Aggregates into current bar."""
        ts_sec = timestamp_ms / 1000.0
        bar_start = self._bar_start_time(ts_sec)
        bar_end = bar_start + self.interval_sec

        with self._lock:
            self.tick_count += 1

            # New bar period — close previous bar if exists
            if self._current_bar and bar_start >= self._current_bar_end:
                self.bars.append(dict(self._current_bar))
                self._current_bar = None

            # Start new bar or update current
            if self._current_bar is None:
                self._current_bar = {
                    "ts": pd.Timestamp(bar_start, unit="s", tz="UTC"),
                    "open":   price,
                    "high":   price,
                    "low":    price,
                    "close":  price,
                    "volume": 1,
                }
                self._current_bar_end = bar_end
            else:
                self._current_bar["high"] = max(self._current_bar["high"], price)
                self._current_bar["low"] = min(self._current_bar["low"], price)
                self._current_bar["close"] = price
                self._current_bar["volume"] += 1

    def get_dataframe(self, limit: int = 288,
                      include_forming: bool = True) -> pd.DataFrame:
        """Return bars as DataFrame. Includes current forming bar if requested."""
        with self._lock:
            bars = list(self.bars)
            forming = dict(self._current_bar) if (include_forming and self._current_bar) else None

        if not bars and not forming:
            return pd.DataFrame()

        all_bars = bars[-limit:]
        if forming:
            all_bars.append(forming)

        df = pd.DataFrame(all_bars).set_index("ts").sort_index()
        return df

    def seed_history(self, hist_df: pd.DataFrame):
        """Pre-seed from external historical data."""
        with self._lock:
            for ts, row in hist_df.iterrows():
                if ts.tzinfo is None:
                    ts = ts.tz_localize("UTC")
                bar = {
                    "ts":     ts,
                    "open":   float(row["open"]),
                    "high":   float(row["high"]),
                    "low":    float(row["low"]),
                    "close":  float(row["close"]),
                    "volume": int(row.get("volume", 0)),
                }
                self.bars.append(bar)


class FinnhubRealtimeProvider:
    """
    Finnhub WebSocket provider for real-time forex/commodity ticks.
    Aggregates ticks into OHLCV bars on any timeframe.

    Free tier: unlimited WebSocket, real-time forex ticks.
    Get API key at: https://finnhub.io (free signup)
    """

    def __init__(self, api_key: str, timeframe: str = "5min",
                 max_bars: int = 500):
        self.api_key = api_key
        self.timeframe = timeframe
        self._ws = None
        self._ws_thread = None
        self._connected = False
        self._running = False

        # Per-symbol aggregators
        self._aggregators: Dict[str, BarAggregator] = {}
        # Map finnhub symbol → ig epic for reverse lookup
        self._finnhub_to_ig: Dict[str, str] = {}

        self.max_bars = max_bars
        self._connect()

    def _connect(self):
        """Connect to Finnhub WebSocket in a background thread."""
        try:
            import websocket
        except ImportError:
            logger.error(
                "[Finnhub] websocket-client not installed. "
                "Run: pip install websocket-client"
            )
            return

        self._running = True
        self._ws_thread = threading.Thread(target=self._ws_loop, daemon=True)
        self._ws_thread.start()

        # Wait briefly for connection
        for _ in range(20):
            if self._connected:
                break
            time.sleep(0.25)

        if self._connected:
            logger.info("[Finnhub] WebSocket connected (real-time forex ticks)")
        else:
            logger.warning("[Finnhub] WebSocket connection pending...")

    def _ws_loop(self):
        """Background thread running the WebSocket connection."""
        import websocket

        url = f"wss://ws.finnhub.io?token={self.api_key}"

        def on_open(ws):
            self._connected = True
            logger.info("[Finnhub] WebSocket opened")
            # Subscribe to all registered symbols
            for fh_symbol in self._finnhub_to_ig.keys():
                msg = json.dumps({"type": "subscribe", "symbol": fh_symbol})
                ws.send(msg)
                logger.info(f"[Finnhub] Subscribed: {fh_symbol}")

        def on_message(ws, message):
            try:
                data = json.loads(message)
                if data.get("type") == "trade":
                    for trade in data.get("data", []):
                        symbol = trade.get("s")
                        price = trade.get("p")
                        ts_ms = trade.get("t")
                        if symbol and price and ts_ms:
                            ig_epic = self._finnhub_to_ig.get(symbol)
                            if ig_epic and ig_epic in self._aggregators:
                                self._aggregators[ig_epic].on_tick(price, ts_ms)
            except Exception as e:
                logger.error(f"[Finnhub] Message parse error: {e}")

        def on_error(ws, error):
            logger.error(f"[Finnhub] WebSocket error: {error}")

        def on_close(ws, close_code, close_msg):
            self._connected = False
            logger.warning(f"[Finnhub] WebSocket closed: {close_code} {close_msg}")
            # Auto-reconnect
            if self._running:
                time.sleep(5)
                self._ws_loop()

        self._ws = websocket.WebSocketApp(
            url,
            on_open=on_open,
            on_message=on_message,
            on_error=on_error,
            on_close=on_close
        )
        self._ws.run_forever()

    def subscribe(self, ig_epic: str, seed_df: Optional[pd.DataFrame] = None):
        """
        Subscribe to real-time ticks for an IG epic.
        Optionally seed with historical bars from TwelveData/etc.
        """
        fh_symbol = IG_TO_FINNHUB.get(ig_epic)
        if not fh_symbol:
            logger.warning(f"[Finnhub] No mapping for {ig_epic}")
            return

        if ig_epic not in self._aggregators:
            agg = BarAggregator(timeframe=self.timeframe, max_bars=self.max_bars)
            if seed_df is not None and not seed_df.empty:
                agg.seed_history(seed_df)
                logger.info(f"[Finnhub] Seeded {len(seed_df)} bars for {ig_epic}")
            self._aggregators[ig_epic] = agg
            self._finnhub_to_ig[fh_symbol] = ig_epic

        # Send subscribe if already connected
        if self._connected and self._ws:
            msg = json.dumps({"type": "subscribe", "symbol": fh_symbol})
            self._ws.send(msg)
            logger.info(f"[Finnhub] Subscribed: {fh_symbol}")

    def get_bars(self, ig_epic: str, timeframe: str = "5min",
                 limit: int = 288) -> pd.DataFrame:
        """Return OHLCV bars for the given epic."""
        agg = self._aggregators.get(ig_epic)
        if agg is None:
            self.subscribe(ig_epic)
            agg = self._aggregators.get(ig_epic)
            if agg is None:
                return pd.DataFrame()

        df = agg.get_dataframe(limit=limit, include_forming=True)

        if not df.empty:
            logger.info(
                f"[Finnhub] {ig_epic} {timeframe}: {len(df)} bars "
                f"(ticks: {agg.tick_count}) | close: {df['close'].iloc[-1]:.2f}"
            )
        return df

    def is_connected(self) -> bool:
        return self._connected

    def disconnect(self):
        self._running = False
        if self._ws:
            self._ws.close()
        self._connected = False
        logger.info("[Finnhub] Disconnected")
