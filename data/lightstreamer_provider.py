"""
IG Lightstreamer Data Provider

Receives real-time OHLCV bars pushed directly by IG via Lightstreamer.
No external API, no rate limits — same price feed as IG's own charts.

How bar capture works:
  - IG sends CHART updates in MERGE mode: each update contains the
    current forming bar's OHLCV + CONS_END flag.
  - CONS_END=1 means the bar just closed (new candle started).
  - We commit a bar when CONS_END=1 OR when UTM changes (new bar started).
  - On startup we pre-seed history from IG REST so the strategy has
    enough bars immediately without waiting for Lightstreamer to fill up.

Subscription:
  Item  : CHART:{epic}:{scale}   e.g. CHART:CS.D.CFEGOLD.CEB.IP:1MINUTE
  Mode  : MERGE
  Fields: UTM, BID_OPEN, BID_HIGH, BID_LOW, BID_CLOSE, CONS_END, CONS_TICK_COUNT
"""

import threading
import time
import logging
import pandas as pd
from collections import deque
from typing import Dict, Optional

logger = logging.getLogger(__name__)

# IG Lightstreamer scale codes
SCALE_MAP = {
    "1min":  "1MINUTE",
    "3min":  "3MINUTE",
    "5min":  "5MINUTE",
    "15min": "15MINUTE",
    "30min": "30MINUTE",
    "60min": "HOUR",
}

# IG REST resolution codes for pre-seeding history
REST_RESOLUTION_MAP = {
    "1min":  "MINUTE",
    "3min":  "MINUTE_3",
    "5min":  "MINUTE_5",
    "15min": "MINUTE_15",
    "30min": "MINUTE_30",
    "60min": "HOUR",
}

FIELDS = ["UTM", "BID_OPEN", "BID_HIGH", "BID_LOW", "BID_CLOSE",
          "OFR_OPEN", "OFR_HIGH", "OFR_LOW", "OFR_CLOSE",
          "CONS_END", "CONS_TICK_COUNT", "LTV"]

# Lightstreamer subscription mode for CHART items
# MERGE: IG requires MERGE mode for CHART subscriptions
CHART_MODE = "MERGE"


class _BarListener:
    """
    Lightstreamer subscription listener.
    Commits completed bars to the deque using two triggers:
      1. CONS_END=1  — IG signals bar is closed
      2. UTM changes — new bar started, so previous bar is complete
    """

    def __init__(self, epic: str, max_bars: int = 500):
        self.epic = epic
        self.bars: deque = deque(maxlen=max_bars)
        self._current: Dict = {}
        self._last_utm: Optional[str] = None
        self._lock = threading.Lock()
        self.update_count = 0

    def onListenStart(self):
        logger.info(f"[LS] Subscription started: {self.epic}")

    def onListenEnd(self):
        logger.info(f"[LS] Subscription ended: {self.epic}")

    def onSubscriptionError(self, code, message):
        logger.error(f"[LS] Subscription error {code}: {message}")

    def onEndOfSnapshot(self, item_name, item_pos):
        logger.debug(f"[LS] End of snapshot: {item_name}")

    def onItemUpdate(self, update):
        """Called on every CHART MERGE update."""
        try:
            def get(field):
                v = update.getValue(field)
                return v if v and v != "" else None

            utm       = get("UTM")
            bid_open  = get("BID_OPEN")
            bid_high  = get("BID_HIGH")
            bid_low   = get("BID_LOW")
            bid_close = get("BID_CLOSE")
            cons_end  = get("CONS_END")
            ltv       = get("LTV")

            with self._lock:
                self.update_count += 1

                # Always update current forming bar with latest values
                if utm:
                    # If UTM changed, the previous bar period ended — commit it
                    if self._last_utm and utm != self._last_utm:
                        self._commit_current_bar()
                        self._current = {}
                    self._last_utm = utm
                    self._current["utm"] = int(utm)

                if bid_open  is not None: self._current["open"]  = float(bid_open)
                if bid_high  is not None: self._current["high"]  = float(bid_high)
                if bid_low   is not None: self._current["low"]   = float(bid_low)
                if bid_close is not None: self._current["close"] = float(bid_close)
                if ltv       is not None: self._current["volume"] = int(float(ltv))

                # CONS_END=1 → IG signals this bar is closed, commit it now
                if cons_end == "1":
                    self._commit_current_bar()
                    self._current = {}
                    self._last_utm = None

                if self.update_count <= 3 or self.update_count % 50 == 0:
                    logger.debug(
                        f"[LS] #{self.update_count} {self.epic} "
                        f"close={bid_close} CONS_END={cons_end} UTM={utm}"
                    )

        except Exception as e:
            logger.error(f"[LS] onItemUpdate error: {e}")

    def _commit_current_bar(self):
        """Commit current forming bar if it has all required fields."""
        c = self._current
        if all(k in c for k in ("utm", "open", "high", "low", "close")):
            ts = pd.Timestamp(c["utm"], unit="ms", tz="UTC")
            bar = {
                "ts":     ts,
                "open":   c["open"],
                "high":   c["high"],
                "low":    c["low"],
                "close":  c["close"],
                "volume": c.get("volume", 0),
            }
            # Avoid duplicate bars
            if self.bars and self.bars[-1]["ts"] == ts:
                self.bars[-1] = bar  # update in place (no log — same bar)
            else:
                self.bars.append(bar)
                logger.debug(
                    f"[LS] Bar committed {self.epic}: "
                    f"O={c['open']} H={c['high']} L={c['low']} C={c['close']} "
                    f"V={c.get('volume', 0)} @ {ts}"
                )

    def seed_history(self, historical_df: pd.DataFrame):
        """Pre-seed bar history from IG REST so strategy has bars immediately."""
        with self._lock:
            for ts, row in historical_df.iterrows():
                # Normalise to UTC-aware timestamp
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
        logger.info(f"[LS] Seeded {len(historical_df)} historical bars for {self.epic}")

    def get_current_bar(self) -> Optional[Dict]:
        """Return the currently forming (incomplete) bar."""
        with self._lock:
            c = self._current
            if all(k in c for k in ("utm", "open", "high", "low", "close")):
                return {
                    "ts":     pd.Timestamp(c["utm"], unit="ms", tz="UTC"),
                    "open":   c["open"],
                    "high":   c["high"],
                    "low":    c["low"],
                    "close":  c["close"],
                    "volume": 0,
                }
        return None

    def get_dataframe(self, limit: int = 250,
                      include_forming: bool = False) -> pd.DataFrame:
        """
        Return accumulated bars as DataFrame.

        Args:
            limit:           Max number of bars to return
            include_forming: If True, append the current forming bar as last row
        """
        with self._lock:
            bars = list(self.bars)
            forming = dict(self._current) if include_forming else None

        if not bars:
            return pd.DataFrame()

        df = pd.DataFrame(bars[-limit:]).set_index("ts").sort_index()

        if include_forming and forming and all(
            k in forming for k in ("utm", "open", "high", "low", "close")
        ):
            forming_ts = pd.Timestamp(forming["utm"], unit="ms", tz="UTC")
            forming_row = pd.DataFrame([{
                "open":   forming["open"],
                "high":   forming["high"],
                "low":    forming["low"],
                "close":  forming["close"],
                "volume": 0,
            }], index=[forming_ts])
            forming_row.index.name = "ts"
            df = pd.concat([df, forming_row])
            df = df[~df.index.duplicated(keep="last")].sort_index()

        return df


class LightstreamerProvider:
    """
    Connects to IG's Lightstreamer server and maintains live
    OHLCV bar streams for subscribed instruments.

    Pre-seeds history from IG REST on subscribe so bars are
    available immediately without waiting.

    Authentication (per IG Streaming API guide):
      User:     Active account identifier (from /session response)
      Password: CST-{cst}|XST-{x_security_token}
    """

    def __init__(self, ls_endpoint: str, cst: str, x_security_token: str,
                 ig_client=None, account_id: str = ""):
        self.ls_endpoint = ls_endpoint
        self.cst = cst
        self.x_security_token = x_security_token
        self.ig_client = ig_client  # used for REST history seeding
        self.account_id = account_id or (ig_client.account_id if ig_client and hasattr(ig_client, 'account_id') else "")

        self._client = None
        self._ls_sub_class = None
        self._listeners: Dict[str, _BarListener] = {}
        self._subscriptions = []  # Track subscriptions for reconnection
        self._connected = False
        self._reconnect_attempts = 0
        self._reconnecting = False  # Prevent concurrent reconnect threads
        self._max_reconnect_attempts = 5
        self._connect()

    def _connect(self):
        """Establish Lightstreamer connection with correct IG auth format."""
        try:
            from lightstreamer.client import LightstreamerClient, Subscription
            self._ls_sub_class = Subscription

            client = LightstreamerClient(self.ls_endpoint, "DEFAULT")

            # IG Streaming API auth format:
            #   User: account identifier (NOT the CST token)
            #   Password: CST-{cst}|XST-{x_security_token}
            if self.account_id:
                client.connectionDetails.setUser(self.account_id)
            else:
                # Fallback: some older IG implementations accept CST as user
                logger.warning("[LS] No account_id available — using CST as user (may fail)")
                client.connectionDetails.setUser(self.cst)

            client.connectionDetails.setPassword(
                "CST-" + self.cst + "|XST-" + self.x_security_token
            )

            # Add connection listener for reconnection handling
            client.addListener(self._create_connection_listener())
            client.connect()

            # Wait up to 10s for connection
            for _ in range(20):
                status = client.getStatus()
                if "CONNECTED" in status:
                    self._connected = True
                    self._reconnect_attempts = 0
                    break
                time.sleep(0.5)

            if self._connected:
                self._client = client
                logger.info(f"[LS] Connected to {self.ls_endpoint} "
                            f"(account: {self.account_id}, status: {client.getStatus()})")
            else:
                logger.error(f"[LS] Failed to connect — status: {client.getStatus()}")

        except ImportError:
            logger.error("[LS] lightstreamer-client-lib not installed. "
                         "Run: pip install lightstreamer-client-lib")
        except Exception as e:
            logger.error(f"[LS] Connection error: {e}")

    def _create_connection_listener(self):
        """Create a connection status listener for handling disconnections."""
        provider = self

        class _ConnectionListener:
            def onListenStart(self):
                pass

            def onListenEnd(self):
                pass

            def onStatusChange(self, status):
                logger.debug(f"[LS] Connection status: {status}")
                if "DISCONNECTED" in status and "WILL-RETRY" not in status and "TRYING-RECOVERY" not in status:
                    # Only trigger reconnect for terminal disconnections
                    # TRYING-RECOVERY and WILL-RETRY mean the library is handling it
                    provider._connected = False
                    logger.warning(f"[LS] Disconnected (terminal): {status}")
                    if (provider._reconnect_attempts < provider._max_reconnect_attempts
                            and not provider._reconnecting):
                        provider._reconnecting = True
                        threading.Thread(
                            target=provider._reconnect,
                            daemon=True,
                            name="ls-reconnect"
                        ).start()
                elif "CONNECTED" in status:
                    provider._connected = True
                    provider._reconnect_attempts = 0
                    provider._reconnecting = False

            def onServerError(self, code, message):
                logger.error(f"[LS] Server error {code}: {message}")
                # Code 4 = expired tokens — need to re-auth
                if code == 4:
                    provider._handle_token_expiry()

        return _ConnectionListener()

    def _reconnect(self):
        """Attempt to reconnect after a disconnection."""
        self._reconnect_attempts += 1
        wait_time = min(30, 2 ** self._reconnect_attempts)
        logger.info(f"[LS] Reconnecting in {wait_time}s "
                    f"(attempt {self._reconnect_attempts}/{self._max_reconnect_attempts})")
        time.sleep(wait_time)

        # Refresh tokens from IG client if available
        if self.ig_client:
            try:
                self.ig_client.login()
                self.cst = self.ig_client.cst
                self.x_security_token = self.ig_client.x_security_token
                self.account_id = getattr(self.ig_client, 'account_id', self.account_id)
                logger.info("[LS] Refreshed auth tokens for reconnection")
            except Exception as e:
                logger.error(f"[LS] Failed to refresh tokens: {e}")
                return

        # Disconnect old client cleanly to prevent ghost subscriptions
        if self._client:
            try:
                self._client.disconnect()
                logger.info("[LS] Old client disconnected")
            except Exception as e:
                logger.debug(f"[LS] Old client disconnect error (non-fatal): {e}")

        # Reconnect
        self._connected = False
        self._client = None
        self._subscriptions.clear()
        self._connect()

        # Re-subscribe all active subscriptions
        if self._connected:
            old_listeners = dict(self._listeners)
            self._listeners.clear()
            for sub_key, listener in old_listeners.items():
                # Only re-subscribe CHART streams (format: "epic:timeframe")
                # Skip PRICE:, TRADE, ACCOUNT which have their own subscribe methods
                if sub_key.startswith("PRICE:") or sub_key in ("TRADE", "ACCOUNT"):
                    continue
                parts = sub_key.split(":", 1)
                if len(parts) == 2:
                    epic, timeframe = parts
                    self._subscribe_raw(epic, timeframe, listener)
            logger.info(f"[LS] Reconnected and re-subscribed {len(self._listeners)} streams")

        self._reconnecting = False

    def _handle_token_expiry(self):
        """Handle expired CST/XST tokens by re-authenticating."""
        logger.warning("[LS] Tokens expired — re-authenticating...")
        self._reconnect()

    def _subscribe_raw(self, epic: str, timeframe: str, listener: _BarListener):
        """Internal: create Lightstreamer subscription without seeding."""
        scale = SCALE_MAP.get(timeframe, "1MINUTE")
        item_name = f"CHART:{epic}:{scale}"
        sub_key = f"{epic}:{timeframe}"

        # Prevent duplicate subscriptions
        if sub_key in self._listeners:
            logger.debug(f"[LS] _subscribe_raw skipping duplicate: {sub_key}")
            return

        try:
            sub = self._ls_sub_class(
                mode=CHART_MODE,
                items=[item_name],
                fields=FIELDS
            )
            sub.addListener(listener)
            self._client.subscribe(sub)
            self._listeners[sub_key] = listener
            self._subscriptions.append(sub)
            logger.info(f"[LS] Subscribed: {item_name}")

        except Exception as e:
            logger.error(f"[LS] Subscribe error for {item_name}: {e}")

    def subscribe(self, epic: str, timeframe: str = "5min",
                  max_bars: int = 500, seed_bars: int = 288):
        """
        Subscribe to CHART bars for an epic/timeframe.
        Pre-seeds last 24h of history from TwelveData on startup.

        Args:
            epic:      IG epic e.g. "CS.D.CFEGOLD.CEB.IP"
            timeframe: "1min", "5min", "15min", "60min" — default 5min
            max_bars:  Max bars to keep in memory ring buffer
            seed_bars: Bars to load on startup. Default 288 = 24h at 5min.
        """
        if not self._connected or self._client is None:
            logger.error("[LS] Not connected — cannot subscribe")
            return

        scale = SCALE_MAP.get(timeframe, "1MINUTE")
        item_name = f"CHART:{epic}:{scale}"
        sub_key = f"{epic}:{timeframe}"

        if sub_key in self._listeners:
            logger.info(f"[LS] Already subscribed: {sub_key}")
            return

        listener = _BarListener(epic=epic, max_bars=max_bars)

        # Pre-seed from TwelveData (spot XAU/USD) or IG REST on startup
        if seed_bars > 0:
            seeded = False

            # Try TwelveData first (no rate-limit issues like IG demo)
            twelve_data_key = (
                self.ig_client.twelve_data_key
                if self.ig_client and hasattr(self.ig_client, 'twelve_data_key')
                else None
            )
            if not twelve_data_key:
                # Try to get from config
                try:
                    from core.config import load_settings
                    cfg = load_settings()
                    twelve_data_key = cfg.get('12data', {}).get('api_key', '')
                except Exception:
                    pass

            if twelve_data_key:
                try:
                    from data.multi_data_provider import TwelveDataProvider
                    td = TwelveDataProvider(twelve_data_key)
                    # Map IG epic to TwelveData symbol
                    symbol_map = {
                        "CS.D.CFEGOLD.CEB.IP": "XAU/USD",
                        "CS.D.CFESILVER.CEB.IP": "XAG/USD",
                        "CS.D.EURUSD.CFD.IP": "EUR/USD",
                        "CS.D.GBPUSD.CFD.IP": "GBP/USD",
                        "CS.D.USDJPY.CFD.IP": "USD/JPY",
                        "CMD.USCrude.CFD.IP": "WTI/USD",
                        # NOTE: SPX excluded — requires TwelveData paid plan
                    }
                    symbol = symbol_map.get(epic)
                    if symbol:
                        hist_df = td.get_bars(symbol, timeframe=timeframe, limit=seed_bars)
                        if not hist_df.empty:
                            listener.seed_history(hist_df)
                            logger.info(f"[LS] Seeded {len(hist_df)} bars from TwelveData ({symbol})")
                            seeded = True
                except Exception as e:
                    logger.warning(f"[LS] TwelveData seed failed: {e}")

            # Fallback to IG REST if TwelveData didn't work
            if not seeded and self.ig_client:
                try:
                    from data.ig_price_bars import bars_from_ig
                    resolution = REST_RESOLUTION_MAP.get(timeframe, "MINUTE")
                    raw = self.ig_client.get_prices(epic, resolution=resolution,
                                                    max=min(seed_bars, 1000))
                    hist_df = bars_from_ig(raw, epic)
                    if not hist_df.empty:
                        listener.seed_history(hist_df)
                        logger.info(f"[LS] Seeded {len(hist_df)} bars from IG REST")
                        seeded = True
                except Exception as e:
                    logger.warning(f"[LS] IG REST seed failed: {e}")

        self._subscribe_raw(epic, timeframe, listener)

    def get_bars(self, epic: str, timeframe: str = "5min",
                 limit: int = 288) -> pd.DataFrame:
        """
        Return OHLCV bars: historical (seeded) + live completed bars
        + current forming bar appended as the last row.

        The strategy uses df.iloc[-2] as the signal bar (penultimate),
        so the forming bar at iloc[-1] is ignored by design.
        Requires at least 2 bars to be useful — returns empty if fewer.
        """
        sub_key = f"{epic}:{timeframe}"

        if sub_key not in self._listeners:
            self.subscribe(epic, timeframe)

        listener = self._listeners.get(sub_key)
        if listener is None:
            return pd.DataFrame()

        df = listener.get_dataframe(limit=limit, include_forming=True)

        if len(df) < 2:
            logger.warning(f"[LS] {epic}: only {len(df)} bars — need at least 2")
            return pd.DataFrame()

        logger.info(
            f"[LS] {epic} {timeframe}: {len(df)} bars "
            f"(updates rcvd: {listener.update_count}) | "
            f"latest close: {df['close'].iloc[-1]:.2f}"
        )
        return df

    def is_connected(self) -> bool:
        return self._connected

    def disconnect(self):
        if self._client:
            self._client.disconnect()
            self._connected = False
            logger.info("[LS] Disconnected")

    # ─── PRICE SUBSCRIPTION (replaces deprecated MARKET:) ─────────────────────

    def subscribe_price(self, epic: str, callback=None):
        """
        Subscribe to real-time bid/offer prices for an epic.
        Uses PRICE:{accountId}:{epic} (MERGE mode) — replaces deprecated MARKET: subscription.

        Args:
            epic:     IG epic e.g. "CS.D.CFEGOLD.CEB.IP"
            callback: Optional function(epic, bid, offer, timestamp) called on each update
        """
        if not self._connected or not self.account_id:
            logger.warning("[LS] Cannot subscribe price — not connected or no account_id")
            return

        item_name = f"PRICE:{self.account_id}:{epic}"
        fields = ["BIDPRICE1", "ASKPRICE1", "TIMESTAMP", "MARKET_STATE"]

        class _PriceListener:
            def __init__(self):
                self.bid = None
                self.offer = None
                self.market_state = None
                self.timestamp = None

            def onListenStart(self):
                logger.info(f"[LS] Price subscription started: {epic}")

            def onListenEnd(self):
                pass

            def onSubscriptionError(self, code, message):
                logger.error(f"[LS] Price subscription error {code}: {message}")

            def onItemUpdate(self, update):
                try:
                    bid = update.getValue("BIDPRICE1")
                    offer = update.getValue("ASKPRICE1")
                    state = update.getValue("MARKET_STATE")
                    ts = update.getValue("TIMESTAMP")

                    if bid: self.bid = float(bid)
                    if offer: self.offer = float(offer)
                    if state: self.market_state = state
                    if ts: self.timestamp = ts

                    if callback and self.bid and self.offer:
                        callback(epic, self.bid, self.offer, self.timestamp)
                except Exception as e:
                    logger.error(f"[LS] Price update error: {e}")

        listener = _PriceListener()
        try:
            sub = self._ls_sub_class(mode="MERGE", items=[item_name], fields=fields)
            sub.addListener(listener)
            self._client.subscribe(sub)
            self._subscriptions.append(sub)
            self._listeners[f"PRICE:{epic}"] = listener
            logger.info(f"[LS] Subscribed price: {item_name}")
        except Exception as e:
            logger.error(f"[LS] Price subscribe error: {e}")

        return listener

    def get_live_price(self, epic: str) -> tuple:
        """Get last streamed bid/offer for an epic.

        Returns:
            (bid, offer) or (None, None) if not subscribed.
        """
        listener = self._listeners.get(f"PRICE:{epic}")
        if listener and hasattr(listener, 'bid'):
            return (listener.bid, listener.offer)
        return (None, None)

    # ─── TRADE SUBSCRIPTION (real-time confirmations) ─────────────────────────

    def subscribe_trades(self, callback=None):
        """
        Subscribe to real-time trade confirmations and position updates.
        Uses TRADE:{accountId} in DISTINCT mode.

        Args:
            callback: Optional function(event_type, data) called on each trade event.
                      event_type: "CONFIRMS", "OPU", "WOU"
                      data: dict with parsed fields
        """
        if not self._connected or not self.account_id:
            logger.warning("[LS] Cannot subscribe trades — not connected or no account_id")
            return

        item_name = f"TRADE:{self.account_id}"
        fields = ["CONFIRMS", "OPU", "WOU"]

        class _TradeListener:
            def __init__(self):
                self.last_confirm = None
                self.last_opu = None
                self.last_wou = None

            def onListenStart(self):
                logger.info(f"[LS] Trade subscription started")

            def onListenEnd(self):
                pass

            def onSubscriptionError(self, code, message):
                logger.error(f"[LS] Trade subscription error {code}: {message}")

            def onItemUpdate(self, update):
                try:
                    import json
                    confirms = update.getValue("CONFIRMS")
                    opu = update.getValue("OPU")
                    wou = update.getValue("WOU")

                    if confirms and confirms.strip():
                        try:
                            data = json.loads(confirms)
                            self.last_confirm = data
                            logger.info(f"[LS] TRADE CONFIRM: {data.get('dealId')} "
                                        f"{data.get('direction')} {data.get('epic')} "
                                        f"status={data.get('dealStatus')}")
                            if callback:
                                callback("CONFIRMS", data)
                        except json.JSONDecodeError:
                            pass

                    if opu and opu.strip():
                        try:
                            data = json.loads(opu)
                            self.last_opu = data
                            logger.info(f"[LS] POSITION UPDATE: {data.get('dealId')} "
                                        f"{data.get('direction')} {data.get('epic')} "
                                        f"status={data.get('status')}")
                            if callback:
                                callback("OPU", data)
                        except json.JSONDecodeError:
                            pass

                    if wou and wou.strip():
                        try:
                            data = json.loads(wou)
                            self.last_wou = data
                            logger.info(f"[LS] WORKING ORDER UPDATE: {data.get('dealId')} "
                                        f"{data.get('direction')} {data.get('epic')} "
                                        f"status={data.get('status')}")
                            if callback:
                                callback("WOU", data)
                        except json.JSONDecodeError:
                            pass

                except Exception as e:
                    logger.error(f"[LS] Trade update error: {e}")

        listener = _TradeListener()
        try:
            sub = self._ls_sub_class(mode="DISTINCT", items=[item_name], fields=fields)
            sub.addListener(listener)
            self._client.subscribe(sub)
            self._subscriptions.append(sub)
            self._listeners["TRADE"] = listener
            logger.info(f"[LS] Subscribed trades: {item_name}")
        except Exception as e:
            logger.error(f"[LS] Trade subscribe error: {e}")

        return listener

    # ─── ACCOUNT SUBSCRIPTION (real-time equity/margin) ───────────────────────

    def subscribe_account(self, callback=None):
        """
        Subscribe to real-time account balance/equity updates.
        Uses ACCOUNT:{accountId} in MERGE mode.

        Args:
            callback: Optional function(data) called on each account update.
        """
        if not self._connected or not self.account_id:
            logger.warning("[LS] Cannot subscribe account — not connected or no account_id")
            return

        item_name = f"ACCOUNT:{self.account_id}"
        fields = ["PNL", "DEPOSIT", "AVAILABLE_CASH", "FUNDS", "MARGIN",
                  "AVAILABLE_TO_DEAL", "EQUITY", "EQUITY_USED"]

        class _AccountListener:
            def __init__(self):
                self.equity = None
                self.pnl = None
                self.available_cash = None
                self.margin = None
                self.available_to_deal = None

            def onListenStart(self):
                logger.info(f"[LS] Account subscription started")

            def onListenEnd(self):
                pass

            def onSubscriptionError(self, code, message):
                logger.error(f"[LS] Account subscription error {code}: {message}")

            def onItemUpdate(self, update):
                try:
                    def val(field):
                        v = update.getValue(field)
                        return float(v) if v and v.strip() else None

                    pnl = val("PNL")
                    equity = val("EQUITY")
                    available = val("AVAILABLE_TO_DEAL")
                    margin = val("MARGIN")
                    cash = val("AVAILABLE_CASH")

                    if pnl is not None: self.pnl = pnl
                    if equity is not None: self.equity = equity
                    if available is not None: self.available_to_deal = available
                    if margin is not None: self.margin = margin
                    if cash is not None: self.available_cash = cash

                    if callback:
                        callback({
                            "pnl": self.pnl,
                            "equity": self.equity,
                            "available_to_deal": self.available_to_deal,
                            "margin": self.margin,
                            "available_cash": self.available_cash,
                        })
                except Exception as e:
                    logger.error(f"[LS] Account update error: {e}")

        listener = _AccountListener()
        try:
            sub = self._ls_sub_class(mode="MERGE", items=[item_name], fields=fields)
            sub.addListener(listener)
            self._client.subscribe(sub)
            self._subscriptions.append(sub)
            self._listeners["ACCOUNT"] = listener
            logger.info(f"[LS] Subscribed account: {item_name}")
        except Exception as e:
            logger.error(f"[LS] Account subscribe error: {e}")

        return listener

    def get_live_equity(self) -> float | None:
        """Get last streamed equity value. Returns None if not subscribed."""
        listener = self._listeners.get("ACCOUNT")
        if listener and hasattr(listener, 'equity'):
            return listener.equity
        return None

    def get_live_pnl(self) -> float | None:
        """Get last streamed P&L value. Returns None if not subscribed."""
        listener = self._listeners.get("ACCOUNT")
        if listener and hasattr(listener, 'pnl'):
            return listener.pnl
        return None
