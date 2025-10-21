"""
Multi-Source Data Provider with Standardized Logging
"""
import urllib3
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Dict, List
import time
import pytz

# Disable SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# STANDARDIZED LOGGING
from core.logging_utils import setup_logging, log_success, log_error, log_warning, safe_log
import logging
logger = logging.getLogger(__name__)


# Add this class to your multi_data_provider.py file
class TimezoneAwareLogger:
    """Wrapper to log all timestamps in Europe/Rome timezone"""

    def __init__(self, logger):
        self.logger = logger
        self.tz_rome = pytz.timezone('Europe/Rome')

    def format_time(self, dt):
        """Convert any datetime to Rome timezone"""
        if dt.tzinfo is None:
            dt = pytz.UTC.localize(dt)
        return dt.astimezone(self.tz_rome).strftime('%Y-%m-%d %H:%M:%S %Z')

    def info(self, msg):
        self.logger.info(msg)

    def debug(self, msg):
        self.logger.debug(msg)

    def warning(self, msg):
        self.logger.warning(msg)

    def error(self, msg):
        self.logger.error(msg)


class YahooFinanceProvider:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        self.request_count = 0

    def get_bars(self, symbol: str, timeframe: str = "5m", limit: int = 100) -> pd.DataFrame:
        self.request_count += 1
        safe_log(logger, 'info', f"[YahooFinance] Fetching {symbol} ({timeframe}) - Request #{self.request_count}")

        interval_map = {"1min": "1m", "3min": "5m", "5min": "5m", "15min": "15m", "60min": "1h"}
        interval = interval_map.get(timeframe, "5m")
        period_map = {"1m": "1d", "5m": "5d", "15m": "60d", "1h": "60d"}
        period = period_map.get(interval, "5d")

        try:
            start_time = time.time()
            url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
            params = {"interval": interval, "range": period}

            # ADD verify=False to disable SSL verification
            response = self.session.get(
                url,
                params=params,
                timeout=30,
                verify=False  # ← FIX: Disable SSL verification
            )
            elapsed = time.time() - start_time

            safe_log(logger, 'debug', f"[YahooFinance] Request completed in {elapsed:.2f}s")
            data = response.json()

            if data.get("chart", {}).get("error"):
                log_warning(logger, f"YahooFinance API Error for {symbol}")
                return pd.DataFrame()

            result = data["chart"]["result"][0]
            timestamps = result["timestamp"]
            quotes = result["indicators"]["quote"][0]

            bars = []
            skipped = 0
            for i in range(len(timestamps)):
                if None in [quotes["open"][i], quotes["high"][i], quotes["low"][i], quotes["close"][i]]:
                    skipped += 1
                    continue
                bars.append({
                    'ts': pd.to_datetime(timestamps[i], unit='s'),
                    'open': float(quotes["open"][i]),
                    'high': float(quotes["high"][i]),
                    'low': float(quotes["low"][i]),
                    'close': float(quotes["close"][i]),
                    'volume': int(quotes.get("volume", [0])[i] or 0)
                })

            if skipped > 0:
                safe_log(logger, 'debug', f"[YahooFinance] Skipped {skipped} incomplete bars")

            df = pd.DataFrame(bars)
            if df.empty:
                log_warning(logger, f"YahooFinance: No valid data for {symbol}")
                return df

            df = df.set_index('ts').sort_index().tail(limit)
            log_success(logger, f"YahooFinance: Retrieved {len(df)} bars for {symbol}")
            safe_log(logger, 'debug', f"[YahooFinance] Current price: {df['close'].iloc[-1]:.2f}")

            return df

        except requests.Timeout:
            log_error(logger, f"YahooFinance timeout fetching {symbol}")
            return pd.DataFrame()
        except Exception as e:
            log_error(logger, f"YahooFinance error: {e}")
            return pd.DataFrame()

class TwelveDataProvider:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.twelvedata.com"
        self.request_count = 0

    def get_bars(self, symbol: str, timeframe: str = "1min", limit: int = 100) -> pd.DataFrame:
        self.request_count += 1
        safe_log(logger, 'info', f"[TwelveData] Fetching {symbol} ({timeframe}) - Request #{self.request_count}")

        interval_map = {"1min": "1min", "3min": "3min", "5min": "5min", "15min": "15min", "60min": "1h"}
        params = {
            "symbol": symbol,
            "interval": interval_map.get(timeframe, "5min"),
            "outputsize": limit,
            "apikey": self.api_key
        }

        try:
            start_time = time.time()
            # ADD verify=False to disable SSL verification
            response = requests.get(
                f"{self.base_url}/time_series",
                params=params,
                timeout=30,
                verify=False  # ← FIX: Disable SSL verification
            )
            elapsed = time.time() - start_time

            safe_log(logger, 'debug', f"[TwelveData] Request completed in {elapsed:.2f}s")
            data = response.json()

            if data.get("status") == "error":
                log_warning(logger, f"TwelveData API Error: {data.get('message')}")
                return pd.DataFrame()

            if "values" not in data:
                log_error(logger, f"TwelveData: No data returned for {symbol}")
                return pd.DataFrame()

            bars = []
            for bar in data["values"]:
                bars.append({
                    'ts': pd.to_datetime(bar['datetime']),
                    'open': float(bar['open']),
                    'high': float(bar['high']),
                    'low': float(bar['low']),
                    'close': float(bar['close']),
                    'volume': int(bar.get('volume', 0))
                })

            df = pd.DataFrame(bars).set_index('ts').sort_index()
            log_success(logger, f"TwelveData: Retrieved {len(df)} bars for {symbol}")

            return df

        except requests.Timeout:
            log_error(logger, f"TwelveData timeout fetching {symbol}")
            return pd.DataFrame()
        except Exception as e:
            log_error(logger, f"TwelveData error: {e}")
            return pd.DataFrame()


class AlphaVantageProvider:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://www.alphavantage.co/query"
        self.request_count = 0

    def get_bars(self, symbol: str, timeframe: str = "1min", limit: int = 100) -> pd.DataFrame:
        self.request_count += 1
        safe_log(logger, 'info', f"[AlphaVantage] Fetching {symbol} ({timeframe}) - Request #{self.request_count}")

        interval_map = {"1min": "1min", "3min": "5min", "5min": "5min", "15min": "15min", "60min": "60min"}
        interval = interval_map.get(timeframe, "5min")

        params = {
            "function": "TIME_SERIES_INTRADAY",
            "symbol": symbol,
            "interval": interval,
            "apikey": self.api_key,
            "outputsize": "full"
        }

        try:
            start_time = time.time()
            # ADD verify=False to disable SSL verification
            response = requests.get(
                self.base_url,
                params=params,
                timeout=30,
                verify=False  # ← FIX: Disable SSL verification
            )
            elapsed = time.time() - start_time

            safe_log(logger, 'debug', f"[AlphaVantage] Request completed in {elapsed:.2f}s")
            data = response.json()

            if "Time Series" not in str(data):
                log_warning(logger, f"AlphaVantage API limit or error for {symbol}")
                return pd.DataFrame()

            ts_key = f"Time Series ({interval})"
            if ts_key not in data:
                log_error(logger, f"AlphaVantage unexpected response format for {symbol}")
                return pd.DataFrame()

            bars = []
            for timestamp, values in list(data[ts_key].items())[:limit]:
                bars.append({
                    'ts': pd.to_datetime(timestamp),
                    'open': float(values['1. open']),
                    'high': float(values['2. high']),
                    'low': float(values['3. low']),
                    'close': float(values['4. close']),
                    'volume': int(values['5. volume'])
                })

            df = pd.DataFrame(bars).set_index('ts').sort_index()
            log_success(logger, f"AlphaVantage: Retrieved {len(df)} bars for {symbol}")

            return df

        except requests.Timeout:
            log_error(logger, f"AlphaVantage timeout fetching {symbol}")
            return pd.DataFrame()
        except Exception as e:
            log_error(logger, f"AlphaVantage error fetching {symbol}: {e}")
            return pd.DataFrame()



class SmartDataAggregator:
    def __init__(self, config: Dict, ig_client=None):
        self.ig_client = ig_client
        self.cache = {}
        self.cache_duration = 60
        self.providers = []
        self.fetch_stats = {
            "total_requests": 0,
            "cache_hits": 0,
            "successful_fetches": 0,
            "failed_fetches": 0,
            "provider_usage": {}
        }

        # ADD THIS:
        self.tz_rome = pytz.timezone('Europe/Rome')

        safe_log(logger, 'info', "=" * 80)
        safe_log(logger, 'info', "Initializing Smart Data Aggregator")
        safe_log(logger, 'info', "=" * 80)


        # Yahoo Finance (FREE)
        self.providers.append(("YahooFinance", YahooFinanceProvider()))
        log_success(logger, "Yahoo Finance provider initialized (FREE)")


        # TwelveData
        if config.get("twelve_data_key"):
            self.providers.append(("TwelveData", TwelveDataProvider(config["twelve_data_key"])))
            log_success(logger, "TwelveData provider initialized")

        # AlphaVantage
        if config.get("alpha_vantage_key"):
            self.providers.append(("AlphaVantage", AlphaVantageProvider(config["alpha_vantage_key"])))
            log_success(logger, "AlphaVantage provider initialized")

        self.symbol_map = self._build_symbol_map()
        safe_log(logger, 'info', f"Loaded {len(self.symbol_map)} symbol mappings")
        safe_log(logger, 'info', "=" * 80)

    def _build_symbol_map(self) -> Dict:
        return {
            "CS.D.CFEGOLD.CEB.IP": {
                "YahooFinance": "GC=F",
                "TwelveData": "XAU/USD",
                "AlphaVantage": "XAUUSD"
            },
            "IX.D.SPTRD.DAILY.IP": {
                "YahooFinance": "^GSPC",
                "TwelveData": "SPX",
                "AlphaVantage": "SPX"
            },
            "CS.D.EURUSD.CFD.IP": {
                "YahooFinance": "EURUSD=X",
                "TwelveData": "EUR/USD",
                "AlphaVantage": "EURUSD"
            },
            "CS.D.GBPUSD.CFD.IP": {
                "YahooFinance": "GBPUSD=X",
                "TwelveData": "GBP/USD",
                "AlphaVantage": "GBPUSD"
            }
        }

    def _get_cache_key(self, epic: str, timeframe: str) -> str:
        return f"{epic}:{timeframe}"

    def _is_cached(self, key: str) -> bool:
        if key not in self.cache:
            return False
        return (time.time() - self.cache[key]["timestamp"]) < self.cache_duration

    def _to_rome_time(self, dt):
        """Convert pandas datetime index to Rome timezone string"""
        if dt.tzinfo is None:
            dt = pytz.UTC.localize(dt)
        return dt.astimezone(self.tz_rome).strftime('%Y-%m-%d %H:%M:%S %Z')

    def get_bars(self, ig_epic: str, timeframe: str = "5min", limit: int = 100) -> pd.DataFrame:
        """
        Fetch bars with Rome timezone logging and penultimate bar focus
        """
        self.fetch_stats["total_requests"] += 1

        rome_time = datetime.now(pytz.UTC).astimezone(self.tz_rome)

        safe_log(logger, 'info', "=" * 80)
        safe_log(logger, 'info',
                 f"DATA FETCH #{self.fetch_stats['total_requests']} - {rome_time.strftime('%Y-%m-%d %H:%M:%S %Z')}")
        safe_log(logger, 'info', f"Epic: {ig_epic} | Timeframe: {timeframe} | Limit: {limit}")
        safe_log(logger, 'info', "=" * 80)

        cache_key = self._get_cache_key(ig_epic, timeframe)

        # Check cache
        if self._is_cached(cache_key):
            self.fetch_stats["cache_hits"] += 1
            cached_data = self.cache[cache_key]
            age = time.time() - cached_data["timestamp"]

            safe_log(logger, 'info', f"[CACHE HIT] Age: {age:.1f}s | Source: {cached_data['source']}")

            # Show PENULTIMATE bar (not latest incomplete)
            if len(cached_data['data']) >= 2:
                penultimate = cached_data['data'].iloc[-2]
                penultimate_time = self._to_rome_time(cached_data['data'].index[-2])
                safe_log(logger, 'info',
                         f"[CACHE] Bars: {len(cached_data['data'])} | Penultimate: {penultimate['close']:.2f} @ {penultimate_time}")

            return cached_data["data"]

        safe_log(logger, 'info', "[CACHE MISS] Fetching fresh data...")

        mappings = self.symbol_map.get(ig_epic, {})
        if not mappings:
            log_warning(logger, f"No symbol mappings for {ig_epic}")

        # Try each provider
        for provider_name, provider in self.providers:
            if provider_name not in mappings:
                continue

            symbol = mappings[provider_name]
            safe_log(logger, 'info', f"[ATTEMPT] {provider_name}: {symbol}")

            try:
                df = provider.get_bars(symbol, timeframe, limit)

                if not df.empty and len(df) >= 10:
                    # ====================================
                    # ROME TIMEZONE LOGGING
                    # ====================================
                    safe_log(logger, 'info', "=" * 80)
                    safe_log(logger, 'info', f"[DATA RECEIVED] {provider_name}: {len(df)} bars")

                    start_rome = self._to_rome_time(df.index[0])
                    end_rome = self._to_rome_time(df.index[-1])
                    safe_log(logger, 'info', f"[TIME RANGE] {start_rome} to {end_rome}")
                    safe_log(logger, 'info', "-" * 80)

                    # Log FIRST 3 bars
                    safe_log(logger, 'info', "[FIRST 3 BARS]")
                    for i in range(min(3, len(df))):
                        bar = df.iloc[i]
                        bar_time = self._to_rome_time(df.index[i])
                        safe_log(logger, 'info',
                                 f"  {bar_time}: O={bar['open']:.4f} H={bar['high']:.4f} "
                                 f"L={bar['low']:.4f} C={bar['close']:.4f} V={bar['volume']}")

                    # Log PENULTIMATE 3 bars (NOT latest - it's incomplete!)
                    safe_log(logger, 'info', "[PENULTIMATE 3 BARS] (Latest bar excluded - still forming)")
                    for i in range(max(0, len(df) - 4), len(df) - 1):  # -4 to -2
                        bar = df.iloc[i]
                        bar_time = self._to_rome_time(df.index[i])
                        safe_log(logger, 'info',
                                 f"  {bar_time}: O={bar['open']:.4f} H={bar['high']:.4f} "
                                 f"L={bar['low']:.4f} C={bar['close']:.4f} V={bar['volume']}")

                    safe_log(logger, 'info', "-" * 80)

                    # Statistics on COMPLETE bars only (exclude last)
                    complete_df = df.iloc[:-1]
                    safe_log(logger, 'info', "[PRICE STATS] (Complete bars only)")
                    safe_log(logger, 'info', f"  Close Min: {complete_df['close'].min():.4f}")
                    safe_log(logger, 'info', f"  Close Max: {complete_df['close'].max():.4f}")
                    safe_log(logger, 'info', f"  Close Mean: {complete_df['close'].mean():.4f}")
                    safe_log(logger, 'info', f"  Close Std: {complete_df['close'].std():.4f}")

                    # Check for flat data in COMPLETE bars
                    recent_complete = complete_df.tail(10)
                    flat_count = 0
                    for idx in range(len(recent_complete)):
                        bar = recent_complete.iloc[idx]
                        if bar['open'] == bar['high'] == bar['low'] == bar['close']:
                            flat_count += 1

                    safe_log(logger, 'info', f"  Flat bars (last 10 complete): {flat_count}/10")

                    # Price movement (complete bars)
                    price_range = complete_df['high'].max() - complete_df['low'].min()
                    avg_price = complete_df['close'].mean()
                    if avg_price > 0:
                        movement_pct = (price_range / avg_price) * 100
                        safe_log(logger, 'info', f"  Price movement: {movement_pct:.6f}%")

                    unique_closes = complete_df['close'].nunique()
                    safe_log(logger, 'info', f"  Unique close prices: {unique_closes}/{len(complete_df)}")

                    safe_log(logger, 'info', "=" * 80)

                    # Data quality checks
                    if flat_count > 7:
                        log_warning(logger,
                                    f"{provider_name}: TOO MANY FLAT BARS ({flat_count}/10) - "
                                    f"Market may be closed. Trying next provider...")
                        continue

                    if unique_closes <= 1:
                        log_warning(logger,
                                    f"{provider_name}: ALL PRICES IDENTICAL - "
                                    f"Data is stale. Trying next provider...")
                        continue

                    if avg_price > 0 and movement_pct < 0.001:  # Less than 0.001%
                        log_warning(logger,
                                    f"{provider_name}: INSUFFICIENT MOVEMENT ({movement_pct:.6f}%) - "
                                    f"Market may be closed. Trying next provider...")
                        continue

                    # Data passed quality checks
                    self.fetch_stats["successful_fetches"] += 1
                    self.fetch_stats["provider_usage"][provider_name] = \
                        self.fetch_stats["provider_usage"].get(provider_name, 0) + 1

                    self.cache[cache_key] = {
                        "data": df,
                        "timestamp": time.time(),
                        "source": provider_name
                    }

                    log_success(logger, f"Data accepted from {provider_name}: {len(df)} bars")
                    return df
                else:
                    log_warning(logger, f"{provider_name} returned insufficient data ({len(df)} bars)")

            except Exception as e:
                log_error(logger, f"{provider_name} exception: {e}")
                import traceback
                safe_log(logger, 'debug', traceback.format_exc())
                continue

        # IG Fallback
        self.fetch_stats["failed_fetches"] += 1

        if self.ig_client:
            log_warning(logger, "All external providers failed, trying IG API...")

            try:
                from data.ig_price_bars import bars_from_ig

                resolution_map = {
                    "1min": "MINUTE",
                    "3min": "MINUTE_3",
                    "5min": "MINUTE_5",
                    "15min": "MINUTE_15"
                }

                resolution = resolution_map.get(timeframe, "MINUTE_5")
                prices = self.ig_client.get_prices(ig_epic, resolution=resolution, max=limit)
                df = bars_from_ig(prices)

                if not df.empty:
                    self.fetch_stats["successful_fetches"] += 1
                    log_success(logger, f"Data from IG API: {len(df)} bars")
                    return df

            except Exception as e:
                log_error(logger, f"IG API fallback failed: {e}")

        log_error(logger, f"All data sources exhausted for {ig_epic}")
        return pd.DataFrame()


    def get_stats(self) -> Dict:
        total = max(1, self.fetch_stats['total_requests'])
        return {
            "cached_symbols": len(self.cache),
            "providers_active": len(self.providers),
            "total_requests": self.fetch_stats["total_requests"],
            "cache_hit_rate": f"{(self.fetch_stats['cache_hits'] / total * 100):.1f}%",
            "success_rate": f"{(self.fetch_stats['successful_fetches'] / total * 100):.1f}%",
            "provider_usage": self.fetch_stats["provider_usage"]
        }

    def log_statistics(self):
        stats = self.get_stats()

        safe_log(logger, 'info', "=" * 80)
        safe_log(logger, 'info', "DATA AGGREGATOR STATISTICS")
        safe_log(logger, 'info', "=" * 80)
        safe_log(logger, 'info', f"Total Requests: {stats['total_requests']}")
        safe_log(logger, 'info', f"Cache Hit Rate: {stats['cache_hit_rate']}")
        safe_log(logger, 'info', f"Success Rate: {stats['success_rate']}")
        safe_log(logger, 'info', "Provider Usage:")
        for provider, count in stats['provider_usage'].items():
            safe_log(logger, 'info', f"  {provider}: {count} requests")
        safe_log(logger, 'info', "=" * 80)


def create_data_aggregator(ig_client=None, **kwargs) -> SmartDataAggregator:
    config = {
        "twelve_data_key": kwargs.get("twelve_data_key"),
        "alpha_vantage_key": kwargs.get("alpha_vantage_key")
    }
    return SmartDataAggregator(config, ig_client)
