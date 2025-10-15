"""
Multi-Source Data Provider with Standardized Logging
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Dict, List
import time

# STANDARDIZED LOGGING
from core.logging_utils import setup_logging, log_success, log_error, log_warning, safe_log
import logging
logger = logging.getLogger(__name__)


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
            response = requests.get(self.base_url, params=params, timeout=30)
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
            response = requests.get(f"{self.base_url}/time_series", params=params, timeout=30)
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

            response = self.session.get(url, params=params, timeout=30)
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

    def get_bars(self, ig_epic: str, timeframe: str = "5min", limit: int = 100) -> pd.DataFrame:
        self.fetch_stats["total_requests"] += 1

        safe_log(logger, 'info', "=" * 80)
        safe_log(logger, 'info', f"DATA FETCH REQUEST #{self.fetch_stats['total_requests']}")
        safe_log(logger, 'info', f"Epic: {ig_epic} | Timeframe: {timeframe} | Limit: {limit}")
        safe_log(logger, 'info', "=" * 80)

        cache_key = self._get_cache_key(ig_epic, timeframe)

        # Check cache
        if self._is_cached(cache_key):
            self.fetch_stats["cache_hits"] += 1
            cached_data = self.cache[cache_key]
            age = time.time() - cached_data["timestamp"]

            safe_log(logger, 'info', f"[CACHE HIT] Data age: {age:.1f}s | Source: {cached_data['source']}")
            safe_log(logger, 'info', f"[CACHE] Bars: {len(cached_data['data'])} | Latest: {cached_data['data']['close'].iloc[-1]:.2f}")

            return cached_data["data"]

        safe_log(logger, 'info', "[CACHE MISS] Fetching fresh data...")

        # Get mappings
        mappings = self.symbol_map.get(ig_epic, {})
        if not mappings:
            log_warning(logger, f"No symbol mappings found for {ig_epic}")

        # Try each provider
        for provider_name, provider in self.providers:
            if provider_name not in mappings:
                continue

            symbol = mappings[provider_name]
            safe_log(logger, 'info', f"[ATTEMPT] {provider_name} with symbol: {symbol}")

            try:
                df = provider.get_bars(symbol, timeframe, limit)

                if not df.empty and len(df) >= 10:
                    self.fetch_stats["successful_fetches"] += 1
                    self.fetch_stats["provider_usage"][provider_name] = \
                        self.fetch_stats["provider_usage"].get(provider_name, 0) + 1

                    self.cache[cache_key] = {
                        "data": df,
                        "timestamp": time.time(),
                        "source": provider_name
                    }

                    safe_log(logger, 'info', "=" * 80)
                    log_success(logger, f"Data fetched from {provider_name}: {len(df)} bars")
                    safe_log(logger, 'info', f"Latest: O={df['open'].iloc[-1]:.2f} H={df['high'].iloc[-1]:.2f} "
                           f"L={df['low'].iloc[-1]:.2f} C={df['close'].iloc[-1]:.2f}")
                    safe_log(logger, 'info', "=" * 80)

                    return df
                else:
                    log_warning(logger, f"{provider_name} returned insufficient data ({len(df)} bars)")

            except Exception as e:
                log_error(logger, f"{provider_name} exception: {e}")
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
                    log_success(logger, f"Data fetched from IG API: {len(df)} bars")
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
