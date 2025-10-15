"""
Multi-Source Data Provider for Commodities and Indices
Aggregates data from multiple free/affordable APIs to overcome IG limitations
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Dict, List
import time
import logging

logger = logging.getLogger(__name__)


class AlphaVantageProvider:
    """
    Free API for stocks, forex, commodities, indices
    Limit: 25 requests/day (free), 500/day (premium $50/month)
    """

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://www.alphavantage.co/query"

    def get_bars(self, symbol: str, timeframe: str = "1min", limit: int = 100) -> pd.DataFrame:
        """
        Fetch intraday bars
        Symbols: GOLD (XAU/USD), SPX (S&P 500), etc.
        """
        interval_map = {
            "1min": "1min",
            "3min": "5min",  # Closest available
            "5min": "5min",
            "15min": "15min",
            "60min": "60min"
        }

        interval = interval_map.get(timeframe, "5min")

        params = {
            "function": "TIME_SERIES_INTRADAY",
            "symbol": symbol,
            "interval": interval,
            "apikey": self.api_key,
            "outputsize": "full"
        }

        try:
            response = requests.get(self.base_url, params=params, timeout=30)
            data = response.json()

            if "Time Series" not in str(data):
                logger.warning(f"AlphaVantage API limit or error: {data}")
                return pd.DataFrame()

            # Parse response
            ts_key = f"Time Series ({interval})"
            if ts_key not in data:
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

            df = pd.DataFrame(bars)
            return df.set_index('ts').sort_index()

        except Exception as e:
            logger.error(f"AlphaVantage error: {e}")
            return pd.DataFrame()


class TwelveDataProvider:
    """
    Free API: 800 requests/day, real-time data
    Premium: $49/month for 8000 requests/day
    Best for: Commodities, indices, forex
    """

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.twelvedata.com"

    def get_bars(self, symbol: str, timeframe: str = "1min", limit: int = 100) -> pd.DataFrame:
        """
        Symbols: XAU/USD (Gold), SPX (S&P 500), DJI (Dow Jones), etc.
        """
        interval_map = {
            "1min": "1min",
            "3min": "3min",
            "5min": "5min",
            "15min": "15min",
            "60min": "1h"
        }

        params = {
            "symbol": symbol,
            "interval": interval_map.get(timeframe, "5min"),
            "outputsize": limit,
            "apikey": self.api_key
        }

        try:
            response = requests.get(f"{self.base_url}/time_series", params=params, timeout=30)
            data = response.json()

            if data.get("status") == "error":
                logger.warning(f"TwelveData error: {data.get('message')}")
                return pd.DataFrame()

            if "values" not in data:
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

            df = pd.DataFrame(bars)
            return df.set_index('ts').sort_index()

        except Exception as e:
            logger.error(f"TwelveData error: {e}")
            return pd.DataFrame()


class FinancialModelingPrepProvider:
    """
    Free: 250 requests/day
    Premium: $14/month for 500 requests/day
    Great for: Major indices, stocks, commodities
    """

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://financialmodelingprep.com/api/v3"

    def get_bars(self, symbol: str, timeframe: str = "5min", limit: int = 100) -> pd.DataFrame:
        """
        Symbols: ^GSPC (S&P 500), ^DJI (Dow Jones), GC=F (Gold Futures)
        """
        interval_map = {
            "1min": "1min",
            "5min": "5min",
            "15min": "15min",
            "60min": "1hour"
        }

        interval = interval_map.get(timeframe, "5min")

        try:
            url = f"{self.base_url}/historical-chart/{interval}/{symbol}"
            params = {"apikey": self.api_key}

            response = requests.get(url, params=params, timeout=30)
            data = response.json()

            if not data or isinstance(data, dict):
                logger.warning(f"FMP: No data or error for {symbol}")
                return pd.DataFrame()

            bars = []
            for bar in data[:limit]:
                bars.append({
                    'ts': pd.to_datetime(bar['date']),
                    'open': float(bar['open']),
                    'high': float(bar['high']),
                    'low': float(bar['low']),
                    'close': float(bar['close']),
                    'volume': int(bar.get('volume', 0))
                })

            df = pd.DataFrame(bars)
            return df.set_index('ts').sort_index()

        except Exception as e:
            logger.error(f"FMP error: {e}")
            return pd.DataFrame()


class YahooFinanceProvider:
    """
    100% FREE! No API key needed
    Excellent for: All major indices, commodities
    Limitations: 15-minute delay on some data, rate limits if excessive
    """

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })

    def get_bars(self, symbol: str, timeframe: str = "5m", limit: int = 100) -> pd.DataFrame:
        """
        Symbols:
        - GC=F (Gold Futures)
        - ^GSPC or SPY (S&P 500)
        - ^DJI (Dow Jones)
        - ^FTSE (FTSE 100)
        - CL=F (Crude Oil)
        - SI=F (Silver)
        """
        interval_map = {
            "1min": "1m",
            "3min": "5m",
            "5min": "5m",
            "15min": "15m",
            "60min": "1h"
        }

        interval = interval_map.get(timeframe, "5m")

        # Calculate time range
        period_map = {
            "1m": "1d",
            "5m": "5d",
            "15m": "60d",
            "1h": "60d"
        }
        period = period_map.get(interval, "5d")

        try:
            url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
            params = {
                "interval": interval,
                "range": period
            }

            response = self.session.get(url, params=params, timeout=30)
            data = response.json()

            if data.get("chart", {}).get("error"):
                logger.warning(f"Yahoo Finance error: {data['chart']['error']}")
                return pd.DataFrame()

            result = data["chart"]["result"][0]
            timestamps = result["timestamp"]
            quotes = result["indicators"]["quote"][0]

            bars = []
            for i in range(len(timestamps)):
                # Skip if any value is None
                if None in [quotes["open"][i], quotes["high"][i], quotes["low"][i], quotes["close"][i]]:
                    continue

                bars.append({
                    'ts': pd.to_datetime(timestamps[i], unit='s'),
                    'open': float(quotes["open"][i]),
                    'high': float(quotes["high"][i]),
                    'low': float(quotes["low"][i]),
                    'close': float(quotes["close"][i]),
                    'volume': int(quotes.get("volume", [0])[i] or 0)
                })

            df = pd.DataFrame(bars)
            if df.empty:
                return df

            df = df.set_index('ts').sort_index()

            # Return only requested limit
            return df.tail(limit)

        except Exception as e:
            logger.error(f"Yahoo Finance error: {e}")
            return pd.DataFrame()


class SmartDataAggregator:
    """
    Intelligent data aggregator that:
    1. Tries multiple sources in order of reliability
    2. Caches results to minimize API calls
    3. Falls back to IG if all else fails
    """

    def __init__(self, config: Dict, ig_client=None):
        self.ig_client = ig_client
        self.cache = {}
        self.cache_duration = 60  # Cache for 60 seconds

        # Initialize providers based on config
        self.providers = []


        # Yahoo Finance (FREE - always available)
        self.providers.append(("YahooFinance", YahooFinanceProvider()))
        logger.info("✓ Yahoo Finance provider initialized (FREE)")

        # TwelveData (if key provided)
        if config.get("twelve_data_key"):
            self.providers.append(("TwelveData", TwelveDataProvider(config["twelve_data_key"])))
            logger.info("✓ TwelveData provider initialized")

        # AlphaVantage (if key provided)
        if config.get("alpha_vantage_key"):
            self.providers.append(("AlphaVantage", AlphaVantageProvider(config["alpha_vantage_key"])))
            logger.info("✓ AlphaVantage provider initialized")

        # FMP (if key provided)
        if config.get("fmp_key"):
            self.providers.append(("FMP", FinancialModelingPrepProvider(config["fmp_key"])))
            logger.info("✓ FMP provider initialized")

        # Symbol mappings for different providers
        self.symbol_map = self._build_symbol_map()

    def _build_symbol_map(self) -> Dict:
        """Map IG EPICs to various provider symbols"""
        return {
            # Gold
            "CS.D.CFEGOLD.CEB.IP": {
                "YahooFinance": "GC=F",
                "TwelveData": "XAU/USD",
                "AlphaVantage": "XAUUSD",
                "FMP": "XAUUSD"
            },
            # S&P 500
            "IX.D.SPTRD.DAILY.IP": {
                "YahooFinance": "^GSPC",
                "TwelveData": "SPX",
                "AlphaVantage": "SPX",
                "FMP": "^GSPC"
            },
            # FTSE 100
            "IX.D.FTSE.CFD.IP": {
                "YahooFinance": "^FTSE",
                "TwelveData": "FTSE",
                "AlphaVantage": "FTSE",
                "FMP": "^FTSE"
            },
            # Dow Jones
            "IX.D.DOW.CFD.IP": {
                "YahooFinance": "^DJI",
                "TwelveData": "DJI",
                "AlphaVantage": "DJI",
                "FMP": "^DJI"
            },
            # Crude Oil
            "CMD.USCrude.CFD.IP": {
                "YahooFinance": "CL=F",
                "TwelveData": "BRENT",
                "AlphaVantage": "BRENT",
                "FMP": "CL=F"
            },
            # Silver
            "CMD.SILVER.IP": {
                "YahooFinance": "SI=F",
                "TwelveData": "XAG/USD",
                "AlphaVantage": "XAGUSD",
                "FMP": "XAGUSD"
            },
            # EUR/USD (forex)
            "CS.D.EURUSD.CFD.IP": {
                "YahooFinance": "EURUSD=X",
                "TwelveData": "EUR/USD",
                "AlphaVantage": "EURUSD",
                "FMP": "EURUSD"
            },
            # GBP/USD
            "CS.D.GBPUSD.CFD.IP": {
                "YahooFinance": "GBPUSD=X",
                "TwelveData": "GBP/USD",
                "AlphaVantage": "GBPUSD",
                "FMP": "GBPUSD"
            }
        }

    def _get_cache_key(self, epic: str, timeframe: str) -> str:
        return f"{epic}:{timeframe}"

    def _is_cached(self, key: str) -> bool:
        if key not in self.cache:
            return False
        cached_time = self.cache[key]["timestamp"]
        return (time.time() - cached_time) < self.cache_duration

    def get_bars(self, ig_epic: str, timeframe: str = "5min", limit: int = 100) -> pd.DataFrame:
        """
        Intelligently fetch bars from best available source
        """
        cache_key = self._get_cache_key(ig_epic, timeframe)

        # Check cache first
        if self._is_cached(cache_key):
            logger.debug(f"Cache hit for {ig_epic}")
            return self.cache[cache_key]["data"]

        # Get symbol mappings for this EPIC
        mappings = self.symbol_map.get(ig_epic, {})

        # Try each provider in order
        for provider_name, provider in self.providers:
            if provider_name not in mappings:
                continue

            symbol = mappings[provider_name]

            try:
                logger.info(f"Trying {provider_name} for {ig_epic} (symbol: {symbol})")
                df = provider.get_bars(symbol, timeframe, limit)

                if not df.empty and len(df) >= 10:
                    logger.info(f"✓ Success! Got {len(df)} bars from {provider_name}")

                    # Cache the result
                    self.cache[cache_key] = {
                        "data": df,
                        "timestamp": time.time(),
                        "source": provider_name
                    }

                    return df
                else:
                    logger.warning(f"✗ {provider_name} returned insufficient data")

            except Exception as e:
                logger.error(f"✗ {provider_name} failed: {e}")
                continue

        # Fallback to IG if all providers fail
        if self.ig_client:
            logger.warning(f"All external providers failed, falling back to IG for {ig_epic}")
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
                    logger.info(f"✓ Got {len(df)} bars from IG (fallback)")
                    return df

            except Exception as e:
                logger.error(f"IG fallback also failed: {e}")

        logger.error(f"All data sources failed for {ig_epic}")
        return pd.DataFrame()

    def get_stats(self) -> Dict:
        """Get usage statistics"""
        sources = {}
        for key, value in self.cache.items():
            source = value["source"]
            sources[source] = sources.get(source, 0) + 1

        return {
            "cached_symbols": len(self.cache),
            "source_usage": sources,
            "providers_active": len(self.providers)
        }


# Convenience function
def create_data_aggregator(ig_client=None, **kwargs) -> SmartDataAggregator:
    """
    Create data aggregator with optional API keys

    Usage:
        aggregator = create_data_aggregator(
            ig_client=ig_client,
            twelve_data_key="your_key",  # Optional
            alpha_vantage_key="your_key", # Optional
            fmp_key="your_key"            # Optional
        )
    """
    config = {
        "twelve_data_key": kwargs.get("twelve_data_key"),
        "alpha_vantage_key": kwargs.get("alpha_vantage_key"),
        "fmp_key": kwargs.get("fmp_key")
    }

    return SmartDataAggregator(config, ig_client)


if __name__ == "__main__":
    # Test the aggregator
    logging.basicConfig(level=logging.INFO)

    print("\n" + "=" * 60)
    print("Testing Multi-Source Data Aggregator")
    print("=" * 60 + "\n")

    # Test with just Yahoo Finance (FREE)
    aggregator = create_data_aggregator()

    # Test various instruments
    test_symbols = [
        ("CS.D.CFEGOLD.CEB.IP", "Gold"),
        ("IX.D.SPTRD.DAILY.IP", "S&P 500"),
        ("CS.D.EURUSD.CFD.IP", "EUR/USD")
    ]

    for epic, name in test_symbols:
        print(f"\nTesting {name} ({epic})...")
        df = aggregator.get_bars(epic, timeframe="5min", limit=50)

        if not df.empty:
            print(f"✓ Success! Got {len(df)} bars")
            print(f"  Latest: {df.index[-1]}")
            print(f"  Close: {df['close'].iloc[-1]:.2f}")
        else:
            print(f"✗ Failed to get data")

    # Show stats
    print("\n" + "=" * 60)
    print("Aggregator Statistics:")
    stats = aggregator.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    print("=" * 60 + "\n")