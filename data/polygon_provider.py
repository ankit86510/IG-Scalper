import requests
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional
import time


class PolygonDataProvider:
    """
    Fetch real-time forex data from Polygon.io
    Free tier: 5 API calls per minute
    """

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.polygon.io"
        self.session = requests.Session()

    def _symbol_to_polygon(self, ig_epic: str) -> Optional[str]:
        """
        Convert IG EPIC to Polygon forex symbol
        IG format: CS.D.EURUSD.CFD.IP -> Polygon: C:EURUSD
        """
        mapping = {
            'CS.D.EURUSD.CFD.IP': 'C:EURUSD',
            'CS.D.GBPUSD.CFD.IP': 'C:GBPUSD',
            'CS.D.USDJPY.CFD.IP': 'C:USDJPY',
            'CS.D.GBPJPY.CFD.IP': 'C:GBPJPY',
            'CS.D.AUDUSD.CFD.IP': 'C:AUDUSD',
            'CS.D.USDCAD.CFD.IP': 'C:USDCAD',
            'CS.D.NZDUSD.CFD.IP': 'C:NZDUSD',
            'CS.D.EURGBP.CFD.IP': 'C:EURGBP',
            'CS.D.EURJPY.CFD.IP': 'C:EURJPY',
            'CS.D.USDCHF.CFD.IP': 'C:USDCHF',
        }
        return mapping.get(ig_epic)

    def get_bars(self, ig_epic: str, timeframe: str = "1", limit: int = 200) -> pd.DataFrame:
        """
        Fetch bars from Polygon.io

        Args:
            ig_epic: IG EPIC code (e.g., CS.D.EURUSD.CFD.IP)
            timeframe: "1" (1min), "5" (5min), "15" (15min), "60" (1hour)
            limit: Number of bars to fetch

        Returns:
            DataFrame with columns: ts, open, high, low, close, volume
        """
        polygon_symbol = self._symbol_to_polygon(ig_epic)

        if not polygon_symbol:
            raise ValueError(f"Unknown EPIC: {ig_epic}")

        # Calculate date range
        to_date = datetime.utcnow()
        from_date = to_date - timedelta(days=2)  # Get last 2 days of data

        # Map timeframe
        timespan_map = {
            "1": "minute",
            "5": "minute",
            "15": "minute",
            "60": "hour",
        }

        multiplier_map = {
            "1": 1,
            "5": 5,
            "15": 15,
            "60": 1,
        }

        timespan = timespan_map.get(timeframe, "minute")
        multiplier = multiplier_map.get(timeframe, 1)

        url = f"{self.base_url}/v2/aggs/ticker/{polygon_symbol}/range/{multiplier}/{timespan}/{from_date.strftime('%Y-%m-%d')}/{to_date.strftime('%Y-%m-%d')}"

        params = {
            'apiKey': self.api_key,
            'adjusted': 'true',
            'sort': 'asc',
            'limit': limit
        }

        try:
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            if data.get('status') != 'OK':
                raise Exception(f"Polygon API error: {data.get('status')}")

            results = data.get('results', [])

            if not results:
                return pd.DataFrame()

            # Convert to DataFrame
            bars = []
            for bar in results:
                bars.append({
                    'ts': pd.to_datetime(bar['t'], unit='ms', utc=True),
                    'open': bar['o'],
                    'high': bar['h'],
                    'low': bar['l'],
                    'close': bar['c'],
                    'volume': bar.get('v', 0)
                })

            df = pd.DataFrame(bars)
            df = df.set_index('ts').sort_index()

            # Get only the most recent bars
            if len(df) > limit:
                df = df.tail(limit)

            return df

        except requests.exceptions.RequestException as e:
            print(f"Polygon API request failed: {e}")
            return pd.DataFrame()
        except Exception as e:
            print(f"Error fetching Polygon data: {e}")
            return pd.DataFrame()

    def get_current_price(self, ig_epic: str) -> Optional[float]:
        """
        Get the current bid price for a symbol
        """
        polygon_symbol = self._symbol_to_polygon(ig_epic)

        if not polygon_symbol:
            return None

        url = f"{self.base_url}/v2/last/nbbo/{polygon_symbol}"
        params = {'apiKey': self.api_key}

        try:
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            if data.get('status') != 'OK':
                return None

            # Return bid price (for shorts) or ask (for longs) - using mid for simplicity
            bid = data.get('results', {}).get('P')
            ask = data.get('results', {}).get('p')

            if bid and ask:
                return (bid + ask) / 2
            return bid or ask

        except Exception as e:
            print(f"Error fetching current price: {e}")
            return None


class HybridDataProvider:
    """
    Uses Polygon.io for real-time data, falls back to IG if unavailable
    """

    def __init__(self, polygon_api_key: str, ig_client):
        self.polygon = PolygonDataProvider(polygon_api_key)
        self.ig_client = ig_client
        self.use_polygon = True

    def get_bars(self, ig_epic: str, timeframe: str, limit: int = 200) -> pd.DataFrame:
        """
        Try Polygon first, fallback to IG if fails
        """
        if self.use_polygon:
            try:
                df = self.polygon.get_bars(ig_epic, timeframe, limit)
                if not df.empty:
                    return df
            except Exception as e:
                print(f"Polygon failed, falling back to IG: {e}")
                self.use_polygon = False

        # Fallback to IG
        from data.ig_price_bars import bars_from_ig

        resolution_map = {
            "1": "MINUTE",
            "5": "MINUTE_5",
            "15": "MINUTE_15"
        }
        resolution = resolution_map.get(timeframe, "MINUTE")

        prices = self.ig_client.get_prices(ig_epic, resolution=resolution, max=limit)
        return bars_from_ig(prices)