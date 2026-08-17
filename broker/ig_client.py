from typing import Optional

import requests
from tenacity import retry, wait_random_exponential, stop_after_attempt
import urllib3

# Disable SSL warnings when verification is disabled
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

class IGClient:
    def __init__(self, api_key, username, password, demo=True, verify_ssl=True):
        self.api_key = api_key
        self.username = username
        self.password = password
        self.base = "https://demo-api.ig.com/gateway/deal" if demo else "https://api.ig.com/gateway/deal"
        self.s = requests.Session()
        self.verify_ssl = verify_ssl  # Add SSL verification control
        self.h = {}

    @retry(wait=wait_random_exponential(multiplier=1, max=30), stop=stop_after_attempt(5))
    def login(self):
        hdr = {
            "X-IG-API-KEY": self.api_key,
            "Content-Type": "application/json; charset=UTF-8",
            "Accept": "application/json; charset=UTF-8",
            "Version": "2",
        }
        r = self.s.post(
            f"{self.base}/session",
            json={"identifier": self.username, "password": self.password},
            headers=hdr,
            timeout=20,
            verify=self.verify_ssl  # Use SSL verification setting
        )
        r.raise_for_status()
        body = r.json()
        self.h = {
            "X-IG-API-KEY": self.api_key,
            "CST": r.headers.get("CST"),
            "X-SECURITY-TOKEN": r.headers.get("X-SECURITY-TOKEN"),
            "Accept": "application/json; charset=UTF-8",
            "Content-Type": "application/json; charset=UTF-8",
        }
        # Lightstreamer endpoint and tokens — available after login
        self.ls_endpoint = body.get("lightstreamerEndpoint", "")
        self.cst = r.headers.get("CST", "")
        self.x_security_token = r.headers.get("X-SECURITY-TOKEN", "")
        self.account_id = body.get("currentAccountId", "")
        return True

    def _hv(self, v="1"):
        h = dict(self.h)
        h["Version"] = v
        return h

    def market_details(self, epic: str):
        r = self.s.get(
            f"{self.base}/markets/{epic}",
            headers=self._hv("3"),
            timeout=20,
            verify=self.verify_ssl
        )
        r.raise_for_status(); return r.json()

    def get_prices(self, epic: str, resolution="MINUTE", max=200, from_date=None, to_date=None):
        """Get historical prices for an instrument.

        Args:
            epic: Instrument epic
            resolution: MINUTE, MINUTE_5, MINUTE_15, HOUR, DAY, etc.
            max: Max price points (ignored if from/to specified)
            from_date: Start datetime (yyyy-MM-dd'T'HH:mm:ss)
            to_date: End datetime (yyyy-MM-dd'T'HH:mm:ss)
        """
        params = {"resolution": resolution, "max": max, "pageSize": 0}
        if from_date:
            params["from"] = from_date
            params.pop("max", None)
        if to_date:
            params["to"] = to_date

        r = self.s.get(
            f"{self.base}/prices/{epic}",
            params=params,
            headers=self._hv("3"),
            timeout=25,
            verify=self.verify_ssl
        )
        r.raise_for_status(); return r.json()

    def place_order(self, epic, direction, size, currency_code="USD", expiry="-",
                    stop_distance=None, limit_distance=None, trailing=None,
                    tif="EXECUTE_AND_ELIMINATE"):
        payload = {
            "epic": epic,
            "expiry": expiry,
            "direction": direction,
            "size": size,
            "orderType": "MARKET",
            "timeInForce": tif,
            "guaranteedStop": False,
            "forceOpen": True,
            "currencyCode": currency_code,
        }
        if trailing:
            payload["trailingStop"] = True
            payload["trailingStopIncrement"] = trailing.get("increment")
            payload["stopDistance"] = trailing.get("initial_distance")
        else:
            if stop_distance is not None:
                payload["stopDistance"] = stop_distance
        if limit_distance is not None:
            payload["limitDistance"] = limit_distance

        r = self.s.post(
            f"{self.base}/positions/otc",
            json=payload,
            headers=self._hv("2"),
            timeout=25,
            verify=self.verify_ssl
        )
        r.raise_for_status()
        return r.json()


    def place_working_order(self, epic: str, direction: str, level: float,
                            size: float, stop_distance: float,
                            limit_distance: Optional[float],
                            good_till_date: str, currency_code: str = "USD",
                            expiry: str = "-") -> dict:
        """POST /workingorders/otc — places a STOP working order."""
        payload = {
            "epic": epic,
            "expiry": expiry,
            "direction": direction,
            "size": size,
            "level": level,
            "type": "STOP",
            "timeInForce": "GOOD_TILL_DATE",
            "goodTillDate": good_till_date,
            "currencyCode": currency_code,
            "guaranteedStop": False,
            "forceOpen": True,
            "stopDistance": stop_distance,
        }
        if limit_distance is not None:
            payload["limitDistance"] = limit_distance

        r = self.s.post(
            f"{self.base}/workingorders/otc",
            json=payload,
            headers=self._hv("2"),
            timeout=25,
            verify=self.verify_ssl
        )
        r.raise_for_status()
        return r.json()

    def close_position(self, deal_id, direction, size):
        payload = {"dealId": deal_id, "direction": direction, "size": size, "orderType": "MARKET"}
        r = self.s.post(
            f"{self.base}/positions/otc/close",
            json=payload,
            headers=self._hv("1"),
            timeout=20,
            verify=self.verify_ssl
        )
        r.raise_for_status(); return r.json()

    def confirm_deal(self, deal_reference):
        """Get deal confirmation with actual dealId from a dealReference."""
        r = self.s.get(
            f"{self.base}/confirms/{deal_reference}",
            headers=self._hv("1"),
            timeout=20,
            verify=self.verify_ssl
        )
        r.raise_for_status(); return r.json()

    def update_position(self, deal_id, stop_level=None, limit_level=None):
        """Update stop/limit on an open position using the real dealId."""
        payload = {}
        if stop_level is not None:
            payload["stopLevel"] = stop_level
        if limit_level is not None:
            payload["limitLevel"] = limit_level
        # trailingStop must be false since we manage trailing manually
        payload["trailingStop"] = False

        r = self.s.put(
            f"{self.base}/positions/otc/{deal_id}",
            json=payload,
            headers=self._hv("2"),
            timeout=20,
            verify=self.verify_ssl
        )
        r.raise_for_status(); return r.json()

    def positions(self):
        r = self.s.get(
            f"{self.base}/positions",
            headers=self._hv("2"),
            timeout=20,
            verify=self.verify_ssl
        )
        r.raise_for_status(); return r.json()

    def account_summary(self):
        r = self.s.get(
            f"{self.base}/accounts",
            headers=self._hv("1"),
            timeout=20,
            verify=self.verify_ssl
        )
        r.raise_for_status(); return r.json()

    def get_transactions(self, from_date=None, to_date=None, transaction_type="ALL_DEAL"):
        """Get transaction history. Dates in format: 2026-06-23T00:00:00"""
        params = {"type": transaction_type, "pageSize": 50}
        if from_date:
            params["from"] = from_date
        if to_date:
            params["to"] = to_date

        r = self.s.get(
            f"{self.base}/history/transactions",
            params=params,
            headers=self._hv("2"),
            timeout=20,
            verify=self.verify_ssl
        )
        r.raise_for_status(); return r.json()

    def get_working_orders(self) -> dict:
        """GET /workingorders — retrieves all active working orders."""
        r = self.s.get(
            f"{self.base}/workingorders",
            headers=self._hv("2"),
            timeout=20,
            verify=self.verify_ssl
        )
        r.raise_for_status()
        return r.json()

    def get_activity(self, from_date=None, to_date=None):
        """Get account activity history."""
        params = {"pageSize": 50}
        if from_date:
            params["from"] = from_date
        if to_date:
            params["to"] = to_date

        r = self.s.get(
            f"{self.base}/history/activity",
            params=params,
            headers=self._hv("3"),
            timeout=20,
            verify=self.verify_ssl
        )
        r.raise_for_status(); return r.json()

    def delete_working_order(self, deal_id: str) -> dict:
        """DELETE /workingorders/otc/{dealId} — cancels a working order."""
        r = self.s.delete(
            f"{self.base}/workingorders/otc/{deal_id}",
            headers=self._hv("2"),
            timeout=20,
            verify=self.verify_ssl
        )
        r.raise_for_status()
        return r.json()