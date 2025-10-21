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
        self.h = {
            "X-IG-API-KEY": self.api_key,
            "CST": r.headers.get("CST"),
            "X-SECURITY-TOKEN": r.headers.get("X-SECURITY-TOKEN"),
            "Accept": "application/json; charset=UTF-8",
            "Content-Type": "application/json; charset=UTF-8",
        }
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

    def get_prices(self, epic: str, resolution="MINUTE", max=200):
        r = self.s.get(
            f"{self.base}/prices/{epic}",
            params={"resolution": resolution, "max": max},
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