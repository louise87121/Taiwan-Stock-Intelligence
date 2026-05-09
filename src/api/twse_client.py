from __future__ import annotations

from typing import Any
from io import StringIO

import pandas as pd
import requests
from requests.exceptions import JSONDecodeError

from src.utils.logging import get_logger


class TWSEClientError(RuntimeError):
    pass


class TWSEClient:
    BASE_URL = "https://openapi.twse.com.tw/v1"

    def __init__(self, base_url: str = BASE_URL, timeout: int = 30, session: requests.Session | None = None) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = session or requests.Session()
        self.session.headers.update(
            {
                "User-Agent": "Mozilla/5.0 TaiwanStockIntelligence/1.0",
                "Accept": "application/json,text/plain,*/*",
            }
        )
        self.logger = get_logger(self.__class__.__name__)

    def url_for(self, endpoint: str) -> str:
        return f"{self.base_url}/{endpoint.lstrip('/')}"

    def fetch(self, endpoint: str, **params: Any) -> pd.DataFrame:
        url = self.url_for(endpoint)
        query = {key: value for key, value in params.items() if value is not None}
        self.logger.info("Requesting TWSE OpenAPI url=%s params=%s", url, query)
        try:
            response = self.session.get(url, params=query, timeout=self.timeout)
            self.logger.info("TWSE status=%s endpoint=%s", response.status_code, endpoint)
            response.raise_for_status()
        except requests.RequestException as exc:
            raise TWSEClientError(f"TWSE request failed for {endpoint}: {exc}") from exc

        try:
            payload = response.json()
        except JSONDecodeError as exc:
            raise TWSEClientError(f"TWSE response is not JSON for {endpoint}") from exc
        if isinstance(payload, list):
            return pd.DataFrame(payload)
        if isinstance(payload, dict):
            for key in ("data", "fields"):
                if key in payload and isinstance(payload[key], list):
                    return pd.DataFrame(payload[key])
        raise TWSEClientError(f"Unexpected TWSE response shape for {endpoint}")

    def fetch_csv_url(self, url: str) -> pd.DataFrame:
        self.logger.info("Requesting TWSE CSV url=%s", url)
        try:
            response = self.session.get(url, timeout=self.timeout)
            self.logger.info("TWSE CSV status=%s url=%s", response.status_code, url)
            response.raise_for_status()
        except requests.RequestException as exc:
            raise TWSEClientError(f"TWSE CSV request failed for {url}: {exc}") from exc
        text = response.content.decode("utf-8-sig", errors="replace")
        if not text or "<html" in text.lower() or "PAGE CANNOT BE ACCESSED" in text:
            raise TWSEClientError(f"TWSE CSV response is not tabular data for {url}")
        return pd.read_csv(StringIO(text))
