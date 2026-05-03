from __future__ import annotations

import os
from typing import Any

import pandas as pd
import requests
from dotenv import load_dotenv

from src.utils.logging import get_logger


class FinMindClientError(RuntimeError):
    pass


class FinMindClient:
    BASE_URL = "https://api.finmindtrade.com/api/v4/data"

    def __init__(
        self,
        token: str | None = None,
        base_url: str = BASE_URL,
        timeout: int = 30,
        session: requests.Session | None = None,
    ) -> None:
        self.token = token if token is not None else self._resolve_token()
        self.base_url = base_url
        self.timeout = timeout
        self.session = session or requests.Session()
        self.logger = get_logger(self.__class__.__name__)

    @staticmethod
    def _resolve_token() -> str | None:
        load_dotenv()
        token = os.getenv("FINMIND_TOKEN")
        if token:
            return token
        try:
            import streamlit as st

            secret_token = st.secrets.get("FINMIND_TOKEN")
            return str(secret_token) if secret_token else None
        except Exception:
            return None

    def build_params(self, dataset: str, **params: Any) -> dict[str, Any]:
        query = {"dataset": dataset}
        query.update({key: value for key, value in params.items() if value is not None})
        if self.token:
            query["token"] = self.token
        return query

    def fetch(self, dataset: str, **params: Any) -> pd.DataFrame:
        query = self.build_params(dataset, **params)
        safe_query = {key: ("***" if key == "token" else value) for key, value in query.items()}
        self.logger.info("Requesting FinMind dataset=%s params=%s", dataset, safe_query)
        try:
            response = self.session.get(self.base_url, params=query, timeout=self.timeout)
            self.logger.info("FinMind status=%s dataset=%s", response.status_code, dataset)
            response.raise_for_status()
        except requests.RequestException as exc:
            raise FinMindClientError(f"FinMind request failed for {dataset}: {exc}") from exc

        payload = response.json()
        if "data" not in payload:
            message = payload.get("msg") or payload.get("message") or "missing data field"
            raise FinMindClientError(f"FinMind response error for {dataset}: {message}")

        data = payload.get("data") or []
        if not data:
            self.logger.warning("FinMind returned empty data for dataset=%s", dataset)
            return pd.DataFrame()
        return pd.DataFrame(data)
