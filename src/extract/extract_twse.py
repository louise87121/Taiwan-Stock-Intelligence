from __future__ import annotations

from datetime import date
from time import sleep

import pandas as pd
from pandas.errors import EmptyDataError
import requests

from src.api.twse_client import TWSEClient, TWSEClientError
from src.config import RAW_DIR
from src.load.local_storage import write_raw
from src.utils.logging import get_logger


logger = get_logger(__name__)

TWSE_STOCK_DAY_URL = "https://www.twse.com.tw/exchangeReport/STOCK_DAY"

TWSE_ENDPOINTS = {
    "TWSECompanyInfo": "opendata/t187ap03_L",
    "TWSEStockDayAll": "exchangeReport/STOCK_DAY_ALL",
    "TWSEMonthlyRevenue": "opendata/t187ap05_L",
    "TWSEIncomeStatement": [
        "https://mopsfin.twse.com.tw/opendata/t187ap06_L_ci.csv",
        "https://mopsfin.twse.com.tw/opendata/t187ap06_L_basi.csv",
        "https://mopsfin.twse.com.tw/opendata/t187ap06_L_bd.csv",
        "https://mopsfin.twse.com.tw/opendata/t187ap06_L_ins.csv",
        "https://mopsfin.twse.com.tw/opendata/t187ap06_L_fh.csv",
        "https://mopsfin.twse.com.tw/opendata/t187ap06_L_mim.csv",
    ],
    "TWSEBalanceSheet": [
        "https://mopsfin.twse.com.tw/opendata/t187ap07_L_ci.csv",
        "https://mopsfin.twse.com.tw/opendata/t187ap07_L_basi.csv",
        "https://mopsfin.twse.com.tw/opendata/t187ap07_L_bd.csv",
        "https://mopsfin.twse.com.tw/opendata/t187ap07_L_ins.csv",
        "https://mopsfin.twse.com.tw/opendata/t187ap07_L_fh.csv",
        "https://mopsfin.twse.com.tw/opendata/t187ap07_L_mim.csv",
    ],
    "TWSECashFlowStatement": [
        "https://mopsfin.twse.com.tw/opendata/t187ap08_L_ci.csv",
        "https://mopsfin.twse.com.tw/opendata/t187ap08_L_basi.csv",
        "https://mopsfin.twse.com.tw/opendata/t187ap08_L_bd.csv",
        "https://mopsfin.twse.com.tw/opendata/t187ap08_L_ins.csv",
        "https://mopsfin.twse.com.tw/opendata/t187ap08_L_fh.csv",
        "https://mopsfin.twse.com.tw/opendata/t187ap08_L_mim.csv",
    ],
}


def extract_twse_dataset(client: TWSEClient, dataset: str) -> pd.DataFrame:
    endpoint = TWSE_ENDPOINTS[dataset]
    if isinstance(endpoint, list):
        frames = []
        for url in endpoint:
            try:
                frame = client.fetch_csv_url(url)
                if not frame.empty:
                    frame["source_url"] = url
                    frames.append(frame)
            except TWSEClientError as exc:
                logger.error("TWSE CSV extraction failed dataset=%s url=%s error=%s", dataset, url, exc)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    try:
        return client.fetch(endpoint)
    except TWSEClientError as exc:
        logger.error("TWSE extraction failed dataset=%s endpoint=%s error=%s", dataset, endpoint, exc)
        return pd.DataFrame()


def extract_all_twse(client: TWSEClient | None = None) -> None:
    client = client or TWSEClient()
    for dataset in TWSE_ENDPOINTS:
        df = extract_twse_dataset(client, dataset)
        if dataset == "TWSEStockDayAll" and not df.empty and "snapshot_date" not in df.columns:
            df["snapshot_date"] = date.today().isoformat()
            existing_path = RAW_DIR / f"{dataset}.csv"
            if existing_path.exists():
                try:
                    existing = pd.read_csv(existing_path)
                    if not existing.empty:
                        df = pd.concat([existing, df], ignore_index=True).drop_duplicates(keep="last")
                except EmptyDataError:
                    logger.warning("Existing raw file is empty and will be replaced: %s", existing_path)
        write_raw(df, dataset, None, RAW_DIR)


def _month_starts(start_date: str, end_date: str) -> list[pd.Timestamp]:
    start = pd.Timestamp(start_date).replace(day=1)
    end = pd.Timestamp(end_date).replace(day=1)
    twse_min = pd.Timestamp("2010-01-01")
    if start < twse_min:
        start = twse_min
    return list(pd.date_range(start=start, end=end, freq="MS"))


def extract_twse_stock_day_history(
    stock_ids: list[str],
    start_date: str,
    end_date: str | None = None,
    sleep_seconds: float = 0.05,
) -> None:
    end_date = end_date or date.today().isoformat()
    months = _month_starts(start_date, end_date)
    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0 TaiwanStockIntelligence/1.0"})

    for stock_id in stock_ids:
        existing_path = RAW_DIR / f"TWSEStockDayHistory_{stock_id}.csv"
        existing = pd.DataFrame()
        fetched_months: set[str] = set()
        if existing_path.exists():
            try:
                existing = pd.read_csv(existing_path)
                if "query_month" in existing.columns:
                    fetched_months = set(existing["query_month"].dropna().astype(str))
            except EmptyDataError:
                existing = pd.DataFrame()

        frames = [existing] if not existing.empty else []
        for month in months:
            query_month = month.strftime("%Y-%m")
            if query_month in fetched_months:
                continue
            params = {"response": "json", "date": month.strftime("%Y%m%d"), "stockNo": stock_id}
            try:
                response = session.get(TWSE_STOCK_DAY_URL, params=params, timeout=30)
                response.raise_for_status()
                payload = response.json()
            except (requests.RequestException, ValueError) as exc:
                logger.warning("TWSE STOCK_DAY history failed stock_id=%s month=%s error=%s", stock_id, query_month, exc)
                continue

            rows = payload.get("data") or []
            fields = payload.get("fields") or []
            if payload.get("stat") != "OK" or not rows or not fields:
                logger.info("No TWSE STOCK_DAY history stock_id=%s month=%s stat=%s", stock_id, query_month, payload.get("stat"))
                continue
            frame = pd.DataFrame(rows, columns=fields)
            frame["stock_id"] = stock_id
            frame["query_month"] = query_month
            frames.append(frame)
            sleep(sleep_seconds)

        combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        if not combined.empty:
            combined = combined.drop_duplicates(["stock_id", "日期"], keep="last")
        write_raw(combined, "TWSEStockDayHistory", stock_id, RAW_DIR)
