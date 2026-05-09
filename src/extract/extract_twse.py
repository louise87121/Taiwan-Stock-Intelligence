from __future__ import annotations

from datetime import date

import pandas as pd
from pandas.errors import EmptyDataError

from src.api.twse_client import TWSEClient, TWSEClientError
from src.config import RAW_DIR
from src.load.local_storage import write_raw
from src.utils.logging import get_logger


logger = get_logger(__name__)

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
