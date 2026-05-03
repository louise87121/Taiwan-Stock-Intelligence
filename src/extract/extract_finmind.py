from __future__ import annotations

import pandas as pd

from src.api.finmind_client import FinMindClient, FinMindClientError
from src.config import RAW_DIR, StockConfig
from src.load.local_storage import write_raw
from src.utils.logging import get_logger


logger = get_logger(__name__)

DATASETS_WITH_STOCK_AND_DATE = [
    "TaiwanStockPrice",
    "TaiwanStockMonthRevenue",
    "TaiwanStockFinancialStatements",
    "TaiwanStockBalanceSheet",
    "TaiwanStockCashFlowsStatement",
    "TaiwanStockDividend",
    "TaiwanStockPER",
]


def extract_dataset(
    client: FinMindClient,
    dataset: str,
    stock_id: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    params = {}
    if stock_id:
        params["data_id"] = stock_id
    if start_date:
        params["start_date"] = start_date
    if end_date:
        params["end_date"] = end_date
    try:
        return client.fetch(dataset, **params)
    except FinMindClientError as exc:
        logger.error("Extraction failed for %s stock_id=%s: %s", dataset, stock_id, exc)
        return pd.DataFrame()


def extract_all(
    stocks: list[StockConfig],
    start_date: str,
    end_date: str,
    client: FinMindClient | None = None,
) -> None:
    client = client or FinMindClient()
    stock_info = extract_dataset(client, "TaiwanStockInfo")
    write_raw(stock_info, "TaiwanStockInfo", None, RAW_DIR)

    for stock in stocks:
        for dataset in DATASETS_WITH_STOCK_AND_DATE:
            df = extract_dataset(
                client=client,
                dataset=dataset,
                stock_id=stock.stock_id,
                start_date=start_date,
                end_date=end_date,
            )
            write_raw(df, dataset, stock.stock_id, RAW_DIR)

