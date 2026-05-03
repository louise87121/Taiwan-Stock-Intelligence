from __future__ import annotations

import pandas as pd

from src.config import StockConfig

SILVER_COMPANY_COLUMNS = ["stock_id", "stock_name", "industry_group", "market_type", "focus_flag"]


def build_silver_company(stocks: list[StockConfig]) -> pd.DataFrame:
    df = pd.DataFrame([stock.__dict__ for stock in stocks])
    if df.empty:
        return pd.DataFrame(columns=SILVER_COMPANY_COLUMNS)
    df["stock_id"] = df["stock_id"].astype(str)
    return df[SILVER_COMPANY_COLUMNS]

