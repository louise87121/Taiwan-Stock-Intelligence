from __future__ import annotations

import pandas as pd

from src.config import StockConfig

SILVER_COMPANY_COLUMNS = ["stock_id", "stock_name", "industry_group", "market_type", "focus_flag"]


def build_silver_company(stocks: list[StockConfig]) -> pd.DataFrame:
    df = pd.DataFrame([stock.__dict__ for stock in stocks])
    if df.empty:
        return pd.DataFrame(columns=SILVER_COMPANY_COLUMNS)
    df["stock_id"] = df["stock_id"].astype(str)
    df["industry_group"] = df["industry_group"].astype(str)
    df["market_type"] = df["market_type"].astype(str)
    return df[SILVER_COMPANY_COLUMNS]


def transform_twse_company_info(raw_df: pd.DataFrame, fallback_stocks: list[StockConfig]) -> pd.DataFrame:
    fallback = build_silver_company(fallback_stocks)
    if raw_df.empty:
        return fallback
    df = raw_df.copy().rename(
        columns={
            "公司代號": "stock_id",
            "公司名稱": "stock_name",
            "產業別": "industry_group",
            "上市別": "market_type",
        }
    )
    for column in SILVER_COMPANY_COLUMNS:
        if column not in df.columns:
            df[column] = pd.NA
    df["stock_id"] = df["stock_id"].astype(str)
    df["industry_group"] = df["industry_group"].astype(str)
    df["market_type"] = df["market_type"].astype(str)
    df["focus_flag"] = False
    df = df[SILVER_COMPANY_COLUMNS].drop_duplicates("stock_id", keep="last")
    configured_flags = fallback[["stock_id", "focus_flag"]].drop_duplicates("stock_id")
    df = df.drop(columns=["focus_flag"]).merge(configured_flags, on="stock_id", how="left")
    df["focus_flag"] = df["focus_flag"].fillna(False)
    missing = fallback[~fallback["stock_id"].isin(df["stock_id"])]
    result = pd.concat([df, missing], ignore_index=True).drop_duplicates("stock_id", keep="first")
    result["industry_group"] = result["industry_group"].astype(str)
    result["market_type"] = result["market_type"].astype(str)
    return result
