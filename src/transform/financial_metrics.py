from __future__ import annotations

import pandas as pd

SILVER_FINANCIAL_COLUMNS = ["date", "stock_id", "type", "value", "normalized_metric_name"]
SILVER_PER_DIVIDEND_COLUMNS = ["date", "stock_id", "pe_ratio", "dividend_yield", "price_book_ratio"]

METRIC_NAME_MAP = {
    "EPS": "earnings_per_share",
    "ROE": "return_on_equity",
    "營業利益": "operating_income",
    "本期淨利（淨損）": "net_income",
    "資產總額": "total_assets",
    "負債總額": "total_liabilities",
    "營業活動之淨現金流入（流出）": "operating_cash_flow",
}


def transform_financial_statement(raw_df: pd.DataFrame) -> pd.DataFrame:
    if raw_df.empty:
        return pd.DataFrame(columns=SILVER_FINANCIAL_COLUMNS)
    df = raw_df.copy()
    for column in ["date", "stock_id", "type", "value"]:
        if column not in df.columns:
            df[column] = pd.NA
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date.astype(str)
    df["stock_id"] = df["stock_id"].astype(str)
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df["normalized_metric_name"] = df["type"].map(METRIC_NAME_MAP).fillna(
        df["type"].astype(str).str.lower().str.replace(" ", "_", regex=False)
    )
    return df[SILVER_FINANCIAL_COLUMNS]


def transform_per_dividend(per_df: pd.DataFrame, dividend_df: pd.DataFrame | None = None) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    if not per_df.empty:
        per = per_df.copy()
        rename_map = {
            "PER": "pe_ratio",
            "PBR": "price_book_ratio",
            "dividend_yield": "dividend_yield",
        }
        per = per.rename(columns=rename_map)
        for column in SILVER_PER_DIVIDEND_COLUMNS:
            if column not in per.columns:
                per[column] = pd.NA
        frames.append(per[SILVER_PER_DIVIDEND_COLUMNS])
    if dividend_df is not None and not dividend_df.empty:
        div = dividend_df.copy()
        if "CashEarningsDistribution" in div.columns:
            div["dividend_yield"] = pd.to_numeric(div["CashEarningsDistribution"], errors="coerce")
        for column in SILVER_PER_DIVIDEND_COLUMNS:
            if column not in div.columns:
                div[column] = pd.NA
        frames.append(div[SILVER_PER_DIVIDEND_COLUMNS])
    if not frames:
        return pd.DataFrame(columns=SILVER_PER_DIVIDEND_COLUMNS)
    df = pd.concat(frames, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date.astype(str)
    df["stock_id"] = df["stock_id"].astype(str)
    for column in ["pe_ratio", "dividend_yield", "price_book_ratio"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    return df.sort_values(["stock_id", "date"]).groupby(["stock_id", "date"], as_index=False).last()

