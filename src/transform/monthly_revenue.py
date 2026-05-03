from __future__ import annotations

import pandas as pd

SILVER_MONTHLY_REVENUE_COLUMNS = ["month", "stock_id", "revenue", "revenue_mom", "revenue_yoy", "revenue_growth_signal"]


def _growth_signal(revenue_mom: float | None, revenue_yoy: float | None) -> str:
    if pd.isna(revenue_mom) and pd.isna(revenue_yoy):
        return "Unknown"
    if revenue_yoy is not None and revenue_yoy > 0.2 and revenue_mom is not None and revenue_mom > 0:
        return "Strong Growth"
    if revenue_yoy is not None and revenue_yoy > 0:
        return "Improving"
    if revenue_yoy is not None and revenue_yoy < 0 and revenue_mom is not None and revenue_mom < 0:
        return "Declining"
    return "Mixed"


def transform_monthly_revenue(raw_df: pd.DataFrame) -> pd.DataFrame:
    if raw_df.empty:
        return pd.DataFrame(columns=SILVER_MONTHLY_REVENUE_COLUMNS)
    df = raw_df.copy()
    if "date" in df.columns:
        df["month"] = pd.to_datetime(df["date"]).dt.to_period("M").astype(str)
    elif "revenue_month" in df.columns:
        df["month"] = pd.to_datetime(df["revenue_month"].astype(str), format="%Y%m").dt.to_period("M").astype(str)
    else:
        df["month"] = pd.NA
    if "stock_id" not in df.columns:
        df["stock_id"] = pd.NA
    if "revenue" not in df.columns:
        df["revenue"] = pd.NA

    df["stock_id"] = df["stock_id"].astype(str)
    df["revenue"] = pd.to_numeric(df["revenue"], errors="coerce")
    df = df.sort_values(["stock_id", "month"])
    grouped = df.groupby("stock_id", group_keys=False)
    df["revenue_mom"] = grouped["revenue"].pct_change()
    df["revenue_yoy"] = grouped["revenue"].pct_change(12)
    df["revenue_growth_signal"] = [
        _growth_signal(mom, yoy) for mom, yoy in zip(df["revenue_mom"], df["revenue_yoy"], strict=False)
    ]
    return df[SILVER_MONTHLY_REVENUE_COLUMNS]

