from __future__ import annotations

import pandas as pd

SILVER_MONTHLY_REVENUE_COLUMNS = ["month", "stock_id", "revenue", "revenue_mom", "revenue_yoy", "revenue_growth_signal"]


def _parse_twse_month(value: object) -> str | pd.NA:
    text = str(value).strip()
    if text.isdigit() and len(text) == 5:
        year = int(text[:3]) + 1911
        return f"{year}-{int(text[3:5]):02d}"
    parsed = pd.to_datetime(value, format="%Y%m", errors="coerce")
    return parsed.to_period("M").strftime("%Y-%m") if pd.notna(parsed) else pd.NA


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
    elif "資料年月" in df.columns:
        df["month"] = df["資料年月"].map(_parse_twse_month)
    else:
        df["month"] = pd.NA
    rename_map = {
        "公司代號": "stock_id",
        "公司名稱": "stock_name",
        "營業收入-當月營收": "revenue",
        "當月營收": "revenue",
        "上月比較增減(%)": "revenue_mom_percent",
        "去年同月增減(%)": "revenue_yoy_percent",
        "營業收入-上月比較增減(%)": "revenue_mom_percent",
        "營業收入-去年同月增減(%)": "revenue_yoy_percent",
    }
    df = df.rename(columns=rename_map)
    if "stock_id" not in df.columns:
        df["stock_id"] = pd.NA
    if "revenue" not in df.columns:
        df["revenue"] = pd.NA

    df["stock_id"] = df["stock_id"].astype(str)
    df["revenue"] = pd.to_numeric(df["revenue"].astype(str).str.replace(",", "", regex=False), errors="coerce")
    df = df.sort_values(["stock_id", "month"])
    grouped = df.groupby("stock_id", group_keys=False)
    df["revenue_mom"] = grouped["revenue"].pct_change()
    df["revenue_yoy"] = grouped["revenue"].pct_change(12)
    if "revenue_mom_percent" in df.columns:
        df["revenue_mom"] = pd.to_numeric(df["revenue_mom_percent"], errors="coerce") / 100
    if "revenue_yoy_percent" in df.columns:
        df["revenue_yoy"] = pd.to_numeric(df["revenue_yoy_percent"], errors="coerce") / 100
    df["revenue_growth_signal"] = [
        _growth_signal(mom, yoy) for mom, yoy in zip(df["revenue_mom"], df["revenue_yoy"], strict=False)
    ]
    return df[SILVER_MONTHLY_REVENUE_COLUMNS]
