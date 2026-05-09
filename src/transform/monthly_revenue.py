from __future__ import annotations

import pandas as pd

SILVER_MONTHLY_REVENUE_COLUMNS = ["month", "stock_id", "revenue", "revenue_mom", "revenue_yoy", "revenue_growth_signal"]


def _parse_twse_month(value: object) -> str | pd.NA:
    numeric_value = pd.to_numeric(value, errors="coerce")
    text = str(int(numeric_value)) if pd.notna(numeric_value) else str(value).strip()
    if text.isdigit() and len(text) == 5:
        year = int(text[:3]) + 1911
        return f"{year}-{int(text[3:5]):02d}"
    parsed = pd.to_datetime(value, format="%Y%m", errors="coerce")
    return parsed.to_period("M").strftime("%Y-%m") if pd.notna(parsed) else pd.NA


def _normalize_stock_id(value: object) -> str:
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    return text


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
    df["month"] = pd.NA
    if "date" in df.columns:
        parsed_date_month = pd.to_datetime(df["date"], errors="coerce").dt.to_period("M").astype(str)
        df["month"] = parsed_date_month.mask(parsed_date_month.eq("NaT"), pd.NA)
    if "revenue_month" in df.columns and "revenue_year" in df.columns:
        revenue_year_month = (
            pd.to_numeric(df["revenue_year"], errors="coerce").astype("Int64").astype(str)
            + "-"
            + pd.to_numeric(df["revenue_month"], errors="coerce").astype("Int64").astype(str).str.zfill(2)
        )
        df["month"] = df["month"].fillna(revenue_year_month.mask(revenue_year_month.str.contains("<NA>", na=True), pd.NA))
    if "資料年月" in df.columns:
        df["month"] = df["month"].fillna(df["資料年月"].map(_parse_twse_month))
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
    df = df.T.groupby(level=0).first().T
    if "stock_id" not in df.columns:
        df["stock_id"] = pd.NA
    if "revenue" not in df.columns:
        df["revenue"] = pd.NA

    df["stock_id"] = df["stock_id"].map(_normalize_stock_id)
    df["revenue"] = pd.to_numeric(df["revenue"].astype(str).str.replace(",", "", regex=False), errors="coerce")
    if "資料年月" in df.columns:
        twse_revenue_mask = df["資料年月"].notna()
        df.loc[twse_revenue_mask, "revenue"] = df.loc[twse_revenue_mask, "revenue"] * 1000
    df = df.sort_values(["stock_id", "month"])
    grouped = df.groupby("stock_id", group_keys=False)
    df["revenue_mom"] = grouped["revenue"].pct_change()
    df["revenue_yoy"] = grouped["revenue"].pct_change(12)
    df["_has_reported_growth"] = False
    if "revenue_mom_percent" in df.columns:
        reported_mom = pd.to_numeric(df["revenue_mom_percent"], errors="coerce") / 100
        df["revenue_mom"] = df["revenue_mom"].mask(reported_mom.notna(), reported_mom)
        df["_has_reported_growth"] = df["_has_reported_growth"] | reported_mom.notna()
    if "revenue_yoy_percent" in df.columns:
        reported_yoy = pd.to_numeric(df["revenue_yoy_percent"], errors="coerce") / 100
        df["revenue_yoy"] = df["revenue_yoy"].mask(reported_yoy.notna(), reported_yoy)
        df["_has_reported_growth"] = df["_has_reported_growth"] | reported_yoy.notna()
    df = df.dropna(subset=["month", "stock_id"]).sort_values(["stock_id", "month", "_has_reported_growth"])
    df = df.drop_duplicates(["stock_id", "month"], keep="last")
    df["revenue_growth_signal"] = [
        _growth_signal(mom, yoy) for mom, yoy in zip(df["revenue_mom"], df["revenue_yoy"], strict=False)
    ]
    return df[SILVER_MONTHLY_REVENUE_COLUMNS]
