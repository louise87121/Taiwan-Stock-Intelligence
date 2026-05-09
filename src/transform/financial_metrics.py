from __future__ import annotations

import pandas as pd

SILVER_FINANCIAL_COLUMNS = ["date", "stock_id", "statement_category", "type", "value", "normalized_metric_name"]
SILVER_PER_DIVIDEND_COLUMNS = ["date", "stock_id", "pe_ratio", "dividend_yield", "price_book_ratio"]

METRIC_NAME_MAP = {
    "EPS": "earnings_per_share",
    "基本每股盈餘": "earnings_per_share",
    "基本每股盈餘（元）": "earnings_per_share",
    "ROE": "return_on_equity",
    "ROA": "return_on_assets",
    "營業收入": "operating_revenue",
    "收益": "operating_revenue",
    "營業毛利": "gross_profit",
    "營業毛利（毛損）": "gross_profit",
    "營業利益": "operating_income",
    "營業利益（損失）": "operating_income",
    "本期淨利（淨損）": "net_income",
    "本期淨利": "net_income",
    "資產總額": "total_assets",
    "資產總計": "total_assets",
    "負債總額": "total_liabilities",
    "負債總計": "total_liabilities",
    "權益總額": "total_equity",
    "權益總計": "total_equity",
    "流動資產": "current_assets",
    "流動負債": "current_liabilities",
    "營業活動之淨現金流入（流出）": "operating_cash_flow",
}


def _normalize_stock_id(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    return text


def _parse_financial_date(value: object) -> str:
    text = str(value).strip()
    if text.isdigit() and len(text) == 7:
        year = int(text[:3]) + 1911
        return f"{year}-{int(text[3:5]):02d}-{int(text[5:7]):02d}"
    parsed = pd.to_datetime(value, errors="coerce")
    return parsed.date().isoformat() if pd.notna(parsed) else text


def transform_financial_statement(raw_df: pd.DataFrame) -> pd.DataFrame:
    if raw_df.empty:
        return pd.DataFrame(columns=SILVER_FINANCIAL_COLUMNS)
    df = raw_df.copy()
    df = df.rename(
        columns={
            "公司代號": "stock_id",
            "公司名稱": "stock_name",
            "年度": "year",
            "季別": "quarter",
            "出表日期": "date",
            "會計項目": "type",
            "金額": "value",
            "本期金額": "value",
            "本期": "value",
        }
    )
    if "type" not in df.columns or "value" not in df.columns:
        id_columns = [col for col in ["date", "year", "quarter", "stock_id", "stock_name", "statement_category", "source_url"] if col in df.columns]
        value_columns = [col for col in df.columns if col not in id_columns]
        if value_columns:
            df = df.melt(id_vars=id_columns, value_vars=value_columns, var_name="type", value_name="value")
    if "date" not in df.columns and {"year", "quarter"}.issubset(df.columns):
        df["date"] = df["year"].astype(str) + "Q" + df["quarter"].astype(str)
    for column in ["date", "stock_id", "statement_category", "type", "value"]:
        if column not in df.columns:
            df[column] = pd.NA
    df["date"] = df["date"].map(_parse_financial_date)
    df["stock_id"] = df["stock_id"].map(_normalize_stock_id)
    df["value"] = pd.to_numeric(df["value"].astype(str).str.replace(",", "", regex=False), errors="coerce")
    df["normalized_metric_name"] = df["type"].map(METRIC_NAME_MAP).fillna(
        df["type"].astype(str).str.lower().str.replace(" ", "_", regex=False)
    )
    return df[SILVER_FINANCIAL_COLUMNS]


def build_fundamental_metrics(
    income_statement: pd.DataFrame,
    balance_sheet: pd.DataFrame,
    cash_flow_statement: pd.DataFrame,
) -> pd.DataFrame:
    frames = [df for df in [income_statement, balance_sheet, cash_flow_statement] if not df.empty]
    columns = [
        "date",
        "stock_id",
        "roe",
        "roa",
        "gross_margin",
        "operating_margin",
        "debt_ratio",
        "current_ratio",
        "eps",
    ]
    if not frames:
        return pd.DataFrame(columns=columns)
    df = pd.concat(frames, ignore_index=True)
    if df.empty:
        return pd.DataFrame(columns=columns)

    pivot = (
        df.assign(metric=df["normalized_metric_name"].fillna(df["type"]))
        .pivot_table(index=["stock_id", "date"], columns="metric", values="value", aggfunc="last")
        .reset_index()
    )

    def first_available(row: pd.Series, names: list[str]) -> float | None:
        for name in names:
            if name in row and pd.notna(row[name]):
                return row[name]
        return None

    def divide(numerator: float | None, denominator: float | None) -> float | pd.NA:
        if numerator is None or denominator is None or pd.isna(numerator) or pd.isna(denominator) or denominator == 0:
            return pd.NA
        return numerator / denominator

    records = []
    for _, row in pivot.iterrows():
        revenue = first_available(row, ["operating_revenue", "revenue", "營業收入", "收益"])
        gross_profit = first_available(row, ["gross_profit", "營業毛利（毛損）", "營業毛利"])
        operating_income = first_available(row, ["operating_income", "營業利益", "營業利益（損失）"])
        net_income = first_available(row, ["net_income", "本期淨利（淨損）", "本期淨利"])
        total_assets = first_available(row, ["total_assets", "資產總額"])
        total_equity = first_available(row, ["total_equity", "權益總額", "權益總計"])
        total_liabilities = first_available(row, ["total_liabilities", "負債總額"])
        current_assets = first_available(row, ["current_assets", "流動資產"])
        current_liabilities = first_available(row, ["current_liabilities", "流動負債"])
        eps = first_available(row, ["earnings_per_share", "EPS", "基本每股盈餘（元）", "基本每股盈餘"])
        records.append(
            {
                "date": row["date"],
                "stock_id": row["stock_id"],
                "roe": divide(net_income, total_equity),
                "roa": divide(net_income, total_assets),
                "gross_margin": divide(gross_profit, revenue),
                "operating_margin": divide(operating_income, revenue),
                "debt_ratio": divide(total_liabilities, total_assets),
                "current_ratio": divide(current_assets, current_liabilities),
                "eps": eps if eps is not None else pd.NA,
            }
        )
    return pd.DataFrame(records, columns=columns).sort_values(["stock_id", "date"])


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
