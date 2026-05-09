from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from pandas.errors import EmptyDataError

from src.config import DEFAULT_START_DATE, GOLD_DIR, RAW_DIR, SILVER_DIR, bootstrap_environment, default_end_date, load_stocks
from src.extract.extract_twse import extract_all_twse, extract_twse_stock_day_history
from src.load.local_storage import load_tables_to_duckdb, read_table, write_table
from src.transform.company import transform_twse_company_info
from src.transform.financial_metrics import build_fundamental_metrics, transform_financial_statement, transform_per_dividend
from src.transform.health_score import append_health_scores
from src.transform.monthly_revenue import transform_monthly_revenue
from src.transform.peer_comparison import build_semiconductor_peer_comparison
from src.transform.stock_price import latest_monthly_price_features, transform_stock_price
from src.utils.logging import get_logger


logger = get_logger(__name__)


def _read_raw(dataset: str, stock_id: str | None = None) -> pd.DataFrame:
    suffix = f"_{stock_id}" if stock_id else ""
    path = RAW_DIR / f"{dataset}{suffix}.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _concat_raw(dataset: str, stock_ids: list[str]) -> pd.DataFrame:
    frames = [_read_raw(dataset, stock_id) for stock_id in stock_ids]
    frames = [df for df in frames if not df.empty]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _read_raw_dataset(dataset: str) -> pd.DataFrame:
    path = RAW_DIR / f"{dataset}.csv"
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except EmptyDataError:
        return pd.DataFrame()


def run_extract(start_date: str = DEFAULT_START_DATE, end_date: str | None = None) -> None:
    bootstrap_environment()
    extract_all_twse()


def run_transform() -> None:
    bootstrap_environment()
    stocks = load_stocks()
    stock_ids = [stock.stock_id for stock in stocks]
    write_table(transform_twse_company_info(_read_raw_dataset("TWSECompanyInfo"), stocks), "silver_company", SILVER_DIR)
    price_frames = [_read_raw_dataset("TWSEStockDayAll"), _concat_raw("TWSEStockDayHistory", stock_ids), _concat_raw("TaiwanStockPrice", stock_ids)]
    price_frames = [df for df in price_frames if not df.empty]
    price_raw = pd.concat(price_frames, ignore_index=True) if price_frames else pd.DataFrame()
    silver_price = transform_stock_price(price_raw)
    if not silver_price.empty:
        silver_price = silver_price[pd.to_datetime(silver_price["date"], errors="coerce") >= pd.Timestamp(DEFAULT_START_DATE)]
    write_table(silver_price, "silver_stock_price", SILVER_DIR)
    revenue_frames = [_read_raw_dataset("TWSEMonthlyRevenue"), _concat_raw("TaiwanStockMonthRevenue", stock_ids)]
    revenue_frames = [df for df in revenue_frames if not df.empty]
    revenue_raw = pd.concat(revenue_frames, ignore_index=True) if revenue_frames else pd.DataFrame()
    silver_revenue = transform_monthly_revenue(revenue_raw)
    if not silver_revenue.empty:
        start_month = pd.Timestamp(DEFAULT_START_DATE).to_period("M").strftime("%Y-%m")
        silver_revenue = silver_revenue[silver_revenue["month"].astype(str).ge(start_month)]
    write_table(silver_revenue, "silver_monthly_revenue", SILVER_DIR)
    financial_frames = []
    for dataset, category, table_name in [
        ("TWSEIncomeStatement", "income_statement", "silver_income_statement"),
        ("TWSEBalanceSheet", "balance_sheet", "silver_balance_sheet"),
        ("TWSECashFlowStatement", "cash_flow_statement", "silver_cash_flow_statement"),
    ]:
        frame = _read_raw_dataset(dataset)
        if not frame.empty:
            frame["statement_category"] = category
        transformed = transform_financial_statement(frame)
        write_table(transformed, table_name, SILVER_DIR)
        if not transformed.empty:
            financial_frames.append(transformed)
    financial_raw = pd.concat([df for df in financial_frames if not df.empty], ignore_index=True) if any(not df.empty for df in financial_frames) else pd.DataFrame()
    write_table(financial_raw, "silver_financial_statement", SILVER_DIR)
    write_table(
        build_fundamental_metrics(
            read_table("silver_income_statement", SILVER_DIR),
            read_table("silver_balance_sheet", SILVER_DIR),
            read_table("silver_cash_flow_statement", SILVER_DIR),
        ),
        "silver_fundamental_metrics",
        SILVER_DIR,
    )
    write_table(
        transform_per_dividend(_concat_raw("TaiwanStockPER", stock_ids), _concat_raw("TaiwanStockDividend", stock_ids)),
        "silver_per_dividend",
        SILVER_DIR,
    )


def run_build_gold() -> None:
    bootstrap_environment()
    company = read_table("silver_company", SILVER_DIR)
    price = read_table("silver_stock_price", SILVER_DIR)
    revenue = read_table("silver_monthly_revenue", SILVER_DIR)
    per_dividend = read_table("silver_per_dividend", SILVER_DIR)
    fundamentals = read_table("silver_fundamental_metrics", SILVER_DIR)

    monthly_price = latest_monthly_price_features(price)
    if not monthly_price.empty:
        monthly_price["month"] = pd.to_datetime(monthly_price["date"]).dt.to_period("M").astype(str)
    if not per_dividend.empty:
        per_dividend["month"] = pd.to_datetime(per_dividend["date"], errors="coerce").dt.to_period("M").astype(str)
        per_dividend = per_dividend.sort_values("date").groupby(["stock_id", "month"], as_index=False).last()
    if not fundamentals.empty:
        fundamentals["month"] = pd.to_datetime(fundamentals["date"], errors="coerce").dt.to_period("M").astype(str)
        fundamentals = fundamentals.sort_values("date").groupby(["stock_id", "month"], as_index=False).last()

    snapshot = revenue.merge(company, on="stock_id", how="left")
    price_cols = [
        "stock_id",
        "month",
        "close_price",
        "monthly_return",
        "volatility_20d",
        "price_above_ma20_flag",
        "price_above_ma60_flag",
    ]
    snapshot = snapshot.merge(monthly_price[price_cols] if not monthly_price.empty else pd.DataFrame(columns=price_cols), on=["stock_id", "month"], how="left")
    per_cols = ["stock_id", "month", "pe_ratio", "dividend_yield", "price_book_ratio"]
    snapshot = snapshot.merge(per_dividend[per_cols] if not per_dividend.empty else pd.DataFrame(columns=per_cols), on=["stock_id", "month"], how="left")
    fundamental_cols = ["stock_id", "month", "roe", "roa", "gross_margin", "operating_margin", "debt_ratio", "current_ratio", "eps"]
    snapshot = snapshot.merge(fundamentals[fundamental_cols] if not fundamentals.empty else pd.DataFrame(columns=fundamental_cols), on=["stock_id", "month"], how="left")
    snapshot = append_health_scores(snapshot)

    company_monthly_cols = [
        "month",
        "stock_id",
        "stock_name",
        "industry_group",
        "market_type",
        "focus_flag",
        "close_price",
        "monthly_return",
        "revenue",
        "revenue_mom",
        "revenue_yoy",
        "pe_ratio",
        "dividend_yield",
        "roe",
        "roa",
        "gross_margin",
        "operating_margin",
        "debt_ratio",
        "current_ratio",
        "eps",
        "volatility_20d",
        "price_above_ma20_flag",
        "price_above_ma60_flag",
        "financial_health_score",
        "risk_level",
    ]
    for column in company_monthly_cols:
        if column not in snapshot.columns:
            snapshot[column] = pd.NA
    gold_company = snapshot[company_monthly_cols].sort_values(["month", "stock_id"])
    write_table(gold_company, "gold_company_monthly_snapshot", GOLD_DIR)

    stock_features = price.merge(company[["stock_id", "stock_name", "industry_group"]], on="stock_id", how="left") if not price.empty else pd.DataFrame()
    stock_feature_cols = [
        "date",
        "stock_id",
        "stock_name",
        "industry_group",
        "close_price",
        "open_price",
        "high_price",
        "low_price",
        "price_change",
        "trading_volume",
        "trading_money",
        "transaction_count",
        "daily_return",
        "monthly_return",
        "ma_20",
        "ma_60",
        "volatility_20d",
        "price_above_ma20_flag",
        "price_above_ma60_flag",
    ]
    write_table(stock_features[stock_feature_cols] if not stock_features.empty else pd.DataFrame(columns=stock_feature_cols), "gold_stock_price_features", GOLD_DIR)

    latest_price = price.sort_values("date").groupby("stock_id", as_index=False).last() if not price.empty else pd.DataFrame()
    latest_price_cols = [
        "stock_id",
        "date",
        "open_price",
        "high_price",
        "low_price",
        "close_price",
        "price_change",
        "trading_volume",
        "trading_money",
        "transaction_count",
        "daily_return",
        "monthly_return",
        "volatility_20d",
        "ma_20",
        "ma_60",
        "price_above_ma20_flag",
        "price_above_ma60_flag",
    ]
    operating_dashboard = gold_company.merge(
        latest_price[latest_price_cols] if not latest_price.empty else pd.DataFrame(columns=["stock_id"]),
        on="stock_id",
        how="left",
        suffixes=("", "_latest_price"),
    )
    for column in ["close_price", "monthly_return", "volatility_20d", "price_above_ma20_flag", "price_above_ma60_flag"]:
        latest_col = f"{column}_latest_price"
        if latest_col in operating_dashboard.columns:
            operating_dashboard[column] = operating_dashboard[column].combine_first(operating_dashboard[latest_col])
            operating_dashboard = operating_dashboard.drop(columns=[latest_col])
    if "date" in operating_dashboard.columns:
        operating_dashboard = operating_dashboard.rename(columns={"date": "latest_price_date"})
    latest_fundamentals = read_table("silver_fundamental_metrics", SILVER_DIR)
    if not latest_fundamentals.empty:
        latest_fundamentals = latest_fundamentals.sort_values("date").groupby("stock_id", as_index=False).last()
        operating_dashboard = operating_dashboard.merge(
            latest_fundamentals[["stock_id", "date", "roe", "roa", "gross_margin", "operating_margin", "debt_ratio", "current_ratio", "eps"]],
            on="stock_id",
            how="left",
            suffixes=("", "_latest_fundamental"),
        )
        for column in ["roe", "roa", "gross_margin", "operating_margin", "debt_ratio", "current_ratio", "eps"]:
            latest_col = f"{column}_latest_fundamental"
            if latest_col in operating_dashboard.columns:
                operating_dashboard[column] = operating_dashboard[column].combine_first(operating_dashboard[latest_col])
                operating_dashboard = operating_dashboard.drop(columns=[latest_col])
        if "date" in operating_dashboard.columns:
            operating_dashboard = operating_dashboard.rename(columns={"date": "latest_fundamental_date"})
        if "date_latest_fundamental" in operating_dashboard.columns:
            operating_dashboard = operating_dashboard.rename(columns={"date_latest_fundamental": "latest_fundamental_date"})
    write_table(operating_dashboard, "gold_operating_dashboard", GOLD_DIR)

    revenue_growth_cols = ["month", "stock_id", "stock_name", "industry_group", "revenue", "revenue_mom", "revenue_yoy", "revenue_growth_signal"]
    for column in revenue_growth_cols:
        if column not in snapshot.columns:
            snapshot[column] = pd.NA
    write_table(snapshot[revenue_growth_cols].sort_values(["month", "stock_id"]), "gold_revenue_growth", GOLD_DIR)

    peers = build_semiconductor_peer_comparison(gold_company)
    write_table(peers, "gold_semiconductor_peer_comparison", GOLD_DIR)

    winbond = gold_company[gold_company["stock_id"].astype(str).eq("2344")].merge(
        peers[["month", "stock_id", "revenue_yoy_rank", "health_score_rank"]],
        on=["month", "stock_id"],
        how="left",
    )
    winbond = winbond.rename(
        columns={
            "revenue_yoy_rank": "semiconductor_revenue_yoy_rank",
            "health_score_rank": "semiconductor_health_score_rank",
        }
    )
    winbond_cols = [
        "month",
        "stock_id",
        "stock_name",
        "close_price",
        "monthly_return",
        "revenue",
        "revenue_mom",
        "revenue_yoy",
        "pe_ratio",
        "dividend_yield",
        "volatility_20d",
        "financial_health_score",
        "risk_level",
        "semiconductor_revenue_yoy_rank",
        "semiconductor_health_score_rank",
    ]
    write_table(winbond[winbond_cols], "gold_winbond_snapshot", GOLD_DIR)


def run_all(start_date: str = DEFAULT_START_DATE, end_date: str | None = None) -> None:
    run_extract(start_date=start_date, end_date=end_date)
    run_transform()
    run_build_gold()
    load_tables_to_duckdb()


def main() -> None:
    parser = argparse.ArgumentParser(description="Taiwan Stock Fundamental & Price Intelligence Platform")
    parser.add_argument("command", choices=["extract", "extract-history", "transform", "build-gold", "load-duckdb", "run-all"])
    parser.add_argument("--start-date", default=DEFAULT_START_DATE)
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--stock-id", action="append", dest="stock_ids", default=None)
    args = parser.parse_args()

    if args.command == "extract":
        run_extract(start_date=args.start_date, end_date=args.end_date)
    elif args.command == "extract-history":
        bootstrap_environment()
        stock_ids = args.stock_ids or [stock.stock_id for stock in load_stocks()]
        extract_twse_stock_day_history(stock_ids, start_date=args.start_date, end_date=args.end_date)
    elif args.command == "transform":
        run_transform()
    elif args.command == "build-gold":
        run_build_gold()
    elif args.command == "load-duckdb":
        bootstrap_environment()
        load_tables_to_duckdb()
    elif args.command == "run-all":
        run_all(start_date=args.start_date, end_date=args.end_date)


if __name__ == "__main__":
    main()
