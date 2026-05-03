from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.api.finmind_client import FinMindClient
from src.config import DEFAULT_START_DATE, GOLD_DIR, RAW_DIR, SILVER_DIR, bootstrap_environment, default_end_date, load_stocks
from src.extract.extract_finmind import extract_all
from src.load.local_storage import load_tables_to_duckdb, read_table, write_table
from src.transform.company import build_silver_company
from src.transform.financial_metrics import transform_financial_statement, transform_per_dividend
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


def run_extract(start_date: str = DEFAULT_START_DATE, end_date: str | None = None) -> None:
    bootstrap_environment()
    stocks = load_stocks()
    extract_all(stocks, start_date=start_date, end_date=end_date or default_end_date(), client=FinMindClient())


def run_transform() -> None:
    bootstrap_environment()
    stocks = load_stocks()
    stock_ids = [stock.stock_id for stock in stocks]
    write_table(build_silver_company(stocks), "silver_company", SILVER_DIR)
    write_table(transform_stock_price(_concat_raw("TaiwanStockPrice", stock_ids)), "silver_stock_price", SILVER_DIR)
    write_table(transform_monthly_revenue(_concat_raw("TaiwanStockMonthRevenue", stock_ids)), "silver_monthly_revenue", SILVER_DIR)
    financial_frames = [
        _concat_raw("TaiwanStockFinancialStatements", stock_ids),
        _concat_raw("TaiwanStockBalanceSheet", stock_ids),
        _concat_raw("TaiwanStockCashFlowsStatement", stock_ids),
    ]
    financial_raw = pd.concat([df for df in financial_frames if not df.empty], ignore_index=True) if any(not df.empty for df in financial_frames) else pd.DataFrame()
    write_table(transform_financial_statement(financial_raw), "silver_financial_statement", SILVER_DIR)
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

    monthly_price = latest_monthly_price_features(price)
    if not monthly_price.empty:
        monthly_price["month"] = pd.to_datetime(monthly_price["date"]).dt.to_period("M").astype(str)
    if not per_dividend.empty:
        per_dividend["month"] = pd.to_datetime(per_dividend["date"], errors="coerce").dt.to_period("M").astype(str)
        per_dividend = per_dividend.sort_values("date").groupby(["stock_id", "month"], as_index=False).last()

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
        "daily_return",
        "ma_20",
        "ma_60",
        "volatility_20d",
        "price_above_ma20_flag",
        "price_above_ma60_flag",
    ]
    write_table(stock_features[stock_feature_cols] if not stock_features.empty else pd.DataFrame(columns=stock_feature_cols), "gold_stock_price_features", GOLD_DIR)

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
    parser.add_argument("command", choices=["extract", "transform", "build-gold", "load-duckdb", "run-all"])
    parser.add_argument("--start-date", default=DEFAULT_START_DATE)
    parser.add_argument("--end-date", default=None)
    args = parser.parse_args()

    if args.command == "extract":
        run_extract(start_date=args.start_date, end_date=args.end_date)
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
