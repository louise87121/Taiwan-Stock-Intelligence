from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import DUCKDB_PATH, GOLD_DIR, SILVER_DIR
from src.utils.logging import get_logger


logger = get_logger(__name__)


def write_table(df: pd.DataFrame, table_name: str, directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    parquet_path = directory / f"{table_name}.parquet"
    csv_path = directory / f"{table_name}.csv"
    try:
        df.to_parquet(parquet_path, index=False)
        return parquet_path
    except Exception as exc:
        logger.warning("Parquet write failed for %s, falling back to CSV: %s", table_name, exc)
        df.to_csv(csv_path, index=False)
        return csv_path


def read_table(table_name: str, directory: Path) -> pd.DataFrame:
    parquet_path = directory / f"{table_name}.parquet"
    csv_path = directory / f"{table_name}.csv"
    if parquet_path.exists():
        return pd.read_parquet(parquet_path)
    if csv_path.exists():
        return pd.read_csv(csv_path)
    return pd.DataFrame()


def write_raw(df: pd.DataFrame, dataset: str, stock_id: str | None, raw_dir: Path) -> Path:
    suffix = f"_{stock_id}" if stock_id else ""
    path = raw_dir / f"{dataset}{suffix}.csv"
    raw_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def load_tables_to_duckdb(
    db_path: Path = DUCKDB_PATH,
    silver_dir: Path = SILVER_DIR,
    gold_dir: Path = GOLD_DIR,
) -> None:
    try:
        import duckdb
    except ModuleNotFoundError as exc:
        raise RuntimeError("duckdb is required for load-duckdb. Install dependencies with: pip install -r requirements.txt") from exc

    table_sources = {
        "silver_company": silver_dir,
        "silver_stock_price": silver_dir,
        "silver_monthly_revenue": silver_dir,
        "silver_financial_statement": silver_dir,
        "silver_per_dividend": silver_dir,
        "gold_company_monthly_snapshot": gold_dir,
        "gold_stock_price_features": gold_dir,
        "gold_revenue_growth": gold_dir,
        "gold_semiconductor_peer_comparison": gold_dir,
        "gold_winbond_snapshot": gold_dir,
    }
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with duckdb.connect(str(db_path)) as con:
        for table_name, directory in table_sources.items():
            df = read_table(table_name, directory)
            if df.empty:
                logger.warning("Skipping DuckDB load for empty/missing table %s", table_name)
                continue
            con.register("df_view", df)
            con.execute(f"CREATE OR REPLACE TABLE {table_name} AS SELECT * FROM df_view")
            con.unregister("df_view")
            logger.info("Loaded %s rows into DuckDB table %s", len(df), table_name)
