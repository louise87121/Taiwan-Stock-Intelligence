from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pandas as pd
import yaml
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / "config"
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
SILVER_DIR = DATA_DIR / "silver"
GOLD_DIR = DATA_DIR / "gold"
DUCKDB_PATH = DATA_DIR / "taiwan_stock_intelligence.duckdb"
DEFAULT_START_DATE = "2021-01-01"


@dataclass(frozen=True)
class StockConfig:
    stock_id: str
    stock_name: str
    industry_group: str
    market_type: str
    focus_flag: bool = False


def bootstrap_environment() -> None:
    load_dotenv(PROJECT_ROOT / ".env")
    for path in (RAW_DIR, SILVER_DIR, GOLD_DIR):
        path.mkdir(parents=True, exist_ok=True)


def default_end_date() -> str:
    return date.today().isoformat()


def load_stocks(path: Path | None = None) -> list[StockConfig]:
    stocks_path = path or CONFIG_DIR / "stocks.yml"
    with stocks_path.open("r", encoding="utf-8") as file:
        payload = yaml.safe_load(file) or {}
    return [StockConfig(**row) for row in payload.get("stocks", [])]


def stocks_to_frame(stocks: list[StockConfig]) -> pd.DataFrame:
    return pd.DataFrame([stock.__dict__ for stock in stocks])

