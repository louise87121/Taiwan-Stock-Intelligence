# Taiwan Stock Intelligence

Taiwan Stock Intelligence 是一個台股基本面與股價智慧分析 App，也是一個 value-oriented data engineering MVP side project。

線上版可直接使用：https://taiwan-stock-intelligence.streamlit.app/ 。

它使用 TWSE OpenAPI 擷取台灣上市公司的每日股價、月營收、財務資料與公司基本資料，整理為 silver/gold analytical tables，並在 Streamlit App 中呈現高流動性台股公司的近五年營運與股價趨勢。

> Disclaimer: 本專案僅作為資料工程與分析 side project，不構成投資建議。Financial Health Score 是 MVP scoring logic，用於展示資料產品設計，不應直接作為買賣依據。

## What It Does

- 比較台灣高流動性公司的營運表現與股價趨勢。
- 追蹤月營收 YoY/MoM、移動平均、波動率與風險訊號。
- 提供公司分析、半導體同業比較、營收成長與風險監控頁面。
- 將 raw TWSE data 轉成 silver/gold analytical tables，並可載入 DuckDB 做 SQL 分析。

## Architecture

```text
TWSE OpenAPI
  -> data/raw CSV
  -> silver tables: cleaned features
  -> gold tables: analytical marts
  -> DuckDB: data/taiwan_stock_intelligence.duckdb
  -> Streamlit app / SQL analysis / Power BI
```

## Data Source

API base URL: `https://openapi.twse.com.tw/v1`

Main datasets:

- 公司基本資料
- 每日收盤行情
- 月營收
- 損益表、資產負債表、現金流量表

## Project Structure

```text
taiwan-stock-intelligence/
├── app.py
├── requirements.txt
├── config/stocks.yml
├── data/raw/
├── data/silver/
├── data/gold/
├── sql/
├── src/
│   ├── api/
│   ├── extract/
│   ├── transform/
│   ├── load/
│   └── main.py
└── tests/
```

## Installation

```bash
cd taiwan-stock-intelligence
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

TWSE OpenAPI endpoints used by this MVP do not require an API token.

## CLI Workflow

```bash
python -m src.main extract
python -m src.main transform
python -m src.main build-gold
python -m src.main load-duckdb
python -m src.main run-all
```

Default data period starts at `2021-01-01` and ends today. You can override it:

```bash
python -m src.main run-all --start-date 2021-01-01 --end-date 2026-05-04
```

`run-all` executes extraction, silver transforms, gold table builds, semiconductor peer comparison, company snapshot generation, and DuckDB loading.

## Run Locally

Generate gold tables first:

```bash
python -m src.main run-all
```

Start the Streamlit app:

```bash
streamlit run app.py
```

If `data/gold/` files do not exist, the app shows a clear message and will not crash.

## Weekly Refresh

The local scheduler runs every Saturday at 08:30 and refreshes data through the most recent Friday:

```bash
python scripts/weekly_refresh.py
```

The launchd job is `com.taiwan-stock-intelligence.weekly-refresh`. Logs are written to `logs/weekly_refresh.log` and `logs/weekly_refresh.err`.

## Deployment

This project can be deployed to Streamlit Community Cloud with `app.py` as the main file. No Streamlit secret is required for the current TWSE OpenAPI workflow.

Generated local data and secrets should not be committed. `.gitignore` excludes `.env`, `data/raw/`, `data/silver/`, `data/gold/`, and `data/*.duckdb`.

## Outputs

Silver tables:

- `silver_company`
- `silver_stock_price`
- `silver_monthly_revenue`
- `silver_financial_statement`
- `silver_income_statement`
- `silver_balance_sheet`
- `silver_cash_flow_statement`
- `silver_fundamental_metrics`
- `silver_per_dividend`

Gold tables:

- `gold_company_monthly_snapshot`
- `gold_operating_dashboard`
- `gold_stock_price_features`
- `gold_revenue_growth`
- `gold_semiconductor_peer_comparison`

Files are written to `data/silver/` and `data/gold/`. The writer prefers Parquet and falls back to CSV if Parquet support is unavailable.

## DuckDB

The local database is created at:

```text
data/taiwan_stock_intelligence.duckdb
```

Run example SQL:

```bash
duckdb data/taiwan_stock_intelligence.duckdb < sql/semiconductor_peer_comparison.sql
duckdb data/taiwan_stock_intelligence.duckdb < sql/top_revenue_growth.sql
```

## Tests

```bash
pytest
```

The tests cover API parameter building, stock price features, monthly revenue YoY, health score bounds, risk mapping, peer ranking, and stock configuration.

## Future Enhancements

- Add data quality checks and anomaly alerts.
- Add incremental extraction state.
- Enrich industry classification.
- Add more robust financial statement metric normalization.
