# Taiwan Stock Intelligence

Taiwan Stock Intelligence 是一個台股基本面與股價智慧分析 App，也是一個 value-oriented data engineering MVP side project。它使用 TWSE OpenAPI 擷取台灣上市公司的每日股價、月營收、財務資料與公司基本資料，整理為 silver/gold analytical tables，並在 Streamlit App 中呈現高流動性台股公司的近五年營運與股價趨勢。

> Disclaimer: 本專案僅作為資料工程與分析 side project，不構成投資建議。Financial Health Score 是 MVP scoring logic，用於展示資料產品設計，不應直接作為買賣依據。

## Business Objective

本專案協助使用者比較台灣高流動性公司營運表現、分析股價趨勢、追蹤月營收 YoY/MoM、評估財務健康程度、監控風險訊號，並支援產業同業比較。MVP 的分析 universe 以最新交易日 `trading_money` 排名前段公司為主，資料窗口預設為近五年。

## Web App Overview

`app.py` 是 Streamlit 入口檔案，展示 `data/gold/` analytical tables。Web app 保留資料工程架構：TWSE OpenAPI extraction、silver transformation、gold marts 與 DuckDB loading 由 `src/` CLI 負責，Streamlit 只讀取 silver/gold layer，並預設呈現近五年資料。

App pages:

- Executive Overview: portfolio-level health score、risk distribution、Revenue YoY Top 10、Health Score Top 10、高風險公司清單。
- Company Analysis: 可選公司分析，包含公司基本資料、月營收、每日股價、股價趨勢、日報酬率與股價財務指標關聯。
- Semiconductor Peer Comparison: 半導體同業 revenue YoY、monthly return、volatility、health score 與排名。
- Revenue Growth Analysis: 可選公司月營收、YoY 趨勢、Top/Bottom 10 與 growth signal 分布。
- Risk Monitoring: Watch、High、YoY 衰退且低於 MA60、高波動公司與風險分布。

Screenshots placeholder:

```text
docs/screenshots/executive_overview.png
docs/screenshots/company_analysis.png
docs/screenshots/semiconductor_peer_comparison.png
```

## Why This Is Not Just a Stock Crawler

本專案不是單純下載股價 CSV，而是建立完整資料工程流程：

- Raw layer: 保存 TWSE OpenAPI 原始資料。
- Silver layer: 標準化欄位、計算報酬、移動平均、波動率、營收 YoY/MoM。
- Gold layer: 建立公司月度快照、營收成長、股價特徵與半導體同業比較資料。
- Local warehouse: 將 silver/gold tables 載入 DuckDB，方便 SQL 分析與 Power BI 連接。

## Architecture

```text
TWSE OpenAPI
  -> data/raw CSV
  -> silver tables: cleaned features
  -> gold tables: analytical marts
  -> DuckDB: data/taiwan_stock_intelligence.duckdb
  -> Power BI / SQL analysis / Streamlit app
```

## Data Source

API base URL: `https://openapi.twse.com.tw/v1`

Datasets:

- `opendata/t187ap03_L`: 公司基本資料
- `exchangeReport/STOCK_DAY_ALL`: 每日收盤行情
- `opendata/t187ap05_L`: 月營收
- `opendata/t187ap06_L`: 損益表來源
- `opendata/t187ap07_L`: 資產負債表來源
- `opendata/t187ap08_L`: 現金流量表來源

## Project Structure

```text
taiwan-stock-intelligence/
├── app.py
├── requirements.txt
├── config/stocks.yml
├── data/raw/
├── data/silver/
├── data/gold/
├── notebooks/exploratory_analysis.ipynb
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

## Secrets

TWSE OpenAPI endpoints used by this MVP do not require an API token. `.env` is still ignored by Git for future extensions, but no secret is required to run the current app.

## CLI

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

### Weekly Refresh

The local scheduler runs every Saturday at 08:30 and refreshes data through the most recent Friday:

```bash
python scripts/weekly_refresh.py
```

The launchd job is `com.taiwan-stock-intelligence.weekly-refresh`. Logs are written to:

- `logs/weekly_refresh.log`
- `logs/weekly_refresh.err`

## Run the Streamlit App Locally

Generate gold tables first:

```bash
python -m src.main run-all
```

Start the app:

```bash
streamlit run app.py
```

If `data/gold/` files do not exist, the app shows:

```text
Data files are not found. Please run:
python -m src.main run-all
```

The app will not crash when data files are missing.

## Deploy to Streamlit Community Cloud

1. Push this repository to GitHub.
2. Confirm `app.py` and `requirements.txt` are in the repo root.
3. In Streamlit Community Cloud, create a new app and set the main file path to `app.py`.
4. No Streamlit secret is required for the current TWSE OpenAPI workflow.
5. Ensure generated local data and secrets are not committed. `.gitignore` excludes `.env`, `data/raw/`, `data/silver/`, `data/gold/`, and `data/*.duckdb`.

For a portfolio deployment, either commit small sample gold tables intentionally after reviewing licensing/data policy, or run the ETL workflow in a controlled environment and publish only approved derived outputs.

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

## Financial Health Score

The score starts from 50 and is capped between 0 and 100.

- Revenue YoY > 20%: +20
- Revenue YoY 5% to 20%: +10
- Revenue YoY 0% to 5%: +5
- Revenue YoY < 0%: -15
- Revenue MoM > 5%: +5
- Revenue MoM < -5%: -5
- Price above MA60: +10
- Price below MA60: -5
- Volatility > 0.05: -10
- Volatility 0.03 to 0.05: -5
- Volatility < 0.03: +5
- PE ratio 8 to 25: +10
- PE ratio > 40: -10
- Dividend yield > 0: +5

Risk levels:

- 85 to 100: `Low`
- 75 to 84: `Mid`
- 65 to 74: `High`
- Below 65: `Watch`

## Operating Dashboard Metrics

`gold_operating_dashboard` is the main app table. It is designed for a Goodinfo-like company operating view with a cleaner Streamlit UI. It includes:

- 股票代號、股票名稱
- 開盤價、最高價、最低價、收盤價、漲跌價差
- 成交股數、成交金額、成交筆數
- 股價趨勢、日報酬率、月報酬率、波動率
- 股價與月營收 YoY/MoM 關聯
- ROE、ROA、毛利率、營業利益率、負債比率、流動比率
- 月營收 YoY、月營收 MoM
- EPS、股價報酬率

## Semiconductor Peer Comparison

`gold_semiconductor_peer_comparison` filters semiconductor companies from the company master table. The MVP supports `Semiconductor`, `半導體業`, and TWSE industry code `24`. For each month it ranks:

- `revenue_yoy_rank`: higher YoY is better.
- `monthly_return_rank`: higher monthly return is better.
- `health_score_rank`: higher score is better.
- `volatility_rank`: lower volatility is better.

同業排名可以與 `gold_company_monthly_snapshot` 合併，用於任一半導體公司的公司分析頁。

## Power BI Dashboard Suggestions

1. Executive Overview: 公司總覽、平均 health score、風險公司數、最近月份營收 YoY Top 10。
2. Company Profile: 可選公司月營收、股價、health score、同業排名。
3. Semiconductor Peer Comparison: 半導體公司 revenue growth、return、volatility 與 health score 比較。
4. Revenue Growth Analysis: 月營收 YoY/MoM、growth signal、連續衰退公司。
5. Stock Price Trend: close price、MA20、MA60、volatility、daily/monthly return。
6. Risk Monitoring: Watch、High、營收衰退且高波動、股價低於 MA60 且 YoY 為負。

## Streamlit Dashboard Pages

The Streamlit app uses wide layout, sidebar navigation, company/month/industry filters, metric cards, sortable dataframes, and Plotly charts.

1. Executive Overview: summarizes latest health score, risk counts, Revenue YoY Top 10, Health Score Top 10, and high-risk companies.
2. Company Analysis: selected company page with company profile, monthly revenue, and monthly stock price.
3. Semiconductor Peer Comparison: compares semiconductor peers using revenue growth, return, volatility, valuation, and ranks.
4. Revenue Growth Analysis: focuses on revenue trend, YoY momentum, growth signals, and top/bottom movers.
5. Risk Monitoring: highlights Watch and High companies plus weak revenue/price trend and high volatility signals.

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
- Add Power BI template file.
- Add scheduled orchestration after MVP validation.
- Add cloud deployment only after local data model is stable.
