from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable

import pandas as pd
import plotly.express as px
import streamlit as st
import yaml


APP_TITLE = "Taiwan Stock Intelligence"
PROJECT_ROOT = Path(__file__).resolve().parent
CONFIG_PATH = PROJECT_ROOT / "config" / "stocks.yml"
GOLD_DIR = PROJECT_ROOT / "data" / "gold"
SILVER_DIR = PROJECT_ROOT / "data" / "silver"
RISK_LEVEL_ORDER = ["Watch", "High", "Mid", "Low"]
RISK_COLOR_MAP = {
    "Watch": "#dc2626",
    "High": "#fca5a5",
    "Mid": "#93c5fd",
    "Low": "#2563eb",
}

TWSE_INDUSTRY_LABELS = {
    "1": "Cement",
    "2": "Food",
    "3": "Plastics",
    "4": "Textiles",
    "5": "Electric Machinery",
    "6": "Electrical and Cable",
    "8": "Glass and Ceramics",
    "9": "Paper and Pulp",
    "10": "Iron and Steel",
    "11": "Rubber",
    "12": "Automobile",
    "14": "Building Material and Construction",
    "15": "Shipping and Transportation",
    "16": "Tourism",
    "17": "Financial and Insurance",
    "18": "Trading and Consumers' Goods",
    "20": "Other",
    "21": "Chemical",
    "22": "Biotechnology and Medical Care",
    "23": "Oil, Gas and Electricity",
    "24": "Semiconductor",
    "25": "Computer and Peripheral Equipment",
    "26": "Optoelectronic",
    "27": "Communications and Internet",
    "28": "Electronic Parts and Components",
    "29": "Electronic Products Distribution",
    "30": "Information Service",
    "31": "Other Electronic",
    "35": "Green Energy and Environmental Services",
    "36": "Digital and Cloud Services",
    "37": "Sports and Leisure",
    "38": "Household",
    "91": "Taiwan Depositary Receipts",
}


st.set_page_config(page_title=APP_TITLE, layout="wide")

st.markdown(
    """
    <style>
    .block-container {padding-top: 1.4rem; padding-bottom: 2rem;}
    div[data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 14px 16px;
        box-shadow: 0 1px 2px rgba(15, 23, 42, 0.06);
    }
    div[data-testid="stMetricLabel"] {font-size: 0.86rem; color: #475569;}
    div[data-testid="stMetricValue"] {font-size: 1.45rem; color: #0f172a;}
    .stTabs [data-baseweb="tab-list"] {gap: 8px;}
    .stTabs [data-baseweb="tab"] {
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 8px 14px;
        background: #f8fafc;
    }
    .stTabs [aria-selected="true"] {
        background: #0f172a !important;
        color: #ffffff !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def load_gold_table(table_name: str) -> pd.DataFrame:
    parquet_path = GOLD_DIR / f"{table_name}.parquet"
    csv_path = GOLD_DIR / f"{table_name}.csv"
    if parquet_path.exists():
        return pd.read_parquet(parquet_path)
    if csv_path.exists():
        return pd.read_csv(csv_path)
    return pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_silver_table(table_name: str) -> pd.DataFrame:
    parquet_path = SILVER_DIR / f"{table_name}.parquet"
    csv_path = SILVER_DIR / f"{table_name}.csv"
    if parquet_path.exists():
        return pd.read_parquet(parquet_path)
    if csv_path.exists():
        return pd.read_csv(csv_path)
    return pd.DataFrame()


def show_missing_data_warning(table_names: Iterable[str]) -> bool:
    missing = [
        table_name
        for table_name in table_names
        if not (GOLD_DIR / f"{table_name}.parquet").exists() and not (GOLD_DIR / f"{table_name}.csv").exists()
    ]
    if missing:
        st.warning(
            "Data files are not found. Please run:\n\n"
            "`python -m src.main run-all`\n\n"
            f"Missing gold tables: {', '.join(missing)}"
        )
        return True
    return False


def normalize_types(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    normalized = df.copy()
    if "stock_id" in normalized.columns:
        normalized["stock_id"] = normalized["stock_id"].astype(str)
    if "month" in normalized.columns:
        normalized["month"] = normalized["month"].astype(str)
    return normalized


def latest_price_date_label(stock_features: pd.DataFrame, operating_dashboard: pd.DataFrame) -> str:
    date_values = []
    if not stock_features.empty and "date" in stock_features.columns:
        date_values.append(pd.to_datetime(stock_features["date"], errors="coerce").max())
    if not operating_dashboard.empty and "latest_price_date" in operating_dashboard.columns:
        date_values.append(pd.to_datetime(operating_dashboard["latest_price_date"], errors="coerce").max())
    valid_dates = [value for value in date_values if pd.notna(value)]
    if not valid_dates:
        return "N/A"
    return max(valid_dates).strftime("%Y-%m-%d")


def page_header(title: str, latest_price_date: str) -> None:
    title_col, date_col = st.columns([0.72, 0.28], vertical_alignment="top")
    with title_col:
        st.title(title)
    with date_col:
        st.markdown(
            f"""
            <div style="text-align: right; padding-top: 0.85rem;">
              <span style="color: #64748b; font-size: 0.86rem;">Latest price date</span><br>
              <strong style="font-size: 1rem;">{latest_price_date}</strong>
            </div>
            """,
            unsafe_allow_html=True,
        )


@st.cache_data(show_spinner=False)
def load_config_stocks() -> list[dict[str, str]]:
    if not CONFIG_PATH.exists():
        return []
    with CONFIG_PATH.open("r", encoding="utf-8") as file:
        payload = yaml.safe_load(file) or {}
    return [
        {
            "stock_id": str(row.get("stock_id")).strip(),
            "stock_name": str(row.get("stock_name", "")).strip(),
        }
        for row in payload.get("stocks", [])
        if row.get("stock_id")
    ]


def load_config_stock_ids() -> list[str]:
    return [row["stock_id"] for row in load_config_stocks()]


def load_config_stock_names() -> dict[str, str]:
    return {row["stock_id"]: row["stock_name"] for row in load_config_stocks() if row.get("stock_name")}


def get_app_universe_stock_ids(operating_dashboard: pd.DataFrame) -> list[str]:
    configured_stock_ids = load_config_stock_ids()
    if configured_stock_ids:
        return configured_stock_ids
    if operating_dashboard.empty or "stock_id" not in operating_dashboard.columns:
        return []
    df = operating_dashboard.copy()
    if "latest_price_date" in df.columns:
        df["latest_price_date"] = pd.to_datetime(df["latest_price_date"], errors="coerce")
    else:
        df["latest_price_date"] = pd.NaT
    if "trading_money" not in df.columns:
        df["trading_money"] = 0
    df["trading_money"] = pd.to_numeric(df["trading_money"], errors="coerce").fillna(0)
    latest = (
        df.sort_values(["latest_price_date", "trading_money"], ascending=[False, False])
        .drop_duplicates("stock_id", keep="first")
        .sort_values("trading_money", ascending=False)
        .head(20)
    )
    stock_ids = latest["stock_id"].astype(str).tolist()
    for required_stock_id in ["2344"]:
        if required_stock_id not in stock_ids:
            stock_ids.append(required_stock_id)
    return stock_ids


def keep_stocks_with_enough_data(
    universe_stock_ids: list[str],
    snapshot: pd.DataFrame,
    stock_features: pd.DataFrame,
    revenue_growth: pd.DataFrame,
    min_months: int = 12,
) -> list[str]:
    if not universe_stock_ids:
        return []

    def month_counts(df: pd.DataFrame) -> pd.Series:
        if df.empty or "stock_id" not in df.columns:
            return pd.Series(dtype="int64")
        working = df.copy()
        if "month" not in working.columns and "date" in working.columns:
            working["month"] = pd.to_datetime(working["date"], errors="coerce").dt.to_period("M").astype(str)
        if "month" not in working.columns:
            return pd.Series(dtype="int64")
        working = working.dropna(subset=["month"])
        return working.groupby(working["stock_id"].astype(str))["month"].nunique()

    snapshot_counts = month_counts(snapshot)
    price_counts = month_counts(stock_features)
    revenue_counts = month_counts(revenue_growth)
    qualified = []
    for stock_id in universe_stock_ids:
        if (
            snapshot_counts.get(stock_id, 0) >= min_months
            and price_counts.get(stock_id, 0) >= min_months
            and revenue_counts.get(stock_id, 0) >= min_months
        ):
            qualified.append(stock_id)
    return qualified


def filter_app_universe(df: pd.DataFrame, universe_stock_ids: list[str]) -> pd.DataFrame:
    if df.empty or not universe_stock_ids or "stock_id" not in df.columns:
        return df
    return df[df["stock_id"].astype(str).isin(universe_stock_ids)].copy()


def latest_month(df: pd.DataFrame) -> str | None:
    if df.empty or "month" not in df.columns:
        return None
    months = sorted(df["month"].dropna().astype(str).unique())
    return months[-1] if months else None


def filter_month_range(df: pd.DataFrame, month_range: tuple[str, str] | None) -> pd.DataFrame:
    if df.empty or "month" not in df.columns or not month_range:
        return df
    start_month, end_month = month_range
    return df[df["month"].between(start_month, end_month)].copy()


def months_from_frame(df: pd.DataFrame) -> list[str]:
    if df.empty:
        return []
    if "month" in df.columns:
        return df["month"].dropna().astype(str).tolist()
    if "date" in df.columns:
        dates = pd.to_datetime(df["date"], errors="coerce")
        return dates.dropna().dt.to_period("M").astype(str).tolist()
    return []


def monthly_stock_price_features(stock_features: pd.DataFrame) -> pd.DataFrame:
    if stock_features.empty or "date" not in stock_features.columns:
        return pd.DataFrame()
    df = stock_features.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values(["stock_id", "date"])
    df["month"] = df["date"].dt.to_period("M").astype(str)
    idx = df.groupby(["stock_id", "month"], as_index=False).tail(1).index
    monthly = df.loc[idx].copy()
    monthly["date"] = monthly["date"].dt.date.astype(str)
    return monthly.sort_values(["stock_id", "month"])


def filter_industry(df: pd.DataFrame, industry_group: str) -> pd.DataFrame:
    if df.empty or industry_group == "All" or "industry_group" not in df.columns:
        return df
    return df[df["industry_group"].eq(industry_group)].copy()


def industry_group_label(industry_group: object) -> str:
    if pd.isna(industry_group):
        return "Unknown"
    value = str(industry_group).strip()
    return TWSE_INDUSTRY_LABELS.get(value, value)


def format_percent(value: float | int | None) -> str:
    if pd.isna(value):
        return "N/A"
    return f"{value:.2%}"


def format_number(value: float | int | None) -> str:
    if pd.isna(value):
        return "N/A"
    return f"{value:,.0f}"


def format_price(value: float | int | None) -> str:
    if pd.isna(value):
        return "N/A"
    return f"{value:,.2f}"


def format_table(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    formatted = df.copy()
    if "industry_group" in formatted.columns:
        formatted["industry_group"] = formatted["industry_group"].map(industry_group_label)
    percent_cols = [
        "revenue_yoy",
        "revenue_mom",
        "daily_return",
        "monthly_return",
        "volatility_20d",
        "dividend_yield",
        "roe",
        "roa",
        "gross_margin",
        "operating_margin",
        "debt_ratio",
    ]
    for column in percent_cols:
        if column in formatted.columns:
            formatted[column] = formatted[column].map(format_percent)
    for column in [
        "revenue",
        "close_price",
        "open_price",
        "high_price",
        "low_price",
        "price_change",
        "pe_ratio",
        "financial_health_score",
        "trading_volume",
        "trading_money",
        "transaction_count",
        "eps",
        "current_ratio",
    ]:
        if column in formatted.columns:
            formatted[column] = formatted[column].map(
                format_price if column in {"close_price", "open_price", "high_price", "low_price", "price_change", "pe_ratio", "eps", "current_ratio"} else format_number
            )
    return formatted


def centered_table(df: pd.DataFrame):
    return df.style.set_properties(**{"text-align": "center"}).set_table_styles(
        [
            {"selector": "th", "props": [("text-align", "center")]},
            {"selector": "td", "props": [("text-align", "center")]},
        ]
    )


def filter_stock(df: pd.DataFrame, stock_id: str) -> pd.DataFrame:
    if df.empty or not stock_id or "stock_id" not in df.columns:
        return pd.DataFrame()
    return df[df["stock_id"].astype(str).eq(stock_id)].copy()


def latest_record_per_stock(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "stock_id" not in df.columns:
        return pd.DataFrame()
    sort_cols = [col for col in ["month", "latest_price_date", "date"] if col in df.columns]
    if not sort_cols:
        return df.drop_duplicates("stock_id", keep="last").copy()
    return df.sort_values(sort_cols).groupby(df["stock_id"].astype(str), as_index=False).tail(1).copy()


def strict_risk_level(score: float) -> str:
    if pd.notna(score) and score >= 85:
        return "Low"
    if pd.notna(score) and score >= 75:
        return "Mid"
    if pd.notna(score) and score >= 65:
        return "High"
    return "Watch"


def apply_monitoring_risk_levels(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "financial_health_score" not in df.columns:
        return df.copy()
    monitored = df.copy()
    scores = pd.to_numeric(monitored["financial_health_score"], errors="coerce")
    ranked_index = monitored.assign(_risk_sort_score=scores).sort_values(
        ["_risk_sort_score", "stock_id"],
        na_position="last",
    ).index.tolist()
    company_count = len(ranked_index)
    if company_count == 0:
        return monitored

    watch_count = max(1, math.ceil(company_count * 0.15))
    high_count = max(1, math.ceil(company_count * 0.25)) if company_count > watch_count else 0
    monitored["risk_level"] = scores.map(strict_risk_level)
    monitored.loc[ranked_index[:watch_count], "risk_level"] = "Watch"
    monitored.loc[ranked_index[watch_count : watch_count + high_count], "risk_level"] = "High"
    return monitored


def show_empty_state(message: str) -> None:
    st.info(message)


def bar_chart(df: pd.DataFrame, x: str, y: str, title: str, color: str | None = None) -> None:
    if df.empty or x not in df.columns or y not in df.columns:
        show_empty_state(f"No data available for {title}.")
        return
    category_orders = {x: df[x].astype(str).tolist()}
    if color == "risk_level":
        category_orders[color] = RISK_LEVEL_ORDER
    fig = px.bar(
        df,
        x=x,
        y=y,
        color=color,
        title=title,
        text_auto=".2s",
        category_orders=category_orders,
        color_discrete_map=RISK_COLOR_MAP if color == "risk_level" else None,
    )
    fig.update_layout(xaxis_title="", yaxis_title=y, height=420)
    fig.update_xaxes(categoryorder="array", categoryarray=category_orders[x])
    st.plotly_chart(fig, use_container_width=True)


def horizontal_bar_chart(df: pd.DataFrame, x: str, y: str, title: str, color: str | None = None) -> None:
    if df.empty or x not in df.columns or y not in df.columns:
        show_empty_state(f"No data available for {title}.")
        return
    fig = px.bar(df, x=x, y=y, color=color, orientation="h", title=title, text_auto=".2s")
    fig.update_layout(xaxis_title=x, yaxis_title="", height=420, yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig, use_container_width=True)


def scatter_chart(df: pd.DataFrame, x: str, y: str, title: str, color: str | None = None) -> None:
    if df.empty or x not in df.columns or y not in df.columns:
        show_empty_state(f"No data available for {title}.")
        return
    fig = px.scatter(df, x=x, y=y, color=color, title=title)
    fig.update_layout(height=420)
    st.plotly_chart(fig, use_container_width=True)


def line_chart(df: pd.DataFrame, x: str, y: str, title: str, color: str | None = None) -> None:
    if df.empty or x not in df.columns or y not in df.columns:
        show_empty_state(f"No data available for {title}.")
        return
    chart_df = df.copy()
    if x == "date":
        chart_df["month"] = pd.to_datetime(chart_df[x], errors="coerce").dt.to_period("M").astype(str)
        x = "month"
    chart_df = chart_df.dropna(subset=[x, y]).sort_values(x)
    if chart_df.empty:
        show_empty_state(f"No data available for {title}.")
        return
    fig = px.line(chart_df, x=x, y=y, color=color, markers=True, title=title)
    fig.update_layout(xaxis_title="", yaxis_title=y, height=420)
    if x == "month":
        fig.update_xaxes(type="category", categoryorder="array", categoryarray=sorted(chart_df[x].dropna().astype(str).unique()))
    st.plotly_chart(fig, use_container_width=True)


def build_industry_peer_ranks(snapshot: pd.DataFrame, stock_id: str, month_range: tuple[str, str] | None) -> pd.DataFrame:
    if snapshot.empty or not stock_id:
        return pd.DataFrame()
    df = filter_month_range(snapshot.copy(), month_range)
    selected = df[df["stock_id"].astype(str).eq(stock_id)].copy()
    if selected.empty or "industry_group" not in selected.columns:
        return pd.DataFrame()

    industry_group = selected.sort_values("month")["industry_group"].dropna().iloc[-1]
    peers = df[df["industry_group"].eq(industry_group)].copy()
    if peers.empty:
        return pd.DataFrame()

    peers["industry_peer_count"] = peers.groupby("month")["stock_id"].transform("nunique")
    peers["industry_revenue_yoy_rank"] = peers.groupby("month")["revenue_yoy"].rank(ascending=False, method="min")
    peers["industry_monthly_return_rank"] = peers.groupby("month")["monthly_return"].rank(ascending=False, method="min")
    peers["industry_health_score_rank"] = peers.groupby("month")["financial_health_score"].rank(ascending=False, method="min")
    peers["industry_volatility_rank"] = peers.groupby("month")["volatility_20d"].rank(ascending=True, method="min")
    rank_cols = [
        "month",
        "stock_id",
        "stock_name",
        "industry_group",
        "industry_peer_count",
        "industry_revenue_yoy_rank",
        "industry_monthly_return_rank",
        "industry_health_score_rank",
        "industry_volatility_rank",
    ]
    return peers[peers["stock_id"].astype(str).eq(stock_id)][rank_cols].sort_values("month")


def financial_statement_section(financials: pd.DataFrame, stock_id: str, title: str) -> None:
    st.subheader(title)
    df = filter_stock(financials, stock_id)
    if df.empty:
        show_empty_state(f"No {title} data is available. Please run `python -m src.main run-all` and confirm silver financial data exists.")
        return

    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date.astype(str)
    df = df.sort_values(["date", "type"])
    latest_date = df["date"].dropna().max()
    latest = df[df["date"].eq(latest_date)].copy() if latest_date else df.tail(50)
    st.caption(f"Latest statement date: {latest_date or 'N/A'}")
    st.dataframe(
        latest[["date", "type", "normalized_metric_name", "value"]].sort_values("type"),
        use_container_width=True,
        hide_index=True,
    )

    metric_options = sorted(df["normalized_metric_name"].dropna().astype(str).unique())
    if metric_options:
        selected_metrics = st.multiselect(
            f"{title} metrics",
            metric_options,
            default=metric_options[: min(3, len(metric_options))],
            key=f"{title}_metrics",
        )
        trend = df[df["normalized_metric_name"].isin(selected_metrics)].copy()
        line_chart(trend, "date", "value", f"{title} Metric Trend", "normalized_metric_name")


def page_executive_overview(snapshot: pd.DataFrame, month_range: tuple[str, str] | None, industry_group: str, latest_price_date: str) -> None:
    page_header("Executive Overview", latest_price_date)
    st.markdown(
        """
        **Taiwan Stock Intelligence** 是一個 value-oriented data application，
        用資料工程流程整合台灣上市公司的每日股價、月營收、財務指標與產業分類，協助使用者快速理解公司營運狀況、
        市場反應與潛在風險。

        這個 App 不是單純的股價查詢工具。資料會先經過 raw、silver、gold 三層處理，
        再轉成適合互動分析的表格與圖表，支援公司分析、月營收追蹤、產業同業比較與風險監控。
        目前資料來源以 TWSE / MOPS 公開資料為主，分析窗口預設為近五年。
        """
    )
    st.info(
        "Disclaimer: 本專案僅作為資料工程與分析 side project，不構成投資建議。"
        "Financial Health Score 是 MVP scoring logic，用於展示資料產品設計，不應直接作為買賣依據。"
    )
    st.caption("Business interpretation: this page summarizes market-wide operating momentum, valuation-sensitive health scores, and current risk concentration across the selected coverage universe.")
    df = filter_industry(filter_month_range(snapshot, month_range), industry_group)
    lm = latest_month(df)
    latest = apply_monitoring_risk_levels(latest_record_per_stock(df))

    if latest.empty:
        show_empty_state("No company monthly snapshot data is available for the selected filters.")
        return

    metric_cols = st.columns(5)
    metric_cols[0].metric("Avg Health Score", format_number(latest["financial_health_score"].mean()))
    for idx, risk_level in enumerate(["Low", "Mid", "High", "Watch"], start=1):
        metric_cols[idx].metric(risk_level, f"{latest['risk_level'].eq(risk_level).sum():,}")

    st.subheader(f"Latest Available Company Records: up to {lm}")
    col1, col2 = st.columns(2)
    with col1:
        top_revenue = latest.sort_values("revenue_yoy", ascending=False, na_position="last").head(10)
        bar_chart(top_revenue, "stock_name", "revenue_yoy", "Revenue YoY Top 10", "risk_level")
    with col2:
        top_health = latest.sort_values("financial_health_score", ascending=False, na_position="last").head(10)
        bar_chart(top_health, "stock_name", "financial_health_score", "Financial Health Score Top 10", "risk_level")

    st.subheader("Watch / High Companies")
    high_risk = latest[latest["risk_level"].isin(["High", "Watch"])].sort_values("financial_health_score")
    st.dataframe(
        format_table(high_risk[["stock_id", "stock_name", "industry_group", "revenue_yoy", "monthly_return", "volatility_20d", "financial_health_score", "risk_level"]]),
        use_container_width=True,
        hide_index=True,
    )


def page_company_analysis(
    snapshot: pd.DataFrame,
    operating_dashboard: pd.DataFrame,
    stock_features: pd.DataFrame,
    revenue_growth: pd.DataFrame,
    peers: pd.DataFrame,
    stock_id: str,
    month_range: tuple[str, str] | None,
    latest_price_date: str,
) -> None:
    page_header("Company Analysis", latest_price_date)
    st.caption("Business interpretation: this page organizes the selected company into operating profile, monthly revenue, and daily stock price analysis.")
    source = operating_dashboard if not operating_dashboard.empty else snapshot
    company = source[source["stock_id"].astype(str).eq(stock_id)].copy() if not source.empty and stock_id else pd.DataFrame()
    company = filter_month_range(company, month_range)
    if company.empty:
        show_empty_state("No company snapshot data is available for the selected company. Please run the ETL workflow first.")
        return

    company_name = company["stock_name"].dropna().iloc[-1] if "stock_name" in company.columns and company["stock_name"].notna().any() else stock_id
    latest = company.sort_values("month").tail(1).iloc[0]
    cols = st.columns(6)
    cols[0].metric("Latest Close Price", format_price(latest.get("close_price")))
    cols[1].metric("Price Change", format_price(latest.get("price_change")))
    cols[2].metric("Daily Return", format_percent(latest.get("daily_return")))
    cols[3].metric("Monthly Return", format_percent(latest.get("monthly_return")))
    cols[4].metric("Revenue YoY", format_percent(latest.get("revenue_yoy")))
    cols[5].metric("EPS", format_price(latest.get("eps")))

    tabs = st.tabs(["公司基本資料", "月營收", "月股價"])

    with tabs[0]:
        st.subheader("公司基本資料")
        profile_cols = [
            "month",
            "latest_price_date",
            "latest_fundamental_date",
            "stock_id",
            "stock_name",
            "industry_group",
            "market_type",
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
            "revenue_yoy",
            "revenue_mom",
            "roe",
            "roa",
            "gross_margin",
            "operating_margin",
            "debt_ratio",
            "current_ratio",
            "eps",
            "risk_level",
            "financial_health_score",
        ]
        industry_rank = build_industry_peer_ranks(source, stock_id, month_range)
        profile = company[[col for col in profile_cols if col in company.columns]].tail(1).copy()
        if not industry_rank.empty:
            latest_rank = industry_rank.tail(1).drop(columns=["stock_id", "stock_name", "industry_group"], errors="ignore")
            profile = profile.merge(latest_rank, on="month", how="left") if "month" in profile.columns else profile

        st.dataframe(
            format_table(profile),
            use_container_width=True,
            hide_index=True,
        )
        profile_price = filter_stock(monthly_stock_price_features(stock_features), stock_id)
        profile_price = filter_month_range(profile_price, month_range)
        line_chart(profile_price, "month", "close_price", f"{company_name} 月股價趨勢")

        if not industry_rank.empty:
            rank_value_cols = [
                "industry_revenue_yoy_rank",
                "industry_monthly_return_rank",
                "industry_health_score_rank",
                "industry_volatility_rank",
            ]
            rank_long = industry_rank[["month", *rank_value_cols]].melt(id_vars="month", var_name="rank_type", value_name="rank")
            industry_name = industry_group_label(industry_rank["industry_group"].dropna().iloc[-1])
            line_chart(rank_long, "month", "rank", f"{company_name} {industry_name} Peer Ranks", "rank_type")
        else:
            show_empty_state("No industry peer rank is available for the selected company.")

        if not industry_rank.empty:
            st.subheader("產業同業排名")
            rank_table_cols = [
                "month",
                "industry_group",
                "industry_peer_count",
                "industry_revenue_yoy_rank",
                "industry_monthly_return_rank",
                "industry_health_score_rank",
                "industry_volatility_rank",
            ]
            rank_table = industry_rank[[col for col in rank_table_cols if col in industry_rank.columns]].sort_values("month", ascending=False)
            rank_table = rank_table.rename(
                columns={
                    "month": "月份",
                    "industry_group": "產業",
                    "industry_peer_count": "同業家數",
                    "industry_revenue_yoy_rank": "營收 YoY 排名",
                    "industry_monthly_return_rank": "月報酬排名",
                    "industry_health_score_rank": "健康分數排名",
                    "industry_volatility_rank": "波動率排名",
                }
            )
            integer_cols = ["同業家數", "營收 YoY 排名", "月報酬排名", "健康分數排名", "波動率排名"]
            for column in integer_cols:
                if column in rank_table.columns:
                    rank_table[column] = pd.to_numeric(rank_table[column], errors="coerce").astype("Int64")
            st.dataframe(centered_table(format_table(rank_table)), use_container_width=True, hide_index=True)

    with tabs[1]:
        st.subheader("月營收")
        revenue = filter_stock(revenue_growth, stock_id)
        revenue = filter_month_range(revenue, month_range)
        if revenue.empty:
            show_empty_state("No monthly revenue data is available for the selected company.")
        else:
            col1, col2 = st.columns(2)
            with col1:
                line_chart(revenue, "month", "revenue", f"{company_name} Monthly Revenue")
            with col2:
                line_chart(revenue, "month", "revenue_yoy", f"{company_name} Revenue YoY")
            st.dataframe(format_table(revenue.sort_values("month", ascending=False)), use_container_width=True, hide_index=True)

        st.subheader("股價與月營收關聯")
        relation_cols = [col for col in ["month", "close_price", "revenue_yoy", "revenue_mom"] if col in company.columns]
        relation = company[relation_cols].dropna(subset=["close_price"]) if "close_price" in relation_cols else pd.DataFrame()
        scatter_chart(relation, "revenue_yoy", "close_price", f"{company_name} 股價與月營收 YoY 關聯")

    with tabs[2]:
        st.subheader("月股價")
        company_price = filter_stock(monthly_stock_price_features(stock_features), stock_id)
        company_price = filter_month_range(company_price, month_range)
        if company_price.empty:
            show_empty_state("No monthly stock price data is available for the selected company.")
        else:
            col1, col2 = st.columns(2)
            with col1:
                line_chart(company_price, "month", "close_price", f"{company_name} Monthly Close Price")
            with col2:
                line_chart(company_price, "month", "volatility_20d", f"{company_name} Monthly 20D Volatility")
            line_chart(company_price, "month", "monthly_return", f"{company_name} 月報酬率")
            st.dataframe(format_table(company_price.sort_values("month", ascending=False)), use_container_width=True, hide_index=True)


def page_semiconductor_peer_comparison(peers: pd.DataFrame, month_range: tuple[str, str] | None, latest_price_date: str) -> None:
    page_header("Semiconductor Peer Comparison", latest_price_date)
    st.caption("Business interpretation: this view compares semiconductor companies by revenue momentum, market reaction, volatility, valuation, and health ranking.")
    df = filter_month_range(peers, month_range)
    if df.empty and not peers.empty:
        st.info("No semiconductor peer rows are available in the selected month range. Showing the latest available semiconductor peer data instead.")
        df = peers.copy()
    lm = latest_month(df)
    latest = df[df["month"].eq(lm)].copy() if lm else pd.DataFrame()
    if latest.empty:
        show_empty_state("No semiconductor peer comparison data is available for the selected period.")
        return

    col1, col2 = st.columns(2)
    with col1:
        bar_chart(latest.sort_values("revenue_yoy", ascending=False), "stock_name", "revenue_yoy", "Latest Revenue YoY by Semiconductor Peer", "risk_level")
    with col2:
        bar_chart(latest.sort_values("financial_health_score", ascending=False), "stock_name", "financial_health_score", "Latest Health Score by Semiconductor Peer", "risk_level")

    col3, col4 = st.columns(2)
    with col3:
        bar_chart(latest.sort_values("monthly_return", ascending=False), "stock_name", "monthly_return", "Latest Monthly Return by Semiconductor Peer", "risk_level")
    with col4:
        bar_chart(latest.sort_values("volatility_20d", ascending=False), "stock_name", "volatility_20d", "Latest 20D Volatility by Semiconductor Peer", "risk_level")

    cols = [
        "stock_id",
        "stock_name",
        "revenue_yoy",
        "monthly_return",
        "volatility_20d",
        "pe_ratio",
        "financial_health_score",
        "risk_level",
        "revenue_yoy_rank",
        "health_score_rank",
        "volatility_rank",
    ]
    st.subheader(f"Peer Table: {lm}")
    st.dataframe(format_table(latest[[col for col in cols if col in latest.columns]].sort_values("health_score_rank")), use_container_width=True, hide_index=True)


def page_revenue_growth(revenue_growth: pd.DataFrame, stock_id: str, month_range: tuple[str, str] | None, industry_group: str, latest_price_date: str) -> None:
    page_header("Revenue Growth Analysis", latest_price_date)
    st.caption("Business interpretation: this page separates operating momentum from price movement, making it easier to find improving or deteriorating revenue trends.")
    df = filter_industry(filter_month_range(revenue_growth, month_range), industry_group)
    if df.empty:
        show_empty_state("No revenue growth data is available for the selected filters.")
        return

    selected = df[df["stock_id"].astype(str).eq(stock_id)].copy()
    col1, col2 = st.columns(2)
    with col1:
        line_chart(selected, "month", "revenue", "Selected Company Monthly Revenue")
    with col2:
        line_chart(selected, "month", "revenue_yoy", "Selected Company Revenue YoY")

    lm = latest_month(df)
    latest = df[df["month"].eq(lm)].copy() if lm else pd.DataFrame()
    col3, col4 = st.columns(2)
    with col3:
        bar_chart(latest.sort_values("revenue_yoy", ascending=False).head(10), "stock_name", "revenue_yoy", "Latest Revenue YoY Top 10", "revenue_growth_signal")
    with col4:
        bar_chart(latest.sort_values("revenue_yoy", ascending=True).head(10), "stock_name", "revenue_yoy", "Latest Revenue YoY Bottom 10", "revenue_growth_signal")

    st.subheader("Revenue Growth Signal Distribution")
    st.caption(
        "分析標準：依最新月份的 `revenue_growth_signal` 分類。"
        "`Strong Growth` 代表 YoY 明顯成長且 MoM 為正；`Improving` 代表 YoY 為正；"
        "`Declining` 代表 YoY 與 MoM 同時轉弱；`Mixed` 代表訊號不一致；`Unknown` 代表資料不足。"
    )
    signal_order = ["Strong Growth", "Improving", "Mixed", "Declining", "Unknown"]
    signal_summary = (
        latest.groupby("revenue_growth_signal", dropna=False)
        .agg(
            company_count=("stock_id", "nunique"),
            avg_revenue_yoy=("revenue_yoy", "mean"),
            median_revenue_yoy=("revenue_yoy", "median"),
            avg_revenue_mom=("revenue_mom", "mean"),
            total_revenue=("revenue", "sum"),
        )
        .reset_index()
    )
    signal_summary["revenue_growth_signal"] = signal_summary["revenue_growth_signal"].fillna("Unknown")
    total_companies = signal_summary["company_count"].sum()
    signal_summary["company_share"] = signal_summary["company_count"] / total_companies if total_companies else 0
    signal_summary["sort_order"] = signal_summary["revenue_growth_signal"].map({label: idx for idx, label in enumerate(signal_order)}).fillna(99)
    signal_summary = signal_summary.sort_values("sort_order")

    metric_cols = st.columns(4)
    for idx, label in enumerate(["Strong Growth", "Improving", "Declining", "Unknown"]):
        count = signal_summary.loc[signal_summary["revenue_growth_signal"].eq(label), "company_count"]
        metric_cols[idx].metric(label, f"{int(count.iloc[0]) if not count.empty else 0:,}")

    col5, col6 = st.columns(2)
    with col5:
        horizontal_bar_chart(signal_summary, "company_count", "revenue_growth_signal", "Company Count by Revenue Growth Signal", "revenue_growth_signal")
    with col6:
        horizontal_bar_chart(signal_summary, "avg_revenue_yoy", "revenue_growth_signal", "Average Revenue YoY by Signal", "revenue_growth_signal")

    st.subheader("Signal Quality Summary")
    summary_table = signal_summary.drop(columns=["sort_order"])
    st.dataframe(format_table(summary_table), use_container_width=True, hide_index=True)

    st.subheader("Companies by Signal")
    signal_filter = st.selectbox(
        "Revenue Growth Signal",
        [label for label in signal_order if label in set(latest["revenue_growth_signal"].fillna("Unknown"))],
        key="revenue_signal_filter",
    )
    signal_companies = latest[latest["revenue_growth_signal"].fillna("Unknown").eq(signal_filter)].copy()
    signal_cols = ["stock_id", "stock_name", "industry_group", "revenue", "revenue_yoy", "revenue_mom", "revenue_growth_signal"]
    st.dataframe(
        format_table(signal_companies[[col for col in signal_cols if col in signal_companies.columns]].sort_values("revenue_yoy", ascending=False)),
        use_container_width=True,
        hide_index=True,
    )


def page_risk_monitoring(snapshot: pd.DataFrame, month_range: tuple[str, str] | None, industry_group: str, latest_price_date: str) -> None:
    page_header("Risk Monitoring", latest_price_date)
    st.caption("Business interpretation: this page highlights companies with weak revenue, price trend deterioration, high volatility, or low health scores for follow-up review.")
    df = filter_industry(filter_month_range(snapshot, month_range), industry_group)
    lm = latest_month(df)
    latest = apply_monitoring_risk_levels(latest_record_per_stock(df))
    if latest.empty:
        show_empty_state("No risk monitoring data is available for the selected filters.")
        return

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Watch Companies")
        st.caption(
            "篩選標準：每檔公司取篩選區間內最新一筆資料，且 `risk_level = Watch`。"
            "此等級來自較嚴格門檻：Health Score 低於 65，或落在目前篩選範圍分數最低的 15%。"
        )
        st.dataframe(format_table(latest[latest["risk_level"].eq("Watch")].sort_values("financial_health_score")), use_container_width=True, hide_index=True)
    with col2:
        st.subheader("High Companies")
        st.caption(
            "篩選標準：每檔公司取篩選區間內最新一筆資料，且 `risk_level = High`。"
            "此等級來自較嚴格門檻：Health Score 介於 65 到 74，或落在 Watch 之後的低分 25%。"
        )
        st.dataframe(format_table(latest[latest["risk_level"].eq("High")].sort_values("financial_health_score")), use_container_width=True, hide_index=True)

    st.subheader("Negative Revenue YoY and Below MA60")
    st.caption(
        "篩選標準：每檔公司取篩選區間內最新一筆資料，且 `revenue_yoy < 0`，同時 `price_above_ma60_flag = False`。"
        "這代表營收年增率為負，且股價低於 60 日均線。"
    )
    weak_trend = latest[(latest["revenue_yoy"] < 0) & (latest["price_above_ma60_flag"] == False)]
    st.dataframe(
        format_table(weak_trend[["stock_id", "stock_name", "industry_group", "revenue_yoy", "monthly_return", "financial_health_score", "risk_level"]]),
        use_container_width=True,
        hide_index=True,
    )

    col3, col4 = st.columns(2)
    with col3:
        bar_chart(latest.sort_values("volatility_20d", ascending=False).head(10), "stock_name", "volatility_20d", "Highest 20D Volatility Top 10", "risk_level")
    with col4:
        risk_counts = latest["risk_level"].value_counts().reset_index()
        risk_counts.columns = ["risk_level", "company_count"]
        risk_counts["sort_order"] = risk_counts["risk_level"].map({label: idx for idx, label in enumerate(RISK_LEVEL_ORDER)}).fillna(99)
        risk_counts = risk_counts.sort_values("sort_order").drop(columns=["sort_order"])
        bar_chart(risk_counts, "risk_level", "company_count", "Latest Risk Level Distribution")


def main() -> None:
    required_tables = [
        "gold_company_monthly_snapshot",
        "gold_stock_price_features",
        "gold_revenue_growth",
        "gold_semiconductor_peer_comparison",
    ]

    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Analysis Page",
        [
            "Executive Overview",
            "Company Analysis",
            "Semiconductor Peer Comparison",
            "Revenue Growth Analysis",
            "Risk Monitoring",
        ],
    )

    missing_data = show_missing_data_warning(required_tables)

    snapshot = normalize_types(load_gold_table("gold_company_monthly_snapshot"))
    operating_dashboard = normalize_types(load_gold_table("gold_operating_dashboard"))
    stock_features = normalize_types(load_gold_table("gold_stock_price_features"))
    revenue_growth = normalize_types(load_gold_table("gold_revenue_growth"))
    peers = normalize_types(load_gold_table("gold_semiconductor_peer_comparison"))

    configured_stock_ids = load_config_stock_ids()
    configured_stock_names = load_config_stock_names()
    universe_stock_ids = get_app_universe_stock_ids(operating_dashboard)
    if not configured_stock_ids:
        universe_stock_ids = keep_stocks_with_enough_data(
            universe_stock_ids,
            snapshot=snapshot,
            stock_features=stock_features,
            revenue_growth=revenue_growth,
        )
    snapshot = filter_app_universe(snapshot, universe_stock_ids)
    operating_dashboard = filter_app_universe(operating_dashboard, universe_stock_ids)
    stock_features = filter_app_universe(stock_features, universe_stock_ids)
    revenue_growth = filter_app_universe(revenue_growth, universe_stock_ids)
    latest_price_date = latest_price_date_label(stock_features, operating_dashboard)

    base_for_filters = operating_dashboard if not operating_dashboard.empty else (snapshot if not snapshot.empty else revenue_growth)
    stock_option_frames = [df for df in [base_for_filters, stock_features] if not df.empty and "stock_id" in df.columns]
    data_stock_options = sorted(
        pd.concat([df[["stock_id"]] for df in stock_option_frames], ignore_index=True)["stock_id"].dropna().astype(str).unique()
    ) if stock_option_frames else []
    if configured_stock_ids:
        stock_options = configured_stock_ids + [stock_id for stock_id in data_stock_options if stock_id not in configured_stock_ids]
    else:
        stock_options = data_stock_options
    default_stock_index = stock_options.index("2330") if "2330" in stock_options else 0
    stock_id = (
        st.sidebar.selectbox(
            "Company stock_id",
            stock_options,
            index=default_stock_index,
            format_func=lambda value: f"{value} {configured_stock_names.get(value, '')}".strip(),
        )
        if stock_options
        else ""
    )

    industry_options = ["All"]
    industry_frames = [df for df in [base_for_filters, stock_features] if not df.empty and "industry_group" in df.columns]
    if industry_frames:
        industry_values = pd.concat(
            [df[["industry_group"]] for df in industry_frames],
            ignore_index=True,
        )["industry_group"].dropna().astype(str).unique()
        industry_options += sorted(
            industry_values,
            key=industry_group_label,
        )
    industry_group = st.sidebar.selectbox(
        "Industry Group",
        industry_options,
        format_func=lambda value: "All Industries" if value == "All" else industry_group_label(value),
    )

    month_options = sorted(
        set(months_from_frame(operating_dashboard) + months_from_frame(stock_features) + months_from_frame(revenue_growth))
    )
    month_range = None
    if len(month_options) > 1:
        five_year_start = (pd.Timestamp.today() - pd.DateOffset(years=5)).to_period("M").strftime("%Y-%m")
        default_start_month = next((month for month in month_options if month >= five_year_start), month_options[0])
        selected_months = st.sidebar.select_slider("Month Range", options=month_options, value=(default_start_month, month_options[-1]))
        month_range = selected_months if isinstance(selected_months, tuple) else (selected_months, selected_months)
    elif len(month_options) == 1:
        month_range = (month_options[0], month_options[0])
        st.sidebar.caption(f"Month Range: {month_options[0]}")

    st.sidebar.divider()
    st.sidebar.caption(f"Universe: Taiwan top coverage by latest trading money ({len(universe_stock_ids)} stocks)")
    st.sidebar.caption("Data source: TWSE OpenAPI")
    st.sidebar.caption("Disclaimer: This project is a data engineering and analytics side project. It is not investment advice.")

    if missing_data and all(df.empty for df in [snapshot, operating_dashboard, stock_features, revenue_growth, peers]):
        st.title(APP_TITLE)
        st.info("After generating gold tables, rerun or refresh this Streamlit app.")
        return

    if page == "Executive Overview":
        page_executive_overview(snapshot, month_range, industry_group, latest_price_date)
    elif page == "Company Analysis":
        page_company_analysis(
            snapshot,
            operating_dashboard,
            stock_features,
            revenue_growth,
            peers,
            stock_id,
            month_range,
            latest_price_date,
        )
    elif page == "Semiconductor Peer Comparison":
        page_semiconductor_peer_comparison(peers, month_range, latest_price_date)
    elif page == "Revenue Growth Analysis":
        page_revenue_growth(revenue_growth, stock_id, month_range, industry_group, latest_price_date)
    elif page == "Risk Monitoring":
        page_risk_monitoring(snapshot, month_range, industry_group, latest_price_date)


if __name__ == "__main__":
    main()
