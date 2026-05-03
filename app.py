from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

import pandas as pd
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv


APP_TITLE = "Taiwan Stock Fundamental & Price Intelligence App"
PROJECT_ROOT = Path(__file__).resolve().parent
GOLD_DIR = PROJECT_ROOT / "data" / "gold"


st.set_page_config(page_title=APP_TITLE, layout="wide")


def get_finmind_token_status() -> str:
    load_dotenv(PROJECT_ROOT / ".env")
    token = os.getenv("FINMIND_TOKEN")
    try:
        token = token or st.secrets.get("FINMIND_TOKEN")
    except Exception:
        pass
    return "Configured" if token else "Not configured"


@st.cache_data(show_spinner=False)
def load_gold_table(table_name: str) -> pd.DataFrame:
    parquet_path = GOLD_DIR / f"{table_name}.parquet"
    csv_path = GOLD_DIR / f"{table_name}.csv"
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


def filter_industry(df: pd.DataFrame, industry_group: str) -> pd.DataFrame:
    if df.empty or industry_group == "All" or "industry_group" not in df.columns:
        return df
    return df[df["industry_group"].eq(industry_group)].copy()


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
    percent_cols = ["revenue_yoy", "revenue_mom", "monthly_return", "volatility_20d", "dividend_yield"]
    for column in percent_cols:
        if column in formatted.columns:
            formatted[column] = formatted[column].map(format_percent)
    for column in ["revenue", "close_price", "pe_ratio", "financial_health_score"]:
        if column in formatted.columns:
            formatted[column] = formatted[column].map(format_price if column in {"close_price", "pe_ratio"} else format_number)
    return formatted


def show_empty_state(message: str) -> None:
    st.info(message)


def bar_chart(df: pd.DataFrame, x: str, y: str, title: str, color: str | None = None) -> None:
    if df.empty or x not in df.columns or y not in df.columns:
        show_empty_state(f"No data available for {title}.")
        return
    fig = px.bar(df, x=x, y=y, color=color, title=title, text_auto=".2s")
    fig.update_layout(xaxis_title="", yaxis_title=y, height=420)
    st.plotly_chart(fig, use_container_width=True)


def line_chart(df: pd.DataFrame, x: str, y: str, title: str, color: str | None = None) -> None:
    if df.empty or x not in df.columns or y not in df.columns:
        show_empty_state(f"No data available for {title}.")
        return
    fig = px.line(df, x=x, y=y, color=color, markers=True, title=title)
    fig.update_layout(xaxis_title="", yaxis_title=y, height=420)
    st.plotly_chart(fig, use_container_width=True)


def page_executive_overview(snapshot: pd.DataFrame, month_range: tuple[str, str] | None, industry_group: str) -> None:
    st.title("Executive Overview")
    st.caption("Business interpretation: this page summarizes market-wide operating momentum, valuation-sensitive health scores, and current risk concentration across the selected coverage universe.")
    df = filter_industry(filter_month_range(snapshot, month_range), industry_group)
    lm = latest_month(df)
    latest = df[df["month"].eq(lm)].copy() if lm else pd.DataFrame()

    if latest.empty:
        show_empty_state("No company monthly snapshot data is available for the selected filters.")
        return

    metric_cols = st.columns(5)
    metric_cols[0].metric("Avg Health Score", format_number(latest["financial_health_score"].mean()))
    for idx, risk_level in enumerate(["Low Risk", "Moderate Risk", "Watch", "High Risk"], start=1):
        metric_cols[idx].metric(risk_level, f"{latest['risk_level'].eq(risk_level).sum():,}")

    st.subheader(f"Latest Month: {lm}")
    col1, col2 = st.columns(2)
    with col1:
        top_revenue = latest.sort_values("revenue_yoy", ascending=False).head(10)
        bar_chart(top_revenue, "stock_name", "revenue_yoy", "Revenue YoY Top 10", "risk_level")
    with col2:
        top_health = latest.sort_values("financial_health_score", ascending=False).head(10)
        bar_chart(top_health, "stock_name", "financial_health_score", "Financial Health Score Top 10", "risk_level")

    st.subheader("High Risk Companies")
    high_risk = latest[latest["risk_level"].isin(["High Risk", "Watch"])].sort_values("financial_health_score")
    st.dataframe(
        format_table(high_risk[["stock_id", "stock_name", "industry_group", "revenue_yoy", "monthly_return", "volatility_20d", "financial_health_score", "risk_level"]]),
        use_container_width=True,
        hide_index=True,
    )


def page_company_analysis(
    snapshot: pd.DataFrame,
    stock_features: pd.DataFrame,
    peers: pd.DataFrame,
    stock_id: str,
    month_range: tuple[str, str] | None,
) -> None:
    st.title("Company Analysis")
    st.caption("Business interpretation: this page connects a selected company's revenue cycle, price trend, health score, and peer ranking when industry peer data is available.")
    company = snapshot[snapshot["stock_id"].astype(str).eq(stock_id)].copy() if not snapshot.empty and stock_id else pd.DataFrame()
    company = filter_month_range(company, month_range)
    if company.empty:
        show_empty_state("No company snapshot data is available for the selected company. Please run the ETL workflow first.")
        return

    company_name = company["stock_name"].dropna().iloc[-1] if "stock_name" in company.columns and company["stock_name"].notna().any() else stock_id
    latest = company.sort_values("month").tail(1).iloc[0]
    cols = st.columns(4)
    cols[0].metric("Latest Close Price", format_price(latest.get("close_price")))
    cols[1].metric("Latest Revenue YoY", format_percent(latest.get("revenue_yoy")))
    cols[2].metric("Health Score", format_number(latest.get("financial_health_score")))
    cols[3].metric("Risk Level", str(latest.get("risk_level", "N/A")))

    col1, col2 = st.columns(2)
    with col1:
        line_chart(company, "month", "revenue", f"{company_name} Monthly Revenue Trend")
    with col2:
        company_price = stock_features[stock_features["stock_id"].astype(str).eq(stock_id)].copy()
        if not company_price.empty:
            company_price["month"] = pd.to_datetime(company_price["date"], errors="coerce").dt.to_period("M").astype(str)
            company_price = filter_month_range(company_price, month_range)
        line_chart(company_price, "date", "close_price", f"{company_name} Stock Price Trend")

    col3, col4 = st.columns(2)
    with col3:
        line_chart(company, "month", "financial_health_score", f"{company_name} Financial Health Score Trend")
    with col4:
        rank_df = peers[peers["stock_id"].astype(str).eq(stock_id)].copy() if not peers.empty and stock_id else pd.DataFrame()
        rank_cols = ["month", "revenue_yoy_rank", "health_score_rank", "volatility_rank"]
        rank_df = rank_df[[col for col in rank_cols if col in rank_df.columns]]
        if not rank_df.empty:
            rank_df = filter_month_range(rank_df, month_range)
            rank_long = rank_df.melt(id_vars="month", var_name="rank_type", value_name="rank")
            line_chart(rank_long, "month", "rank", f"{company_name} Semiconductor Peer Ranks", "rank_type")
        else:
            show_empty_state("No semiconductor peer rank is available for the selected company.")

    st.subheader("Company Snapshot")
    st.dataframe(format_table(company.sort_values("month", ascending=False)), use_container_width=True, hide_index=True)


def page_semiconductor_peer_comparison(peers: pd.DataFrame, month_range: tuple[str, str] | None) -> None:
    st.title("Semiconductor Peer Comparison")
    st.caption("Business interpretation: this view compares semiconductor companies by revenue momentum, market reaction, volatility, valuation, and health ranking.")
    df = filter_month_range(peers, month_range)
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


def page_revenue_growth(revenue_growth: pd.DataFrame, stock_id: str, month_range: tuple[str, str] | None, industry_group: str) -> None:
    st.title("Revenue Growth Analysis")
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
    signal_counts = latest["revenue_growth_signal"].value_counts().reset_index()
    signal_counts.columns = ["revenue_growth_signal", "company_count"]
    bar_chart(signal_counts, "revenue_growth_signal", "company_count", "Latest Revenue Growth Signal Distribution")


def page_risk_monitoring(snapshot: pd.DataFrame, month_range: tuple[str, str] | None, industry_group: str) -> None:
    st.title("Risk Monitoring")
    st.caption("Business interpretation: this page highlights companies with weak revenue, price trend deterioration, high volatility, or low health scores for follow-up review.")
    df = filter_industry(filter_month_range(snapshot, month_range), industry_group)
    lm = latest_month(df)
    latest = df[df["month"].eq(lm)].copy() if lm else pd.DataFrame()
    if latest.empty:
        show_empty_state("No risk monitoring data is available for the selected filters.")
        return

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("High Risk Companies")
        st.dataframe(format_table(latest[latest["risk_level"].eq("High Risk")].sort_values("financial_health_score")), use_container_width=True, hide_index=True)
    with col2:
        st.subheader("Watch Companies")
        st.dataframe(format_table(latest[latest["risk_level"].eq("Watch")].sort_values("financial_health_score")), use_container_width=True, hide_index=True)

    st.subheader("Negative Revenue YoY and Below MA60")
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
    stock_features = normalize_types(load_gold_table("gold_stock_price_features"))
    revenue_growth = normalize_types(load_gold_table("gold_revenue_growth"))
    peers = normalize_types(load_gold_table("gold_semiconductor_peer_comparison"))

    base_for_filters = snapshot if not snapshot.empty else revenue_growth
    stock_options = sorted(base_for_filters["stock_id"].dropna().astype(str).unique()) if "stock_id" in base_for_filters.columns else []
    stock_id = st.sidebar.selectbox("Company stock_id", stock_options) if stock_options else ""

    industry_options = ["All"]
    if "industry_group" in base_for_filters.columns and not base_for_filters.empty:
        industry_options += sorted(base_for_filters["industry_group"].dropna().astype(str).unique())
    industry_group = st.sidebar.selectbox("Industry Group", industry_options)

    month_options = sorted(base_for_filters["month"].dropna().astype(str).unique()) if "month" in base_for_filters.columns else []
    month_range = None
    if month_options:
        selected_months = st.sidebar.select_slider("Month Range", options=month_options, value=(month_options[0], month_options[-1]))
        month_range = selected_months if isinstance(selected_months, tuple) else (selected_months, selected_months)

    st.sidebar.divider()
    st.sidebar.caption("Data source: FinMind API")
    st.sidebar.caption(f"FINMIND_TOKEN: {get_finmind_token_status()}")
    st.sidebar.caption("Disclaimer: This project is a data engineering and analytics side project. It is not investment advice.")

    if missing_data and all(df.empty for df in [snapshot, stock_features, revenue_growth, peers]):
        st.title(APP_TITLE)
        st.info("After generating gold tables, rerun or refresh this Streamlit app.")
        return

    if page == "Executive Overview":
        page_executive_overview(snapshot, month_range, industry_group)
    elif page == "Company Analysis":
        page_company_analysis(snapshot, stock_features, peers, stock_id, month_range)
    elif page == "Semiconductor Peer Comparison":
        page_semiconductor_peer_comparison(peers, month_range)
    elif page == "Revenue Growth Analysis":
        page_revenue_growth(revenue_growth, stock_id, month_range, industry_group)
    elif page == "Risk Monitoring":
        page_risk_monitoring(snapshot, month_range, industry_group)


if __name__ == "__main__":
    main()
