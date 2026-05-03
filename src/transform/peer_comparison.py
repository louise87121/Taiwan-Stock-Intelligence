from __future__ import annotations

import pandas as pd


PEER_COLUMNS = [
    "month",
    "stock_id",
    "stock_name",
    "revenue",
    "revenue_yoy",
    "monthly_return",
    "volatility_20d",
    "pe_ratio",
    "financial_health_score",
    "risk_level",
    "revenue_yoy_rank",
    "monthly_return_rank",
    "health_score_rank",
    "volatility_rank",
]


def build_semiconductor_peer_comparison(gold_company_monthly_snapshot: pd.DataFrame) -> pd.DataFrame:
    if gold_company_monthly_snapshot.empty:
        return pd.DataFrame(columns=PEER_COLUMNS)
    df = gold_company_monthly_snapshot.copy()
    df = df[df["industry_group"].eq("Semiconductor")].copy()
    if df.empty:
        return pd.DataFrame(columns=PEER_COLUMNS)
    df["revenue_yoy_rank"] = df.groupby("month")["revenue_yoy"].rank(ascending=False, method="min")
    df["monthly_return_rank"] = df.groupby("month")["monthly_return"].rank(ascending=False, method="min")
    df["health_score_rank"] = df.groupby("month")["financial_health_score"].rank(ascending=False, method="min")
    df["volatility_rank"] = df.groupby("month")["volatility_20d"].rank(ascending=True, method="min")
    return df[PEER_COLUMNS].sort_values(["month", "health_score_rank", "stock_id"])

