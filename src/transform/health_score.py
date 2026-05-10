from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class HealthScoreResult:
    score: int
    risk_level: str


def map_risk_level(score: int | float) -> str:
    if score >= 85:
        return "Low"
    if score >= 75:
        return "Mid"
    if score >= 65:
        return "High"
    return "Watch"


def calculate_financial_health_score(
    revenue_yoy: float | None = None,
    revenue_mom: float | None = None,
    pe_ratio: float | None = None,
    price_above_ma60_flag: bool | None = None,
    volatility_20d: float | None = None,
    dividend_yield: float | None = None,
) -> HealthScoreResult:
    score = 50
    if pd.notna(revenue_yoy):
        if revenue_yoy > 0.20:
            score += 20
        elif revenue_yoy >= 0.05:
            score += 10
        elif revenue_yoy >= 0:
            score += 5
        else:
            score -= 15

    if pd.notna(revenue_mom):
        if revenue_mom > 0.05:
            score += 5
        elif revenue_mom < -0.05:
            score -= 5

    if price_above_ma60_flag is True:
        score += 10
    elif price_above_ma60_flag is False:
        score -= 5

    if pd.notna(volatility_20d):
        if volatility_20d > 0.05:
            score -= 10
        elif volatility_20d >= 0.03:
            score -= 5
        elif volatility_20d < 0.03:
            score += 5

    if pd.notna(pe_ratio):
        if 8 <= pe_ratio <= 25:
            score += 10
        elif pe_ratio > 40:
            score -= 10

    if pd.notna(dividend_yield) and dividend_yield > 0:
        score += 5

    score = max(0, min(100, int(round(score))))
    return HealthScoreResult(score=score, risk_level=map_risk_level(score))


def append_health_scores(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        df["financial_health_score"] = pd.Series(dtype="int64")
        df["risk_level"] = pd.Series(dtype="object")
        return df
    scored = df.copy()
    results = [
        calculate_financial_health_score(
            revenue_yoy=row.get("revenue_yoy"),
            revenue_mom=row.get("revenue_mom"),
            pe_ratio=row.get("pe_ratio"),
            price_above_ma60_flag=row.get("price_above_ma60_flag"),
            volatility_20d=row.get("volatility_20d"),
            dividend_yield=row.get("dividend_yield"),
        )
        for _, row in scored.iterrows()
    ]
    scored["financial_health_score"] = [result.score for result in results]
    scored["risk_level"] = [result.risk_level for result in results]
    return scored
