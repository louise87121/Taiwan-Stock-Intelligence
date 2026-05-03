SELECT
    month,
    stock_id,
    stock_name,
    close_price,
    monthly_return,
    revenue,
    revenue_mom,
    revenue_yoy,
    pe_ratio,
    dividend_yield,
    volatility_20d,
    financial_health_score,
    risk_level,
    semiconductor_revenue_yoy_rank,
    semiconductor_health_score_rank
FROM gold_winbond_snapshot
ORDER BY month DESC
LIMIT 24;

