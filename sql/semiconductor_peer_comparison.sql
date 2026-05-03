WITH latest_month AS (
    SELECT MAX(month) AS month
    FROM gold_semiconductor_peer_comparison
)
SELECT
    p.month,
    p.stock_id,
    p.stock_name,
    p.revenue_yoy,
    p.monthly_return,
    p.volatility_20d,
    p.pe_ratio,
    p.financial_health_score,
    p.risk_level,
    p.revenue_yoy_rank,
    p.monthly_return_rank,
    p.health_score_rank,
    p.volatility_rank
FROM gold_semiconductor_peer_comparison p
JOIN latest_month lm ON p.month = lm.month
ORDER BY p.health_score_rank, p.revenue_yoy_rank;

