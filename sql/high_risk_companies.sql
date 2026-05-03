WITH latest_month AS (
    SELECT MAX(month) AS month
    FROM gold_company_monthly_snapshot
)
SELECT
    g.month,
    g.stock_id,
    g.stock_name,
    g.industry_group,
    g.revenue_yoy,
    g.monthly_return,
    g.volatility_20d,
    g.financial_health_score,
    g.risk_level
FROM gold_company_monthly_snapshot g
JOIN latest_month lm ON g.month = lm.month
WHERE g.risk_level IN ('High Risk', 'Watch')
ORDER BY g.financial_health_score ASC, g.volatility_20d DESC NULLS LAST;

