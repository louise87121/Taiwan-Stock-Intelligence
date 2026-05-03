WITH latest_month AS (
    SELECT MAX(month) AS month
    FROM gold_revenue_growth
)
SELECT
    g.month,
    g.stock_id,
    g.stock_name,
    g.industry_group,
    g.revenue,
    g.revenue_yoy,
    g.revenue_mom,
    g.revenue_growth_signal
FROM gold_revenue_growth g
JOIN latest_month lm ON g.month = lm.month
ORDER BY g.revenue_yoy DESC NULLS LAST
LIMIT 10;

