from src.transform.health_score import calculate_financial_health_score, map_risk_level


def test_health_score_is_bounded_and_maps_risk():
    result = calculate_financial_health_score(
        revenue_yoy=0.3,
        revenue_mom=0.1,
        pe_ratio=15,
        price_above_ma60_flag=True,
        volatility_20d=0.02,
        dividend_yield=2.0,
    )
    assert 0 <= result.score <= 100
    assert result.risk_level == map_risk_level(result.score)


def test_risk_level_mapping():
    assert map_risk_level(80) == "Low Risk"
    assert map_risk_level(60) == "Moderate Risk"
    assert map_risk_level(40) == "Watch"
    assert map_risk_level(39) == "High Risk"

