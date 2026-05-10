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
    assert map_risk_level(85) == "Low"
    assert map_risk_level(75) == "Mid"
    assert map_risk_level(65) == "High"
    assert map_risk_level(64) == "Watch"
