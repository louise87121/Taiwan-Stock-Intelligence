import pandas as pd

from src.transform.monthly_revenue import transform_monthly_revenue


def test_revenue_yoy_is_calculated():
    raw = pd.DataFrame(
        {
            "date": pd.date_range("2023-01-01", periods=13, freq="MS").astype(str),
            "stock_id": ["2344"] * 13,
            "revenue": [100] * 12 + [125],
        }
    )
    result = transform_monthly_revenue(raw)
    assert result.loc[12, "revenue_yoy"] == 0.25
    assert result.loc[12, "revenue_growth_signal"] in {"Strong Growth", "Improving", "Mixed"}

