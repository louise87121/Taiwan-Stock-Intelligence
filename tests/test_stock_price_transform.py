import pandas as pd
import pytest

from src.transform.stock_price import transform_stock_price


def test_stock_price_features_are_calculated():
    raw = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=65, freq="D").astype(str),
            "stock_id": ["2344"] * 65,
            "open": range(10, 75),
            "max": range(11, 76),
            "min": range(9, 74),
            "close": range(10, 75),
            "Trading_Volume": [1000] * 65,
            "Trading_money": [10000] * 65,
        }
    )
    result = transform_stock_price(raw)
    assert result.loc[1, "daily_return"] == pytest.approx(0.1)
    assert result.loc[19, "ma_20"] == 19.5
    assert pd.notna(result.loc[64, "ma_60"])
    assert pd.notna(result.loc[64, "volatility_20d"])
    february_return = result.loc[result["date"].eq("2024-02-01"), "monthly_return"].iloc[0]
    assert february_return == pytest.approx((69 - 40) / 40)
