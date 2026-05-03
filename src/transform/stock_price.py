from __future__ import annotations

import pandas as pd

SILVER_STOCK_PRICE_COLUMNS = [
    "date",
    "stock_id",
    "open_price",
    "high_price",
    "low_price",
    "close_price",
    "trading_volume",
    "trading_money",
    "daily_return",
    "monthly_return",
    "ma_20",
    "ma_60",
    "volatility_20d",
    "price_above_ma20_flag",
    "price_above_ma60_flag",
]


def transform_stock_price(raw_df: pd.DataFrame) -> pd.DataFrame:
    if raw_df.empty:
        return pd.DataFrame(columns=SILVER_STOCK_PRICE_COLUMNS)

    df = raw_df.copy()
    rename_map = {
        "stock_id": "stock_id",
        "date": "date",
        "Trading_Volume": "trading_volume",
        "Trading_money": "trading_money",
        "open": "open_price",
        "max": "high_price",
        "min": "low_price",
        "close": "close_price",
    }
    df = df.rename(columns=rename_map)
    for column in ["date", "stock_id", "open_price", "high_price", "low_price", "close_price", "trading_volume", "trading_money"]:
        if column not in df.columns:
            df[column] = pd.NA

    df["date"] = pd.to_datetime(df["date"])
    df["stock_id"] = df["stock_id"].astype(str)
    numeric_cols = ["open_price", "high_price", "low_price", "close_price", "trading_volume", "trading_money"]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
    df = df.sort_values(["stock_id", "date"])

    grouped = df.groupby("stock_id", group_keys=False)
    df["daily_return"] = grouped["close_price"].pct_change()
    df["ma_20"] = grouped["close_price"].transform(lambda s: s.rolling(20, min_periods=1).mean())
    df["ma_60"] = grouped["close_price"].transform(lambda s: s.rolling(60, min_periods=1).mean())
    df["volatility_20d"] = grouped["daily_return"].transform(lambda s: s.rolling(20, min_periods=2).std())
    df["month"] = df["date"].dt.to_period("M").astype(str)
    month_end_close = (
        df.sort_values("date")
        .groupby(["stock_id", "month"], as_index=False)
        .tail(1)[["stock_id", "month", "close_price"]]
        .sort_values(["stock_id", "month"])
    )
    month_end_close["monthly_return"] = month_end_close.groupby("stock_id")["close_price"].pct_change()
    df = df.merge(month_end_close[["stock_id", "month", "monthly_return"]], on=["stock_id", "month"], how="left")
    df["price_above_ma20_flag"] = df["close_price"] > df["ma_20"]
    df["price_above_ma60_flag"] = df["close_price"] > df["ma_60"]
    df["date"] = df["date"].dt.date.astype(str)
    return df[SILVER_STOCK_PRICE_COLUMNS]


def latest_monthly_price_features(silver_stock_price: pd.DataFrame) -> pd.DataFrame:
    if silver_stock_price.empty:
        return pd.DataFrame()
    df = silver_stock_price.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["month"] = df["date"].dt.to_period("M").astype(str)
    idx = df.sort_values("date").groupby(["stock_id", "month"]).tail(1).index
    return df.loc[idx].copy()
