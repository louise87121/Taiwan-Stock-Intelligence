from __future__ import annotations

import pandas as pd

SILVER_STOCK_PRICE_COLUMNS = [
    "date",
    "stock_id",
    "open_price",
    "high_price",
    "low_price",
    "close_price",
    "price_change",
    "trading_volume",
    "trading_money",
    "transaction_count",
    "daily_return",
    "monthly_return",
    "ma_20",
    "ma_60",
    "volatility_20d",
    "price_above_ma20_flag",
    "price_above_ma60_flag",
]


def _parse_twse_date(value: object) -> pd.Timestamp:
    text = str(value).strip()
    if text.isdigit() and len(text) == 7:
        year = int(text[:3]) + 1911
        return pd.Timestamp(year=year, month=int(text[3:5]), day=int(text[5:7]))
    return pd.to_datetime(value, errors="coerce")


def transform_stock_price(raw_df: pd.DataFrame) -> pd.DataFrame:
    if raw_df.empty:
        return pd.DataFrame(columns=SILVER_STOCK_PRICE_COLUMNS)

    df = raw_df.copy()
    rename_map = {
        "stock_id": "stock_id",
        "Code": "stock_id",
        "證券代號": "stock_id",
        "date": "date",
        "snapshot_date": "date",
        "Trading_Volume": "trading_volume",
        "TradeVolume": "trading_volume",
        "成交股數": "trading_volume",
        "Trading_money": "trading_money",
        "TradeValue": "trading_money",
        "成交金額": "trading_money",
        "Transaction": "transaction_count",
        "成交筆數": "transaction_count",
        "open": "open_price",
        "OpeningPrice": "open_price",
        "開盤價": "open_price",
        "max": "high_price",
        "HighestPrice": "high_price",
        "最高價": "high_price",
        "min": "low_price",
        "LowestPrice": "low_price",
        "最低價": "low_price",
        "close": "close_price",
        "ClosingPrice": "close_price",
        "收盤價": "close_price",
        "Change": "price_change",
        "漲跌價差": "price_change",
    }
    df = df.rename(columns=rename_map)
    for column in ["date", "stock_id", "open_price", "high_price", "low_price", "close_price", "price_change", "trading_volume", "trading_money", "transaction_count"]:
        if column not in df.columns:
            df[column] = pd.NA

    df["date"] = df["date"].map(_parse_twse_date)
    df["stock_id"] = df["stock_id"].astype(str)
    numeric_cols = ["open_price", "high_price", "low_price", "close_price", "price_change", "trading_volume", "trading_money", "transaction_count"]
    for column in numeric_cols:
        df[column] = (
            df[column]
            .astype(str)
            .str.replace(",", "", regex=False)
            .str.replace("+", "", regex=False)
            .replace({"--": pd.NA, "-": pd.NA, "nan": pd.NA, "": pd.NA})
        )
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
    df = df.dropna(subset=["date", "stock_id"])
    df = df.drop_duplicates(["stock_id", "date"], keep="last")
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
