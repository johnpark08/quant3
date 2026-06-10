from __future__ import annotations

import pandas as pd


def build_training_dataset(prices: pd.DataFrame, macro: pd.DataFrame) -> pd.DataFrame:
    prices = prices.sort_values("date").copy()
    macro = macro.sort_values("date").copy()
    prices["date"] = pd.to_datetime(prices["date"]).dt.normalize().astype("datetime64[ns]")
    macro["date"] = pd.to_datetime(macro["date"]).dt.normalize().astype("datetime64[ns]")

    macro_daily = macro.set_index("date").resample("B").ffill().interpolate(method="linear").reset_index()
    macro_daily["date"] = pd.to_datetime(macro_daily["date"]).dt.normalize().astype("datetime64[ns]")
    dataset = pd.merge_asof(
        prices,
        macro_daily,
        on="date",
        direction="backward",
    )
    dataset = dataset.sort_values(["ticker", "date"])
    dataset = dataset.dropna(subset=["target_next_return", "target_next_volatility"])
    return dataset.reset_index(drop=True)
