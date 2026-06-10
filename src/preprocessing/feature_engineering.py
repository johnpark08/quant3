from __future__ import annotations

import numpy as np
import pandas as pd


def add_macro_features(macro: pd.DataFrame, window: int = 12) -> pd.DataFrame:
    macro = macro.sort_values("date").copy()
    value_columns = [column for column in macro.columns if column != "date"]

    for column in value_columns:
        macro[f"{column}_pct_change"] = macro[column].pct_change()
        macro[f"{column}_volatility"] = macro[f"{column}_pct_change"].rolling(window=window, min_periods=3).std()

    return macro


def add_price_features(prices: pd.DataFrame, volatility_window: int = 21) -> pd.DataFrame:
    prices = prices.sort_values(["ticker", "date"]).copy()
    grouped = prices.groupby("ticker", group_keys=False)
    prices["target_next_return"] = grouped["log_return"].shift(-1)
    prices["realized_volatility"] = grouped["log_return"].transform(
        lambda series: series.rolling(volatility_window, min_periods=5).std() * np.sqrt(252)
    )
    prices["target_next_volatility"] = grouped["realized_volatility"].shift(-1)
    return prices
