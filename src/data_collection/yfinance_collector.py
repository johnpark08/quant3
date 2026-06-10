from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd


def _fallback_prices(tickers: list[str], start_date: str, end_date: str | None) -> pd.DataFrame:
    end = pd.Timestamp(end_date or date.today())
    dates = pd.bdate_range(start=start_date, end=end)
    rng = np.random.default_rng(42)
    frames = []

    for idx, ticker in enumerate(tickers):
        returns = rng.normal(loc=0.0005, scale=0.025 + idx * 0.004, size=len(dates))
        close = 100 * np.exp(np.cumsum(returns))
        volume = rng.integers(20_000_000, 180_000_000, size=len(dates))
        frame = pd.DataFrame(
            {
                "date": dates,
                "ticker": ticker,
                "open": close * (1 + rng.normal(0, 0.003, len(dates))),
                "high": close * (1 + rng.uniform(0.001, 0.02, len(dates))),
                "low": close * (1 - rng.uniform(0.001, 0.02, len(dates))),
                "close": close,
                "adj_close": close,
                "volume": volume,
            }
        )
        frames.append(frame)

    return pd.concat(frames, ignore_index=True)


def collect_prices(tickers: list[str], start_date: str, end_date: str | None = None) -> pd.DataFrame:
    try:
        import yfinance as yf

        cache_dir = Path("data/cache/yfinance")
        cache_dir.mkdir(parents=True, exist_ok=True)
        try:
            yf.set_tz_cache_location(str(cache_dir))
        except AttributeError:
            pass

        downloaded = yf.download(
            tickers=tickers,
            start=start_date,
            end=end_date,
            auto_adjust=False,
            progress=False,
            group_by="ticker",
            threads=True,
        )
        if downloaded.empty:
            raise RuntimeError("yfinance returned an empty data frame.")

        frames = []
        for ticker in tickers:
            ticker_frame = downloaded[ticker].copy() if len(tickers) > 1 else downloaded.copy()
            ticker_frame = ticker_frame.reset_index()
            ticker_frame.columns = [str(column).lower().replace(" ", "_") for column in ticker_frame.columns]
            ticker_frame["ticker"] = ticker
            frames.append(ticker_frame)

        prices = pd.concat(frames, ignore_index=True)
        prices = prices.rename(columns={"adj_close": "adj_close"})
        prices["date"] = pd.to_datetime(prices["date"])
        columns = ["date", "ticker", "open", "high", "low", "close", "adj_close", "volume"]
        prices = prices[[column for column in columns if column in prices.columns]]
        return prices.sort_values(["ticker", "date"]).reset_index(drop=True)
    except Exception as exc:
        print(f"[WARN] yfinance collection failed; using generated sample prices. Reason: {exc}")
        return _fallback_prices(tickers, start_date, end_date)


def add_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    prices = prices.sort_values(["ticker", "date"]).copy()
    prices["log_return"] = prices.groupby("ticker")["adj_close"].transform(lambda series: np.log(series).diff())
    return prices.dropna(subset=["log_return"]).reset_index(drop=True)
