from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize


def optimize_latest_portfolio(predictions: pd.DataFrame) -> pd.DataFrame:
    latest_date = predictions["date"].max()
    latest = predictions[predictions["date"] == latest_date].sort_values("ticker").copy()
    tickers = latest["ticker"].tolist()
    expected_returns = latest["predicted_return"].to_numpy()
    volatilities = latest["predicted_volatility"].to_numpy()
    covariance = np.diag(np.square(volatilities))

    def sharpe(weights: np.ndarray) -> float:
        portfolio_return = float(weights @ expected_returns)
        portfolio_vol = float(np.sqrt(weights @ covariance @ weights))
        if portfolio_vol <= 0:
            return -1e6
        return portfolio_return / portfolio_vol

    n_assets = len(tickers)
    initial = np.repeat(1 / n_assets, n_assets)
    rng = np.random.default_rng(42)
    candidates = np.vstack([initial, rng.dirichlet(np.ones(n_assets), size=5000)])
    weights = candidates[np.argmax([sharpe(candidate) for candidate in candidates])]

    return pd.DataFrame(
        {
            "date": latest_date,
            "ticker": tickers,
            "expected_return": expected_returns,
            "predicted_volatility": volatilities,
            "weight": weights,
        }
    )


def optimize_rebalancing_history(predictions: pd.DataFrame, days: int = 60) -> pd.DataFrame:
    dates = sorted(predictions["date"].unique())[-days:]
    rows = []

    for date in dates:
        day_predictions = predictions[predictions["date"] == date]
        if day_predictions.empty:
            continue
        weights = optimize_latest_portfolio(day_predictions)
        rows.append(weights)

    if not rows:
        return pd.DataFrame(columns=["date", "ticker", "expected_return", "predicted_volatility", "weight"])

    return pd.concat(rows, ignore_index=True)


def efficient_frontier(predictions: pd.DataFrame, points: int = 40) -> pd.DataFrame:
    latest_date = predictions["date"].max()
    latest = predictions[predictions["date"] == latest_date].sort_values("ticker")
    returns = latest["predicted_return"].to_numpy()
    vols = latest["predicted_volatility"].to_numpy()
    covariance = np.diag(np.square(vols))
    n_assets = len(returns)
    initial = np.repeat(1 / n_assets, n_assets)
    bounds = tuple((0, 1) for _ in range(n_assets))

    def portfolio_risk(weights: np.ndarray) -> float:
        return float(np.sqrt(weights @ covariance @ weights))

    min_variance = minimize(
        portfolio_risk,
        initial,
        method="SLSQP",
        bounds=bounds,
        constraints=({"type": "eq", "fun": lambda weights: np.sum(weights) - 1},),
    )
    min_variance_weights = min_variance.x if min_variance.success else initial
    min_variance_return = float(min_variance_weights @ returns)
    max_return = float(returns.max())
    if max_return <= min_variance_return:
        max_return = float(np.percentile(returns, 95))

    targets = np.linspace(min_variance_return, max_return, points)
    rows = []

    for target in targets:
        result = minimize(
            portfolio_risk,
            initial,
            method="SLSQP",
            bounds=bounds,
            constraints=(
                {"type": "eq", "fun": lambda weights: np.sum(weights) - 1},
                {"type": "eq", "fun": lambda weights, target=target: float(weights @ returns - target)},
            ),
        )
        if result.success:
            rows.append({"target_return": target, "risk": portfolio_risk(result.x)})

    frontier = pd.DataFrame(rows).sort_values("target_return")
    return frontier.drop_duplicates(subset=["risk"]).reset_index(drop=True)
