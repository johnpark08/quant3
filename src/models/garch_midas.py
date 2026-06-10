from __future__ import annotations

import numpy as np
import pandas as pd


MACRO_FEATURES = [
    "fed_funds_pct_change",
    "cpi_pct_change",
    "m2_pct_change",
    "ppi_pct_change",
    "industrial_production_pct_change",
    "wti_oil_pct_change",
    "usd_krw_pct_change",
    "dollar_index_pct_change",
    "vix_pct_change",
    "fed_funds_volatility",
    "cpi_volatility",
    "m2_volatility",
    "ppi_volatility",
    "industrial_production_volatility",
    "wti_oil_volatility",
    "usd_krw_volatility",
    "dollar_index_volatility",
    "vix_volatility",
]


def beta_weights(length: int, shape: float = 2.0) -> np.ndarray:
    positions = np.arange(1, length + 1, dtype=float)
    weights = (positions / length) ** (shape - 1)
    weights = weights / weights.sum()
    return weights


def _weighted_macro_component(frame: pd.DataFrame, lookback: int = 22, macro_weight: float = 0.15) -> pd.Series:
    available = [column for column in MACRO_FEATURES if column in frame.columns]
    if not available:
        return pd.Series(1.0, index=frame.index)

    macro = frame[available].replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0)
    standardized = (macro - macro.mean()) / macro.std().replace(0, 1)
    macro_pressure = standardized.abs().mean(axis=1)
    weights = beta_weights(lookback)

    long_run = macro_pressure.rolling(lookback, min_periods=3).apply(
        lambda values: float(np.dot(values, weights[-len(values) :])),
        raw=True,
    )
    long_run = long_run.ffill().bfill().fillna(0)
    return 1.0 + long_run.clip(lower=0) * macro_weight


def _predict_with_params(
    fit_frame: pd.DataFrame,
    predict_frame: pd.DataFrame,
    params: dict[str, float | int],
    volatility_floor: float = 0.0001,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    combined = pd.concat([fit_frame, predict_frame], ignore_index=True).sort_values("date").reset_index(drop=True)

    returns = combined["log_return"].replace([np.inf, -np.inf], np.nan).fillna(0)
    garch_variance = np.zeros(len(combined))
    initial_variance = float(np.nanvar(fit_frame["log_return"].tail(63))) if len(fit_frame) else 0.0001
    garch_variance[0] = max(initial_variance, volatility_floor**2)

    omega = float(params["omega"])
    alpha = float(params["alpha"])
    beta = float(params["beta"])
    for idx in range(1, len(combined)):
        garch_variance[idx] = omega + alpha * returns.iloc[idx - 1] ** 2 + beta * garch_variance[idx - 1]

    short_run_vol = np.sqrt(np.maximum(garch_variance, volatility_floor**2)) * np.sqrt(252)
    long_run_macro = _weighted_macro_component(
        combined,
        lookback=int(params["lookback"]),
        macro_weight=float(params["macro_weight"]),
    )
    volatility_scale = float(params.get("volatility_scale", 1.0))
    predicted_volatility = np.maximum(short_run_vol * long_run_macro.to_numpy() * volatility_scale, volatility_floor)

    return_window = int(params["return_window"])
    predicted_return = (
        combined["log_return"]
        .rolling(return_window, min_periods=3)
        .mean()
        .shift(1)
        .ffill()
        .bfill()
        .fillna(0)
        .to_numpy()
    )

    test_start = len(fit_frame)
    return (
        predicted_return[test_start:],
        predicted_volatility[test_start:],
        short_run_vol[test_start:],
        long_run_macro.to_numpy()[test_start:],
    )


def _parameter_grid() -> list[dict[str, float | int]]:
    grid = []
    for alpha in [0.05, 0.08, 0.12]:
        for beta in [0.84, 0.88, 0.92]:
            if alpha + beta >= 0.99:
                continue
            for macro_weight in [0.05, 0.10, 0.15, 0.20]:
                for lookback in [22, 44, 66]:
                    grid.append(
                        {
                            "omega": 0.000002,
                            "alpha": alpha,
                            "beta": beta,
                            "macro_weight": macro_weight,
                            "lookback": lookback,
                            "return_window": 21,
                            "volatility_scale": 1.0,
                        }
                    )
    return grid


def _tune_params(train: pd.DataFrame) -> tuple[dict[str, float | int], float]:
    validation_size = max(60, int(len(train) * 0.2))
    fit_frame = train.iloc[:-validation_size].copy()
    validation = train.iloc[-validation_size:].copy()
    if len(fit_frame) < 60 or validation.empty:
        default = {
            "omega": 0.000002,
            "alpha": 0.08,
            "beta": 0.88,
            "macro_weight": 0.10,
            "lookback": 22,
            "return_window": 21,
            "volatility_scale": 1.0,
        }
        return default, np.nan

    best_params = None
    best_mae = np.inf
    actual = validation["target_next_volatility"].to_numpy()
    for params in _parameter_grid():
        _, predicted_volatility, _, _ = _predict_with_params(fit_frame, validation, params)
        valid_mask = predicted_volatility > 0
        if valid_mask.any():
            scale = float(np.nanmedian(actual[valid_mask] / predicted_volatility[valid_mask]))
            scale = float(np.clip(scale, 0.5, 1.5))
        else:
            scale = 1.0
        scaled_volatility = predicted_volatility * scale
        mae = float(np.mean(np.abs(actual - scaled_volatility)))
        if mae < best_mae:
            best_mae = mae
            best_params = {**params, "volatility_scale": scale}

    if best_params is None:
        raise RuntimeError("GARCH-MIDAS tuning failed to select parameters.")
    return best_params, best_mae


def predict_garch_midas(
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, float | int]]:
    params, validation_mae = _tune_params(train)
    predicted_return, predicted_volatility, short_run_volatility, long_run_macro_component = _predict_with_params(
        train,
        test,
        params,
    )
    selected = dict(params)
    selected["validation_volatility_mae"] = validation_mae
    return predicted_return, predicted_volatility, short_run_volatility, long_run_macro_component, selected
