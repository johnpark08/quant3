from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from src.models.garch_midas import predict_garch_midas


def _feature_columns(dataset: pd.DataFrame) -> list[str]:
    blocked = {
        "date",
        "ticker",
        "target_next_return",
        "target_next_volatility",
        "open",
        "high",
        "low",
        "close",
        "adj_close",
        "volume",
    }
    return [
        column
        for column in dataset.columns
        if column not in blocked and pd.api.types.is_numeric_dtype(dataset[column])
    ]


def _fit_ridge(x: pd.DataFrame, y: pd.Series, alpha: float = 1.0) -> tuple[np.ndarray, pd.Series, pd.Series]:
    means = x.mean()
    stds = x.std().replace(0, 1)
    x_scaled = ((x - means) / stds).to_numpy()
    design = np.column_stack([np.ones(len(x_scaled)), x_scaled])
    penalty = np.eye(design.shape[1]) * alpha
    penalty[0, 0] = 0
    coefficients = np.linalg.pinv(design.T @ design + penalty) @ design.T @ y.to_numpy()
    return coefficients, means, stds


def _predict_ridge(x: pd.DataFrame, model: tuple[np.ndarray, pd.Series, pd.Series]) -> np.ndarray:
    coefficients, means, stds = model
    x_scaled = ((x - means) / stds).to_numpy()
    design = np.column_stack([np.ones(len(x_scaled)), x_scaled])
    return design @ coefficients


def _mse(actual: pd.Series, predicted: np.ndarray) -> float:
    return float(np.mean(np.square(actual.to_numpy() - predicted)))


def _mae(actual: pd.Series, predicted: np.ndarray) -> float:
    return float(np.mean(np.abs(actual.to_numpy() - predicted)))


def _qlike(actual_volatility: pd.Series, predicted_volatility: np.ndarray) -> float:
    actual_variance = np.maximum(np.square(actual_volatility.to_numpy()), 1e-8)
    predicted_variance = np.maximum(np.square(predicted_volatility), 1e-8)
    return float(np.mean(np.log(predicted_variance) + actual_variance / predicted_variance))


def _append_metrics(
    metrics: list[dict[str, float | str]],
    model_name: str,
    ticker: str,
    actual_return: pd.Series,
    predicted_return: np.ndarray,
    actual_volatility: pd.Series,
    predicted_volatility: np.ndarray,
) -> None:
    metrics.append(
        {
            "model": model_name,
            "ticker": ticker,
            "return_mse": _mse(actual_return, predicted_return),
            "return_mae": _mae(actual_return, predicted_return),
            "volatility_mse": _mse(actual_volatility, predicted_volatility),
            "volatility_mae": _mae(actual_volatility, predicted_volatility),
            "volatility_qlike": _qlike(actual_volatility, predicted_volatility),
        }
    )


def _append_prediction_frame(
    predictions: list[pd.DataFrame],
    test: pd.DataFrame,
    model_name: str,
    predicted_return: np.ndarray,
    predicted_volatility: np.ndarray,
) -> None:
    model_test = test.copy()
    model_test["model"] = model_name
    model_test["predicted_return"] = predicted_return
    model_test["predicted_volatility"] = predicted_volatility
    model_test["short_run_volatility"] = np.nan
    model_test["long_run_macro_component"] = np.nan
    predictions.append(
        model_test[
            [
                "date",
                "ticker",
                "model",
                "target_next_return",
                "target_next_volatility",
                "predicted_return",
                "predicted_volatility",
                "short_run_volatility",
                "long_run_macro_component",
            ]
        ]
    )


def train_predict(dataset: pd.DataFrame, test_size: float = 0.2, random_state: int = 42) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    predictions = []
    metrics = []
    tuning_rows = []

    for ticker, ticker_data in dataset.groupby("ticker"):
        ticker_data = ticker_data.sort_values("date").reset_index(drop=True)
        split_at = max(1, int(len(ticker_data) * (1 - test_size)))
        train = ticker_data.iloc[:split_at]
        test = ticker_data.iloc[split_at:].copy()
        features = _feature_columns(ticker_data)

        if test.empty or len(train) < 30:
            continue

        x_train = train[features].replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0)
        x_test = test[features].replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0)

        ridge_return_model = _fit_ridge(x_train, train["target_next_return"])
        ridge_vol_model = _fit_ridge(x_train, train["target_next_volatility"])
        ridge_return_pred = _predict_ridge(x_test, ridge_return_model)
        ridge_vol_pred = np.maximum(_predict_ridge(x_test, ridge_vol_model), 0.0001)

        _append_metrics(
            metrics,
            "Ridge",
            ticker,
            test["target_next_return"],
            ridge_return_pred,
            test["target_next_volatility"],
            ridge_vol_pred,
        )
        _append_prediction_frame(predictions, test, "Ridge", ridge_return_pred, ridge_vol_pred)

        sklearn_models = {
            "RandomForest": (
                RandomForestRegressor(n_estimators=300, min_samples_leaf=5, random_state=random_state, n_jobs=-1),
                RandomForestRegressor(n_estimators=300, min_samples_leaf=5, random_state=random_state + 1, n_jobs=-1),
            ),
            "ExtraTrees": (
                ExtraTreesRegressor(n_estimators=300, min_samples_leaf=5, random_state=random_state + 2, n_jobs=-1),
                ExtraTreesRegressor(n_estimators=300, min_samples_leaf=5, random_state=random_state + 3, n_jobs=-1),
            ),
            "GradientBoosting": (
                GradientBoostingRegressor(n_estimators=250, learning_rate=0.04, max_depth=3, random_state=random_state + 4),
                GradientBoostingRegressor(n_estimators=250, learning_rate=0.04, max_depth=3, random_state=random_state + 5),
            ),
            "SVR": (
                make_pipeline(StandardScaler(), SVR(C=1.0, epsilon=0.01, gamma="scale")),
                make_pipeline(StandardScaler(), SVR(C=1.0, epsilon=0.01, gamma="scale")),
            ),
            "KNN": (
                make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=15, weights="distance")),
                make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=15, weights="distance")),
            ),
        }

        for model_name, (return_model, vol_model) in sklearn_models.items():
            return_model.fit(x_train, train["target_next_return"])
            vol_model.fit(x_train, train["target_next_volatility"])
            model_return_pred = return_model.predict(x_test)
            model_vol_pred = np.maximum(vol_model.predict(x_test), 0.0001)

            _append_metrics(
                metrics,
                model_name,
                ticker,
                test["target_next_return"],
                model_return_pred,
                test["target_next_volatility"],
                model_vol_pred,
            )
            _append_prediction_frame(predictions, test, model_name, model_return_pred, model_vol_pred)

        gm_return_pred, gm_vol_pred, gm_short_vol, gm_long_macro, gm_params = predict_garch_midas(train, test)
        tuning_rows.append({"ticker": ticker, **gm_params})
        _append_metrics(
            metrics,
            "GARCH-MIDAS",
            ticker,
            test["target_next_return"],
            gm_return_pred,
            test["target_next_volatility"],
            gm_vol_pred,
        )

        gm_test = test.copy()
        gm_test["model"] = "GARCH-MIDAS"
        gm_test["predicted_return"] = gm_return_pred
        gm_test["predicted_volatility"] = gm_vol_pred
        gm_test["short_run_volatility"] = gm_short_vol
        gm_test["long_run_macro_component"] = gm_long_macro
        predictions.append(
            gm_test[
                [
                    "date",
                    "ticker",
                    "model",
                    "target_next_return",
                    "target_next_volatility",
                    "predicted_return",
                    "predicted_volatility",
                    "short_run_volatility",
                    "long_run_macro_component",
                ]
            ]
        )

    if not predictions:
        raise RuntimeError("No predictions were generated. Check dataset size and missing values.")

    return pd.concat(predictions, ignore_index=True), pd.DataFrame(metrics), pd.DataFrame(tuning_rows)
