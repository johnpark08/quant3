from __future__ import annotations

from src.data_collection.fred_collector import collect_macro
from src.data_collection.yfinance_collector import add_log_returns, collect_prices
from src.models.baseline_model import train_predict
from src.portfolio.optimizer import optimize_latest_portfolio
from src.preprocessing.align_frequency import build_training_dataset
from src.preprocessing.feature_engineering import add_macro_features, add_price_features
from src.utils import load_config, resolve_path


def run_pipeline() -> None:
    config = load_config()
    paths = config["paths"]

    prices = collect_prices(config["assets"], config["start_date"], config["end_date"])
    prices = add_log_returns(prices)
    prices.to_csv(resolve_path(paths["raw_prices"]), index=False)

    macro = collect_macro(config["macro_indicators"], config["start_date"], config["end_date"])
    macro.to_csv(resolve_path(paths["raw_macro"]), index=False)
    macro_sources = macro.attrs.get("sources")
    if macro_sources is not None:
        macro_sources.to_csv(resolve_path("data/raw/macro_sources.csv"), index=False)

    macro_features = add_macro_features(macro)
    price_features = add_price_features(prices, config["model"]["volatility_window"])
    dataset = build_training_dataset(price_features, macro_features)
    dataset.to_csv(resolve_path(paths["training_dataset"]), index=False)

    predictions, metrics, garch_midas_params = train_predict(
        dataset,
        test_size=config["model"]["test_size"],
        random_state=config["model"]["random_state"],
    )
    predictions.to_csv(resolve_path(paths["predictions"]), index=False)
    metrics.to_csv(resolve_path("data/predictions/model_metrics.csv"), index=False)
    garch_midas_params.to_csv(resolve_path("data/predictions/garch_midas_params.csv"), index=False)

    default_model = "GARCH-MIDAS"
    portfolio_predictions = predictions[predictions["model"] == default_model].copy()
    weights = optimize_latest_portfolio(portfolio_predictions)
    weights["model"] = default_model
    weights.to_csv(resolve_path(paths["portfolio_weights"]), index=False)

    print("Pipeline completed.")
    print(f"Training rows: {len(dataset):,}")
    print(f"Prediction rows: {len(predictions):,}")
    print("Latest portfolio weights:")
    print(weights[["ticker", "weight"]].to_string(index=False))


if __name__ == "__main__":
    run_pipeline()
