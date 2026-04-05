from __future__ import annotations

from sklearn.ensemble import GradientBoostingRegressor

from ngboost import NGBRegressor
from sktime.forecasting.arima import ARIMA
from sktime.forecasting.base import BaseForecaster
from sktime.forecasting.compose import YfromX


def build_forecaster(
    estimator=None,
    pooling: str = "global",
):
    if estimator is None:
        estimator = NGBRegressor(
            n_estimators=100,
            learning_rate=0.05,
            random_state=42,
        )
    if isinstance(estimator, str):
        key = estimator.lower()
        if key in {"arima", "sktime_arima", "baseline_arima"}:
            return ARIMA(order=(1, 1, 1), seasonal_order=(0, 0, 0, 0), suppress_warnings=True)
        raise ValueError(f"Unknown estimator string: {estimator}")
    if isinstance(estimator, BaseForecaster):
        return estimator
    return YfromX(estimator=estimator, pooling=pooling)
