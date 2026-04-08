from __future__ import annotations

from ngboost import NGBRegressor
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
    if isinstance(estimator, BaseForecaster):
        return estimator
    return YfromX(estimator=estimator, pooling=pooling)
