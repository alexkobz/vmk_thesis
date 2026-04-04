from __future__ import annotations

from sklearn.ensemble import GradientBoostingRegressor
from sktime.forecasting.compose import YfromX


def build_forecaster(
    estimator=None,
    pooling: str = "global",
):
    if estimator is None:
        estimator = GradientBoostingRegressor(n_estimators=100, random_state=42)
    return YfromX(estimator=estimator, pooling=pooling)
