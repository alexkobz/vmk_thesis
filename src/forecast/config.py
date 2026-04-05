from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ForecastConfig:
    y_name: str
    cap_col: str = "dailycapitalization"
    y_alias: str = "log_return"
    drop_cols: tuple[str, ...] = ()
    horizon_days: int = 365
    initial_years: int = 5
    step_days: int = 365
    ticker: str | None = None
    estimator: object | None = None
    pooling: str = "global"
