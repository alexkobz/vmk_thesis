from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class EstimatorType(str, Enum):
    LINEAR = "LINEAR"
    TREE = "TREE"

@dataclass(frozen=True)
class ForecastConfig:
    estimator: object | None = None
    estimator_type: EstimatorType = EstimatorType.TREE
    pooling: str = "global"
    initial_window: int = 365 * 5
    horizon_days: int = 365
    step_length: int = 365
    ticker: str | None = None
    save_model: bool = True
    save_metrics: bool = True