from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, Optional

import pandas as pd


class EstimatorType(str, Enum):
    LINEAR = "LINEAR"
    TREE = "TREE"
    NGBOOST = "NGBOOST"


@dataclass(frozen=True)
class ForecastConfig:
    """Configuration for forecasting runs.

    estimator can be provided directly (an instantiated estimator), or by name using
    """

    estimator: Any | None = None
    estimator_type: EstimatorType = EstimatorType.TREE
    y: pd.Series | pd.DataFrame | None = None
    X: pd.Series | pd.DataFrame | None = None
    ticker: Optional[str] = None
    save_model: bool = True
    save_metrics: bool = True
