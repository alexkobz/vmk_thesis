from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class ForecastConfig:
    """Configuration for forecasting runs.

    estimator can be provided directly (an instantiated estimator), or by name using
    """

    model: Any | None = None
    run_name: str = None
    data_reader: Callable | None = None
