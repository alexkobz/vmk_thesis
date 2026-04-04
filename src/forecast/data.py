from __future__ import annotations

import pandas as pd

from src.forecast.config import ForecastConfig


def prepare_xy(df: pd.DataFrame, cfg: ForecastConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    drop = [cfg.y_name, *cfg.drop_cols]
    y = df[cfg.y_name].to_frame(cfg.y_alias)
    X = df.drop(columns=drop)
    return y, X
