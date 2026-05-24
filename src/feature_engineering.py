import numpy as np
import pandas as pd
from statsmodels.nonparametric.smoothers_lowess import lowess

from logs.logger import logger
from src.utils.utils import show_shape


@show_shape
def add_log_returns(df: pd.DataFrame, features: dict, col_prefix: str) -> pd.DataFrame:
    df = df.copy()
    for col, lags in features.items():
        for lag in lags:
            col_name = f"{col_prefix}_{col}_{lag}"
            df[col_name] = (
                df[col].groupby(level='secid')
                .apply(lambda x: np.log(x).diff(lag))
                .reset_index(level=0, drop=True)
            )
    return df


@show_shape
def lowess_smooth(df: pd.DataFrame, cols: list[str], col_prefix: str, frac=0.2) -> pd.DataFrame:
    df = df.copy()
    smoothed_cols: dict[str, pd.Series] = {}
    for col in cols:
        s = df[col]
        col_name = f"{col_prefix}_{col}"
        if isinstance(s.index, pd.MultiIndex):
            date_level_idx = s.index.names.index('tradedate')
            group_levels = [i for i in range(s.index.nlevels) if i != date_level_idx]

            def _apply(group):
                g = group.droplevel(group_levels) if group_levels else group
                g_filled = g.sort_index().fillna(0)
                smoothed = lowess(
                    g_filled.to_numpy(dtype=float),
                    g_filled.index,
                    frac=frac,
                    return_sorted=False,
                )
                out = pd.Series(smoothed, index=g_filled.index).reindex(g.index)
                out.index = group.index
                return out
            smoothed_cols[col_name] = s.groupby(level=group_levels, group_keys=False).apply(_apply)
            logger.info(f"Applied lowess smoothing to {s.name}")
    if smoothed_cols:
        df = pd.concat([df, pd.DataFrame(smoothed_cols, index=df.index)], axis=1)
    return df
