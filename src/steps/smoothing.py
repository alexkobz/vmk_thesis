import numpy as np
import pandas as pd
from statsmodels.nonparametric.smoothers_lowess import lowess


def lowess_smooth(s: pd.Series, frac=0.2, date_level="tradedate"):
    if isinstance(s.index, pd.MultiIndex):
        if date_level in s.index.names:
            date_level_idx = s.index.names.index(date_level)
        else:
            date_level_idx = -1

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

        return s.groupby(level=group_levels, group_keys=False).apply(_apply)

    s_filled = s.sort_index().fillna(0)
    smoothed = lowess(
        s_filled.to_numpy(dtype=float),
        s_filled.index,
        frac=frac,
        return_sorted=False,
    )
    return pd.Series(smoothed, index=s_filled.index).reindex(s.index)

def apply_ewm_multiindex(df: pd.DataFrame, cols: list[str], alpha: float = 0.2) -> pd.DataFrame:
    """
    Применяет ewm(alpha) по каждой группе 'secid' для указанных колонок.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame с MultiIndex ['secid', 'tradedate']
    cols : list[str]
        Список колонок для ewm
    alpha : float
        Параметр сглаживания ewm

    Returns
    -------
    pd.DataFrame
        Новый DataFrame с колонками <col>_ewm
    """
    df = df.copy()

    for col in cols:
        ewm_col = f"{col}_ewm"
        df[ewm_col] = df.groupby(level='secid')[col].transform(
            lambda x: x.ewm(alpha=alpha, adjust=False).mean()
        )

    return df
