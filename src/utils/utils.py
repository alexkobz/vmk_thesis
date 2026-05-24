from typing import Callable

import numpy as np
import pandas as pd

from logs.logger import logger


def clip_cap_outliers(
    s: pd.Series,
    max_jump: float = 2.0
) -> pd.Series:
    """
    Замена выбросов предыдущим значением.

    max_jump=2.0:
    допускается максимум +200% / -200%
    относительно предыдущего значения.
    """
    s = s.copy()
    prev = s.shift(1)
    
    # относительное изменение
    rel_change = (s - prev).abs() / (prev.abs() + 1e-8)
    
    # выброс
    outliers = rel_change > max_jump
    
    # заменяем предыдущим значением
    s[outliers] = prev[outliers]
    
    return s


def restore_cap(logret_series: pd.Series, cap0_series: pd.Series):
    """
    восстановление капитализации по каждому secid
    
    Parameters
    ----------
    logret_series : pd.Series
        Series с MultiIndex (secid, tradedate)
    cap0_series : pd.Series
        Series с индексом secid (начальная капитализация)
        
    Returns
    -------
    pd.Series
        Восстановленные значения капитализации
    """
    csum = logret_series.groupby(level='secid').cumsum()
    # выравниваем cap0 на MultiIndex
    cap0_aligned = logret_series.index.get_level_values('secid').map(cap0_series)
    cap0_aligned = pd.Series(cap0_aligned, index=logret_series.index)
    return cap0_aligned * np.exp(csum)


def restore_cap_safe(logret_series: pd.Series, cap0_series: pd.Series) -> pd.Series:
    """
    Safe version of restore_cap that handles NaN values.
    
    Parameters
    ----------
    logret_series : pd.Series
        Series с MultiIndex (secid, tradedate)
    cap0_series : pd.Series
        Series с индексом secid
        
    Returns
    -------
    pd.Series
        Восстановленные значения капитализации (NaN for invalid values)
    """
    result = restore_cap(logret_series, cap0_series)
    # Handle any inf or invalid values
    result = result.replace([np.inf, -np.inf], np.nan)
    return result

def hard_clip_logret(s, low=-0.2, high=0.2):
    return s.clip(lower=low, upper=high)

def robust_clip(s, z=6):
    mu = s.rolling(20, min_periods=1).mean()
    std = s.rolling(20, min_periods=1).std()

    upper = mu + z * std
    lower = mu - z * std

    return s.clip(lower=lower, upper=upper)

def prepare_xy(
    df: pd.DataFrame,
    y_name: str = "log_returns_dailycapitalization_1",
    cap_col: str = "dailycapitalization",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    y = df[y_name].to_frame()
    cap = df[cap_col]
    X = df.drop(columns=[y_name, cap_col])
    return y, X, cap

def show_shape(fn: Callable):
    def _wrapped(df, *args, **kwargs):
        out: pd.DataFrame = fn(df, *args, **kwargs)
        logger.info(f"Step {fn.__name__} finished. Shape: ({out.shape})")
        return out
    return _wrapped
