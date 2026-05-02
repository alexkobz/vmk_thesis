import numpy as np
import pandas as pd


def detect_outliers(s: pd.Series, threshold=3.0):
    """
    Detect outliers in a Series using z-score.

    Parameters:
        s (pd.Series): Input Series (can be MultiIndex)
        threshold (float): Z-score threshold

    Returns:
        pd.Series: Boolean mask where True indicates an outlier
    """
    mean = s.mean()
    std = s.std()
    mask = (s - mean).abs() > threshold * std
    return mask

def apply_group_outlier_filter(s: pd.Series, threshold=3.0, group_level=0):
    """
    Replace outliers with previous value (forward fill) per group.

    Parameters:
        s (pd.Series): Input Series (can be MultiIndex)
        threshold (float): Z-score threshold
        group_level (int or str): Level of MultiIndex to group by

    Returns:
        pd.Series: Series with outliers replaced by previous value
    """
    def f(group):
        mask = detect_outliers(group, threshold)
        return group.mask(mask).ffill().bfill()

    return s.groupby(level=group_level).apply(f)

def apply_outlier_filter(s: pd.Series, threshold=3.0, group_level=0):
    """
    Replace outliers with previous value (forward fill) per group.

    Parameters:
        s (pd.Series): Input Series (can be MultiIndex)
        threshold (float): Z-score threshold
        group_level (int or str): Level of MultiIndex to group by

    Returns:
        pd.Series: Series with outliers replaced by previous value
    """

    mask = detect_outliers(s, threshold)
    return s.mask(mask).dropna()

# восстановление капитализации по каждому secid
def restore_cap(logret_series: pd.Series, cap0_series: pd.Series):
    # logret_series: pd.Series с MultiIndex (secid, tradedate)
    # cap0_series: pd.Series с индексом secid
    csum = logret_series.groupby(level='secid').cumsum()
    # выравниваем cap0 на MultiIndex
    cap0_aligned = logret_series.index.get_level_values('secid').map(cap0_series)
    cap0_aligned = pd.Series(cap0_aligned, index=logret_series.index)
    return cap0_aligned * np.exp(csum)

def prepare_xy(
    df: pd.DataFrame,
    y_name: str = "log_returns_dailycapitalization_1",
    cap_col: str = "dailycapitalization",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    y = df[y_name].to_frame()
    cap = df[cap_col]
    X = df.drop(columns=[y_name, cap_col])
    return y, X, cap
