import numpy as np
import pandas as pd


y_name = 'dailycapitalization'

def create_lagged_features(df: pd.DataFrame, lags: list[int]) -> pd.DataFrame:
    """
    Create lagged features and returns for specified columns and lags.
    """
    for lag in lags:
        df[f'log_returns_{lag}'] = np.log(df[y_name]).diff(lag)
        df[f'pct_returns_{lag}'] = df[y_name].pct_change(periods=lag) * 100
    return df


def create_windowed_features(df: pd.DataFrame, windows: list[int]) -> pd.DataFrame:
    """
    Create rolling window features and returns for specified columns and window sizes.
    """
    for window in windows:
        df[f'cap_rolling_mean_{window}'] = df[y_name].shift(1).rolling(window).mean()
        df[f'log_returns_rolling_mean_{window}'] = np.log(df[y_name] / df[f'cap_rolling_mean_{window}'])
        df[f'pct_returns_rolling_mean_{window}'] = df[y_name] / df[f'cap_rolling_mean_{window}'] * 100
    return df


