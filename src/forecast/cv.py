from __future__ import annotations

import numpy as np
import pandas as pd
from sktime.forecasting.base import ForecastingHorizon
from sktime.split import ExpandingWindowSplitter

from src.forecast.config import ForecastConfig


def build_cv(cfg: ForecastConfig) -> ExpandingWindowSplitter:
    fh = np.arange(1, cfg.horizon_days + 1)
    return ExpandingWindowSplitter(
        initial_window=cfg.horizon_days * cfg.initial_years,
        step_length=cfg.step_days,
        fh=fh,
    )


def split_fold(
    y: pd.DataFrame,
    X: pd.DataFrame,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    return y_train, y_test, X_train, X_test


def align_test_indices(
    y_test: pd.DataFrame,
    X_test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    common_idx = y_test.index.intersection(X_test.index)
    return y_test.loc[common_idx], X_test.loc[common_idx]


def drop_cap_col(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    cap_col: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    return X_train.drop(columns=[cap_col]), X_test.drop(columns=[cap_col])


def _filter_secids_with_full_dates(X: pd.DataFrame, test_dates: pd.Index) -> pd.Index:
    n_dates = len(test_dates)
    idx = X.index
    if not isinstance(idx, pd.MultiIndex):
        raise ValueError("X must have a MultiIndex with level 'secid' and 'tradedate'")
    secids = idx.get_level_values("secid")
    dates = idx.get_level_values("tradedate")
    counts = pd.Series(dates, index=secids).groupby(level=0).nunique()
    return counts[counts == n_dates].index


def filter_secids(
    y_train: pd.DataFrame,
    y_test: pd.DataFrame,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    test_dates: pd.Index,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Index]:
    secids_keep = _filter_secids_with_full_dates(X_test, test_dates)
    y_test = y_test.loc[pd.IndexSlice[secids_keep, :]]
    X_test = X_test.loc[pd.IndexSlice[secids_keep, :]]
    y_train = y_train.loc[pd.IndexSlice[secids_keep, :]]
    X_train = X_train.loc[pd.IndexSlice[secids_keep, :]]
    return y_train, y_test, X_train, X_test, secids_keep


def build_relative_fh(test_dates: pd.Index) -> ForecastingHorizon:
    return ForecastingHorizon(
        np.arange(1, len(test_dates) + 1),
        is_relative=True,
        freq="D",
    )


def get_initial_cap(
    X_test: pd.DataFrame,
    secids_keep: pd.Index,
    cap_col: str,
) -> pd.Series:
    return (
        X_test.loc[pd.IndexSlice[secids_keep, :], cap_col]
        .groupby(level="secid")
        .first()
    )
