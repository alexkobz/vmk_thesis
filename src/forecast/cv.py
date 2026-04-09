from __future__ import annotations

import numpy as np
import pandas as pd
from sktime.forecasting.base import ForecastingHorizon
from sktime.split import ExpandingWindowSplitter

from src.forecast.config import ForecastConfig


def build_cv(cfg: ForecastConfig) -> ExpandingWindowSplitter:
    fh = np.arange(1, cfg.horizon_days + 1)
    return ExpandingWindowSplitter(
        initial_window=cfg.initial_window,
        step_length=cfg.step_length,
        fh=fh,
    )

def split_multiindex_by_date(y, splitter, date_level="tradedate"):
    """
    Глобальный сплит для MultiIndex (secid, tradedate) по датам.

    splitter: любой sktime splitter (например, ExpandingWindowSplitter),
    применяемый к шаблонному одномерному индексу дат.
    Возвращает генератор (train_iloc, test_iloc) для y.
    """
    if not isinstance(y.index, pd.MultiIndex):
        raise ValueError("y must have a MultiIndex")
    if date_level not in y.index.names:
        raise ValueError(f"date level '{date_level}' not in y.index.names")

    dates = y.index.get_level_values(date_level)
    template = pd.Index(dates.unique()).sort_values()

    for train_dates, test_dates in splitter.split_loc(template):
        train_mask = dates.isin(train_dates)
        test_mask = dates.isin(test_dates)
        yield np.flatnonzero(train_mask), np.flatnonzero(test_mask)

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
    cap_col: str = 'dailycapitalization',
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if cap_col not in X_train.columns:
        return X_train, X_test
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
    cap_col: str = 'dailycapitalization',
) -> pd.Series:
    return (
        X_test.loc[pd.IndexSlice[secids_keep, :], cap_col]
        .groupby(level="secid")
        .first()
    )
