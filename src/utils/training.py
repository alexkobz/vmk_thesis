"""
Custom expanding window splitter that splits by year boundaries.
Useful for time series with MultiIndex (secid, tradedate).
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from typing import Generator

import numpy as np
import pandas as pd

from src.training.ModelConfig import ModelConfig, model_configs


class ExpandingWindowYearSplitter:
    """
    Expanding window splitter for time series with year boundaries.

    For each fold, expands training window to include all years up to year N,
    then tests on year N+1. This ensures clean year-based boundaries.

    Parameters
    ----------
    min_train_years : int, default=5
        Minimum number of complete years to include in first training fold.

    Examples
    --------
    >>> splitter = ExpandingWindowYearSplitter(min_train_years=5)
    >>> for train_idx, test_idx in splitter.split(y):
    ...     print(f"Train shape: {y.iloc[train_idx].shape}, Test shape: {y.iloc[test_idx].shape}")
    """

    def __init__(self, min_train_years: int = 4):
        self.min_train_years = min_train_years

    def split(
        self,
        y: pd.DataFrame | pd.Series,
        date_level: str = "tradedate",
    ) -> Generator[tuple[np.ndarray, np.ndarray], None, None]:
        """
        Generate train/test indices for expanding window split by year.

        Parameters
        ----------
        y : pd.DataFrame or pd.Series
            Time series with MultiIndex containing date_level (e.g., 'tradedate').
        date_level : str, default="tradedate"
            Name of the date level in MultiIndex.

        Yields
        ------
        train_idx : np.ndarray
            Boolean or integer array of training indices.
        test_idx : np.ndarray
            Boolean or integer array of test indices.
        """
        if not isinstance(y.index, pd.MultiIndex):
            raise ValueError(f"y must have MultiIndex, got {type(y.index)}")
        if date_level not in y.index.names:
            raise ValueError(f"'{date_level}' not in index levels: {y.index.names}")

        dates = y.index.get_level_values(date_level)
        years = dates.year.values
        unique_years = sorted(np.unique(years))

        if len(unique_years) < self.min_train_years + 1:
            raise ValueError(
                f"Not enough years ({len(unique_years)}) for min_train_years={self.min_train_years} "
                "plus at least 1 test year"
            )

        # Generate expanding windows: train on years [min_year, ..., year_N], test on year_N+1
        for i in range(len(unique_years) - self.min_train_years):
            train_year_end = unique_years[i + self.min_train_years - 1]
            test_year = unique_years[i + self.min_train_years]

            train_mask = years <= train_year_end
            test_mask = years == test_year

            train_idx = np.flatnonzero(train_mask)
            test_idx = np.flatnonzero(test_mask)

            # Skip if test set is empty
            if len(test_idx) > 0:
                yield train_idx, test_idx

    def get_n_splits(self, y: pd.DataFrame | pd.Series, date_level: str = "tradedate") -> int:
        """Return the number of splitting iterations in the cross-validator."""
        if not isinstance(y.index, pd.MultiIndex):
            raise ValueError(f"y must have MultiIndex, got {type(y.index)}")
        if date_level not in y.index.names:
            raise ValueError(f"'{date_level}' not in index levels: {y.index.names}")

        dates = y.index.get_level_values(date_level)
        unique_years = sorted(np.unique(dates.year))
        n_splits = len(unique_years) - self.min_train_years
        return max(0, n_splits)


class ExpandingWindowYearSplitterBySecid:
    """
    Expanding window splitter that splits by year but respects security IDs.

    Each security is split independently by year, then aligned across all securities.
    Useful when different securities have different date ranges but you want
    synchronized train/test splits across years.

    Parameters
    ----------
    min_train_years : int, default=5
        Minimum number of complete years to include in first training fold.
    secid_level : str, default="secid"
        Name of the security ID level in MultiIndex.
    date_level : str, default="tradedate"
        Name of the date level in MultiIndex.

    Examples
    --------
    >>> splitter = ExpandingWindowYearSplitterBySecid(min_train_years=4)
    >>> for train_idx, test_idx in splitter.split(y):
    ...     y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    """

    def __init__(
        self,
        min_train_years: int = 5,
        secid_level: str = "secid",
        date_level: str = "tradedate",
    ):
        self.min_train_years = min_train_years
        self.secid_level = secid_level
        self.date_level = date_level

    def split(
        self,
        y: pd.DataFrame | pd.Series,
    ) -> Generator[tuple[np.ndarray, np.ndarray], None, None]:
        """
        Generate train/test indices for expanding window split by year, per security.

        Each security's data is split by year independently, then indices are
        combined to create a global train/test split across all securities.

        Yields
        ------
        train_idx : np.ndarray
            Indices of all training records across all securities for this fold.
        test_idx : np.ndarray
            Indices of all test records across all securities for this fold.
        """
        if not isinstance(y.index, pd.MultiIndex):
            raise ValueError(f"y must have MultiIndex, got {type(y.index)}")
        if self.secid_level not in y.index.names:
            raise ValueError(f"'{self.secid_level}' not in index: {y.index.names}")
        if self.date_level not in y.index.names:
            raise ValueError(f"'{self.date_level}' not in index: {y.index.names}")

        dates = y.index.get_level_values(self.date_level)
        years = dates.year.values
        secids = y.index.get_level_values(self.secid_level)

        # Find year boundaries across all securities
        unique_years = sorted(np.unique(years))

        if len(unique_years) < self.min_train_years + 1:
            raise ValueError(
                f"Not enough years ({len(unique_years)}) for min_train_years={self.min_train_years}"
            )

        # Generate expanding windows
        for i in range(len(unique_years) - self.min_train_years):
            train_year_end = unique_years[i + self.min_train_years - 1]
            test_year = unique_years[i + self.min_train_years]

            train_mask = years <= train_year_end
            test_mask = years == test_year

            train_idx = np.flatnonzero(train_mask)
            test_idx = np.flatnonzero(test_mask)

            if len(test_idx) > 0:
                yield train_idx, test_idx

    def get_n_splits(self, y: pd.DataFrame | pd.Series) -> int:
        """Return the number of splitting iterations."""
        if not isinstance(y.index, pd.MultiIndex):
            raise ValueError(f"y must have MultiIndex, got {type(y.index)}")

        dates = y.index.get_level_values(self.date_level)
        unique_years = sorted(np.unique(dates.year))
        n_splits = len(unique_years) - self.min_train_years
        return max(0, n_splits)


def split_fold(
    y: pd.DataFrame,
    X: pd.DataFrame,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split y and X using same indices, keeping all data without filtering.

    Parameters
    ----------
    y : pd.DataFrame
        Target with MultiIndex (secid, tradedate)
    X : pd.DataFrame
        Features with MultiIndex (secid, tradedate)
    train_idx, test_idx : np.ndarray
        Indices from splitter

    Returns
    -------
    y_train, y_test, X_train, X_test : pd.DataFrame
        Split data with aligned indices (no filtering).
    """
    # Indices come from splitting y, so always slice y positionally,
    # then align X by index (X and y may not have identical ordering/length).
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    X_train = X.reindex(y_train.index)
    X_test = X.reindex(y_test.index)

    return y_train, y_test, X_train, X_test


def _make_absolute_fh(dates: pd.Index):
    """Create an absolute ForecastingHorizon without importing sktime at module import time."""
    from sktime.forecasting.base import ForecastingHorizon

    return ForecastingHorizon(dates, is_relative=False)


def _full_instance_horizon_index(
    *,
    secids: pd.Index,
    dates: pd.Index,
    secid_level: str,
    date_level: str,
) -> pd.MultiIndex:
    return pd.MultiIndex.from_product(
        [secids, dates],
        names=[secid_level, date_level],
    )


def prepare_yfromx_fold(
    y: pd.DataFrame,
    X: pd.DataFrame,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    *,
    secid_level: str = "secid",
    date_level: str = "tradedate",
    fill_value: float = 0.0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, object]:
    """
    Prepare a fold for sktime's YfromX.

    Returns y_train/y_test and X_train plus an X_pred and fh suitable for calling:
        model.fit(y_train, X=X_train)
        model.predict(fh=fh, X=X_pred)

    Why X_pred is expanded:
    YfromX may create a prediction index that is the cartesian product of
    (instances seen in training) x (fh dates). If X is missing some of those rows,
    sktime will raise KeyError internally when subsetting X with .loc[fh_idx].
    """
    y_train, y_test, X_train, X_test = split_fold(y, X, train_idx, test_idx)

    # fh is defined by the dates in the test fold.
    target_dates = (
        y_test.index.get_level_values(date_level)
        .unique()
        .sort_values()
    )
    fh = _make_absolute_fh(target_dates)

    # Ensure X contains all instance/date combinations sktime may request.
    secids = y_train.index.get_level_values(secid_level).unique()
    full_idx = _full_instance_horizon_index(
        secids=secids,
        dates=target_dates,
        secid_level=secid_level,
        date_level=date_level,
    )
    X_pred = X_test.reindex(full_idx).fillna(fill_value)

    return y_train, y_test, X_train, X_pred, fh


def align_y(
    y_test: pd.DataFrame,
    y_pred,
) -> tuple[pd.Series, pd.Series]:
    """
    Convert y_test (DataFrame with 1 col) and y_pred (Series/DataFrame/ndarray) to
    1D Series and align on intersection of indices.
    """
    # y_test comes as (n, 1) DataFrame in this project.
    if isinstance(y_test, pd.DataFrame):
        if y_test.shape[1] != 1:
            raise ValueError(f"Expected y_test to have 1 column, got {y_test.shape[1]}")
        y_test_s = y_test.iloc[:, 0]
    else:
        y_test_s = pd.Series(np.asarray(y_test).reshape(-1), index=y_test.index)

    if isinstance(y_pred, pd.DataFrame):
        if y_pred.shape[1] == 0:
            raise ValueError("y_pred DataFrame has 0 columns")
        y_pred_s = y_pred.iloc[:, 0]
    elif isinstance(y_pred, pd.Series):
        y_pred_s = y_pred
    else:
        # fall back to positional conversion if estimator returns ndarray-like
        y_pred_s = pd.Series(np.asarray(y_pred).reshape(-1), index=y_test_s.index)

    return y_test_s.align(y_pred_s, join="inner")


def parse_bool(value: str) -> bool:
    value_normalized = value.strip().lower()
    if value_normalized in {"1", "true", "t", "yes", "y"}:
        return True
    if value_normalized in {"0", "false", "f", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def build_train_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_config", choices=sorted(model_configs))
    parser.add_argument("--n-estimators", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--random-state", type=int)
    parser.add_argument("--verbose", type=parse_bool)
    parser.add_argument("--pooling")
    return parser


def apply_cli_overrides(cfg: ModelConfig, args: argparse.Namespace) -> ModelConfig:
    overrides = {
        "n_estimators": args.n_estimators,
        "learning_rate": args.learning_rate,
        "random_state": args.random_state,
        "verbose": args.verbose,
        "pooling": args.pooling,
    }
    overrides = {key: value for key, value in overrides.items() if value is not None}
    return replace(cfg, **overrides) if overrides else cfg
