"""
Custom expanding window splitter that splits by year boundaries.
Useful for time series with MultiIndex (secid, tradedate).
"""

from __future__ import annotations

from typing import Generator
import numpy as np
import pandas as pd


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

    def __init__(self, min_train_years: int = 5):
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
    >>> splitter = ExpandingWindowYearSplitterBySecid(min_train_years=5)
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
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]

    return y_train, y_test, X_train, X_test
