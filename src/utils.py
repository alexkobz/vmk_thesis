import numpy as np
import pandas as pd


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

# восстановление капитализации по каждому secid
def restore_cap(logret_series, cap0_series):
    # logret_series: pd.Series с MultiIndex (secid, tradedate)
    # cap0_series: pd.Series с индексом secid
    csum = logret_series.groupby(level='secid').cumsum()
    # выравниваем cap0 на MultiIndex
    cap0_aligned = logret_series.index.get_level_values('secid').map(cap0_series)
    cap0_aligned = pd.Series(cap0_aligned, index=logret_series.index)
    return cap0_aligned * np.exp(csum)
