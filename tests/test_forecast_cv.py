import numpy as np
import pandas as pd
import pytest

from src.forecast.config import ForecastConfig
from src.forecast.cv import (
    align_test_indices,
    build_cv,
    build_relative_fh,
    drop_cap_col,
    filter_secids,
    get_initial_cap,
)
from src.forecast.data import prepare_xy


def _panel_df(values, secids, dates, col):
    index = pd.MultiIndex.from_product(
        [secids, dates],
        names=["secid", "tradedate"],
    )
    return pd.DataFrame({col: values}, index=index)


@pytest.mark.parametrize(
    "drop_cols",
    [
        ("drop_me",),
        ("drop_me", "drop_me2"),
    ],
)
def test_prepare_xy_drops_columns_and_sets_alias(drop_cols):
    cfg = ForecastConfig(
        y_name="y",
        y_alias="log_return",
        drop_cols=drop_cols,
    )
    df = pd.DataFrame(
        {
            "y": [1.0, 2.0],
            "x1": [10.0, 11.0],
            "drop_me": [0.0, 0.0],
            "drop_me2": [1.0, 1.0],
        }
    )

    y, X = prepare_xy(df, cfg)

    assert list(y.columns) == ["log_return"]
    assert "y" not in X.columns
    for col in drop_cols:
        assert col not in X.columns


@pytest.mark.parametrize(
    "horizon_days,initial_years,step_days",
    [
        (10, 2, 5),
        (5, 3, 1),
    ],
)
def test_build_cv_returns_splitter(horizon_days, initial_years, step_days):
    cfg = ForecastConfig(y_name="y", horizon_days=horizon_days, initial_years=initial_years, step_days=step_days)
    cv = build_cv(cfg)
    assert cv.initial_window == horizon_days * initial_years
    assert cv.step_length == step_days


@pytest.mark.parametrize(
    "dates_y,dates_x,expected_len",
    [
        (["2024-01-01", "2024-01-02"], ["2024-01-02", "2024-01-03"], 1),
        (["2024-01-01"], ["2024-01-01", "2024-01-02"], 1),
    ],
)
def test_align_test_indices_intersects(dates_y, dates_x, expected_len):
    secids = ["A"]
    dates_y = pd.to_datetime(dates_y)
    dates_x = pd.to_datetime(dates_x)

    y_test = _panel_df([1.0] * len(dates_y), secids, dates_y, "y")
    X_test = _panel_df([10.0] * len(dates_x), secids, dates_x, "x")

    y_aligned, x_aligned = align_test_indices(y_test, X_test)

    assert len(y_aligned) == expected_len
    assert len(x_aligned) == expected_len
    assert y_aligned.index.equals(x_aligned.index)


@pytest.mark.parametrize(
    "cap_col",
    ["cap", "dailycapitalization"],
)
def test_drop_cap_col(cap_col):
    X_train = pd.DataFrame({cap_col: [1.0], "x": [2.0]})
    X_test = pd.DataFrame({cap_col: [3.0], "x": [4.0]})

    X_train_m, X_test_m = drop_cap_col(X_train, X_test, cap_col)

    assert cap_col not in X_train_m.columns
    assert cap_col not in X_test_m.columns


@pytest.mark.parametrize(
    "missing_b_date",
    [True, False],
)
def test_filter_secids_keeps_full_dates(missing_b_date):
    dates = pd.to_datetime(["2024-01-01", "2024-01-02"])

    y_train = _panel_df([1.0, 2.0, 3.0, 4.0], ["A", "B"], dates, "y")

    if missing_b_date:
        test_index = pd.MultiIndex.from_tuples(
            [
                ("A", dates[0]),
                ("A", dates[1]),
                ("B", dates[0]),
            ],
            names=["secid", "tradedate"],
        )
        y_test = pd.DataFrame({"y": [5.0, 6.0, 7.0]}, index=test_index)
    else:
        y_test = _panel_df([5.0, 6.0, 7.0, 8.0], ["A", "B"], dates, "y")

    X_train = _panel_df([10.0, 11.0, 12.0, 13.0], ["A", "B"], dates, "x")
    X_test = y_test.rename(columns={"y": "x"})

    test_dates = dates
    y_tr, y_te, X_tr, X_te, secids_keep = filter_secids(
        y_train,
        y_test,
        X_train,
        X_test,
        test_dates,
    )

    expected = ["A"] if missing_b_date else ["A", "B"]
    assert list(secids_keep) == expected
    assert y_te.index.get_level_values("secid").unique().tolist() == expected
    assert X_te.index.get_level_values("secid").unique().tolist() == expected


@pytest.mark.parametrize(
    "n_dates",
    [1, 3, 5],
)
def test_build_relative_fh_length_and_relative(n_dates):
    dates = pd.date_range("2024-01-01", periods=n_dates, freq="D")
    fh = build_relative_fh(pd.Index(dates))
    assert len(fh) == n_dates
    assert fh.is_relative


@pytest.mark.parametrize(
    "cap_col",
    ["cap", "dailycapitalization"],
)
def test_get_initial_cap(cap_col):
    dates = pd.to_datetime(["2024-01-01", "2024-01-02"])
    X_test = _panel_df([100.0, 110.0, 200.0, 210.0], ["A", "B"], dates, cap_col)
    secids_keep = pd.Index(["A", "B"])

    cap0 = get_initial_cap(X_test, secids_keep, cap_col)

    assert np.isclose(cap0.loc["A"], 100.0)
    assert np.isclose(cap0.loc["B"], 200.0)
