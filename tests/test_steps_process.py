import numpy as np
import pandas as pd
import pytest

from src.steps.process import (
    add_log_returns,
    categorize,
    drop_additional_issues,
    ffill_bfill,
    filter_boards,
    filter_null_cols,
    filter_types,
    filter_years,
    filter_zero_target,
    gather_secids,
    replace_target,
    replace_zeros_with_nan,
    set_index,
    fill_days,
    target_imputer,
)
from src.utils import restore_cap


def _panel_df(values, secids, dates, cols):
    index = pd.MultiIndex.from_product([secids, dates], names=["secid", "tradedate"])
    return pd.DataFrame(values, index=index, columns=cols)


@pytest.mark.parametrize(
    "boards,expected",
    [
        ({"TQBR"}, ["TQBR"]),
        ({"TQBR", "EQBR"}, ["TQBR", "EQBR"]),
    ],
)
def test_filter_boards(boards, expected):
    df = pd.DataFrame({"boardid": ["TQBR", "EQBR", "OTHER"], "x": [1, 2, 3]})
    out = filter_boards(df, boards)
    assert out["boardid"].tolist() == expected


def test_drop_additional_issues_sets_sumcap():
    df = pd.DataFrame(
        {
            "secid": ["AAA-01", "AAA-02"],
            "tradedate": pd.to_datetime(["2024-01-01", "2024-01-01"]),
            "boardid": ["TQBR", "TQBR"],
            "cap": [10.0, 5.0],
        }
    )
    out = drop_additional_issues(df, "cap")
    assert (out["cap"] == 15.0).all()


def test_filter_null_cols_removes_all_nan_tickers():
    df = pd.DataFrame(
        {
            "secid": ["A", "A", "B", "B"],
            "m1": [np.nan, np.nan, 1.0, 2.0],
            "l1": [np.nan, np.nan, 3.0, 4.0],
        }
    )
    out = filter_null_cols(df, mults=["m1"], lines=["l1"])
    assert out["secid"].unique().tolist() == ["B"]


def test_categorize_fills_missing_and_sets_category():
    df = pd.DataFrame({"type": [None, "common_share"], "sector": [None, "tech"]})
    out = categorize(df)
    assert out["type"].isna().sum() == 0
    assert out["sector"].isna().sum() == 0
    assert out["type"].dtype.name == "category"
    assert out["sector"].dtype.name == "category"


@pytest.mark.parametrize(
    "types,expected",
    [
        ({"common_share"}, ["common_share"]),
        ({"preferred_share"}, ["preferred_share"]),
    ],
)
def test_filter_types(types, expected):
    df = pd.DataFrame({"type": ["common_share", "preferred_share", "bond"]})
    out = filter_types(df, types)
    assert out["type"].tolist() == expected


@pytest.mark.parametrize(
    "values,expected",
    [
        ([0, 1, 2], [np.nan, 1.0, 2.0]),
        ([0, 0, 3], [np.nan, np.nan, 3.0]),
    ],
)
def test_replace_zeros_with_nan(values, expected):
    df = pd.DataFrame({"x": values})
    out = replace_zeros_with_nan(df)
    assert np.allclose(out["x"].to_numpy(), np.array(expected), equal_nan=True)


def test_gather_secids_maps_to_shortest_by_type():
    df = pd.DataFrame(
        {
            "secid": ["AAA", "AAAA", "BB", "BBB"],
            "inn": ["1", "1", "2", "2"],
            "type": ["common_share", "common_share", "preferred_share", "preferred_share"],
        }
    )
    out = gather_secids(df)
    common = out[out["type"] == "common_share"]["secid"].unique().tolist()
    preferred = out[out["type"] == "preferred_share"]["secid"].unique().tolist()
    assert common == ["AAA"]
    assert preferred == ["BB"]


def test_set_index_picks_highest_volume_then_y():
    df = pd.DataFrame(
        {
            "secid": ["A", "A"],
            "tradedate": pd.to_datetime(["2024-01-01", "2024-01-01"]),
            "volume": [10, 5],
            "cap": [100.0, 200.0],
        }
    )
    out = set_index(df, index=["secid", "tradedate"], y_name="cap")
    assert out.loc[("A", pd.Timestamp("2024-01-01")), "cap"] == 100.0


def test_fill_days_adds_missing_and_marks_vacation():
    dates = pd.to_datetime(["2024-01-01", "2024-01-03"])
    df = _panel_df([[1.0], [3.0]], ["A"], dates, ["year"])
    out = fill_days(df)
    assert len(out) == 3
    missing_day = pd.Timestamp("2024-01-02")
    assert bool(out.loc[("A", missing_day), "is_vacation"]) is True


def test_ffill_bfill_fills_within_secid():
    dates = pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"])
    df = _panel_df([[1.0], [np.nan], [3.0]], ["A"], dates, ["x"])
    out = ffill_bfill(df)
    assert out["x"].tolist() == [1.0, 1.0, 3.0]


@pytest.mark.parametrize(
    "values",
    [
        [np.nan, 2.0, 4.0],
        [np.nan, np.nan, 3.0],
    ],
)
def test_target_imputer_fills_with_median(values):
    df = pd.DataFrame({"cap": values, "close": [10.0, 10.0, 10.0]})
    out = target_imputer(df, y_name="cap")
    median = np.nanmedian(values)
    assert np.isclose(out["cap"].iloc[0], median)


def test_add_log_returns_positive_only():
    dates = pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"])
    df = _panel_df([[1.0], [2.0], [4.0], [1.0], [-2.0], [4.0]], ["A", "B"], dates, ["x"])
    out = add_log_returns(df, "x", [1])

    a_vals = out.loc[pd.IndexSlice["A", :], "log_returns_x_1"].to_numpy()
    b_vals = out.loc[pd.IndexSlice["B", :], "log_returns_x_1"].to_numpy()

    assert np.isclose(a_vals[1], np.log(2.0) - np.log(1.0))
    assert np.isnan(b_vals).sum() == 3


def test_replace_target_replaces_zero_sum_secids():
    dates = pd.to_datetime(["2024-01-01", "2024-01-02"])
    index = pd.MultiIndex.from_product([["A", "B"], dates], names=["secid", "tradedate"])
    df = pd.DataFrame(
        {
            "cap": [100.0, 100.0, 200.0, 220.0],
            "log_returns_cap_1": [0.0, 0.0, 0.1, 0.1],
            "log_returns_close_1": [0.1, 0.2, 0.3, 0.4],
        },
        index=index,
    )

    out = replace_target(df, y_name="cap")

    a_expected = restore_cap(
        df.loc[pd.IndexSlice["A", :], "log_returns_close_1"],
        pd.Series({"A": 100.0}),
    )
    assert np.allclose(
        out.loc[pd.IndexSlice["A", :], "cap"].to_numpy(),
        a_expected.to_numpy(),
    )
    assert np.allclose(
        out.loc[pd.IndexSlice["B", :], "log_returns_cap_1"].to_numpy(),
        df.loc[pd.IndexSlice["B", :], "log_returns_cap_1"].to_numpy(),
    )


def test_filter_years_drops_nan_years():
    df = pd.DataFrame({"year": [2024.0, np.nan]})
    out = filter_years(df)
    assert out["year"].tolist() == [2024.0]


def test_filter_zero_target_drops_small_sum():
    dates = pd.to_datetime(["2024-01-01", "2024-01-02"])
    index = pd.MultiIndex.from_product([["A", "B"], dates], names=["secid", "tradedate"])
    df = pd.DataFrame(
        {
            "log_returns_cap_1": [0.0, 0.0, 1e-4, 1e-4],
        },
        index=index,
    )
    out = filter_zero_target(df, y_name="cap")
    assert out.index.get_level_values("secid").unique().tolist() == ["B"]
