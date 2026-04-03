import numpy as np
import pandas as pd
import pytest

from src.evaluate import mape_by_secid, wmape


def _panel_series(values, secids, dates):
    index = pd.MultiIndex.from_product(
        [secids, dates],
        names=["secid", "tradedate"],
    )
    return pd.Series(values, index=index)


@pytest.mark.parametrize(
    "y_vals,yhat_vals,expected_by_secid",
    [
        (
            [100, 110, 50, 100],
            [90, 100, 55, 80],
            {
                "A": np.mean(np.abs((np.array([100, 110]) - np.array([90, 100])) / np.array([100, 110]))),
                "B": np.mean(np.abs((np.array([50, 100]) - np.array([55, 80])) / np.array([50, 100]))),
            },
        ),
    ],
)
def test_mape_by_secid(y_vals, yhat_vals, expected_by_secid):
    secids = ["A", "B"]
    dates = pd.to_datetime(["2024-01-01", "2024-01-02"])

    y = _panel_series(y_vals, secids, dates)
    yhat = _panel_series(yhat_vals, secids, dates)

    result = mape_by_secid(y, yhat)

    assert np.isclose(result.loc["A"], expected_by_secid["A"])
    assert np.isclose(result.loc["B"], expected_by_secid["B"])


@pytest.mark.parametrize(
    "y_vals,yhat_vals,eps,max_ape,expected",
    [
        ([0.0, 1.0], [1.0, 2.0], 1e-6, 2.0, 1.5),
    ],
)
def test_mape_by_secid_caps_and_eps(y_vals, yhat_vals, eps, max_ape, expected):
    secids = ["A"]
    dates = pd.to_datetime(["2024-01-01", "2024-01-02"])
    y = _panel_series(y_vals, secids, dates)
    yhat = _panel_series(yhat_vals, secids, dates)

    result = mape_by_secid(y, yhat, eps=eps, max_ape=max_ape)

    assert np.isclose(result.loc["A"], expected)


@pytest.mark.parametrize(
    "y_vals,yhat_vals,expected",
    [
        ([100, 200], [90, 210], (np.abs(np.array([100, 200]) - np.array([90, 210])).sum()) / np.array([100, 200]).sum()),
    ],
)
def test_wmape_unweighted(y_vals, yhat_vals, expected):
    secids = ["A"]
    dates = pd.to_datetime(["2024-01-01", "2024-01-02"])

    y = _panel_series(y_vals, secids, dates)
    yhat = _panel_series(yhat_vals, secids, dates)

    assert np.isclose(wmape(y, yhat), expected)


@pytest.mark.parametrize(
    "y_vals,yhat_vals,w_vals,expected",
    [
        (
            [100, 200],
            [90, 210],
            [1, 3],
            (np.array([1, 3]) * np.abs(np.array([100, 200]) - np.array([90, 210]))).sum()
            / (np.array([1, 3]) * np.array([100, 200])).sum(),
        ),
    ],
)
def test_wmape_weighted(y_vals, yhat_vals, w_vals, expected):
    secids = ["A"]
    dates = pd.to_datetime(["2024-01-01", "2024-01-02"])

    y = _panel_series(y_vals, secids, dates)
    yhat = _panel_series(yhat_vals, secids, dates)
    weights = _panel_series(w_vals, secids, dates)

    assert np.isclose(wmape(y, yhat, weights=weights), expected)


@pytest.mark.parametrize(
    "y_vals,yhat_vals,w_vals,expected",
    [
        ([1.0, np.nan, 3.0], [2.0, 2.0, np.inf], [1.0, 1.0, 1.0], 1.0),
    ],
)
def test_wmape_aligns_and_drops_nonfinite(y_vals, yhat_vals, w_vals, expected):
    secids = ["A"]
    dates = pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"])
    y = _panel_series(y_vals, secids, dates)
    yhat = _panel_series(yhat_vals, secids, dates)
    weights = _panel_series(w_vals, secids, dates)

    result = wmape(y, yhat, weights=weights)

    assert np.isclose(result, expected)
