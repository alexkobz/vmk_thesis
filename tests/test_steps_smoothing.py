import numpy as np
import pandas as pd
import pytest

from src.steps.smoothing import apply_ewm_multiindex, lowess_smooth


@pytest.mark.parametrize(
    "values",
    [
        [1.0, 1.0, 1.0],
        [2.0, 2.0, 2.0],
    ],
)
def test_lowess_smooth_constant_series(values):
    s = pd.Series(values, index=pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]))
    out = lowess_smooth(s, frac=0.5)
    assert np.allclose(out.to_numpy(), np.array(values))


@pytest.mark.parametrize(
    "secids",
    [["A"], ["A", "B"]],
)
def test_lowess_smooth_multiindex_preserves_index(secids):
    dates = pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"])
    index = pd.MultiIndex.from_product([secids, dates], names=["secid", "tradedate"])
    s = pd.Series(np.arange(len(index), dtype=float), index=index)

    out = lowess_smooth(s, frac=0.5)

    assert out.index.equals(s.index)
    assert len(out) == len(s)


@pytest.mark.parametrize(
    "alpha",
    [0.2, 0.5],
)
def test_apply_ewm_multiindex_adds_column(alpha):
    dates = pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"])
    index = pd.MultiIndex.from_product([["A"], dates], names=["secid", "tradedate"])
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0]}, index=index)

    out = apply_ewm_multiindex(df, ["x"], alpha=alpha)

    expected = df.groupby(level="secid")["x"].transform(lambda x: x.ewm(alpha=alpha, adjust=False).mean())
    assert "x_ewm" in out.columns
    assert np.allclose(out["x_ewm"].to_numpy(), expected.to_numpy())
