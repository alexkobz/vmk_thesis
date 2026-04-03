import numpy as np
import pandas as pd
def _align_series(y, yhat, weights=None):
    idx = y.index.intersection(yhat.index)
    y = y.loc[idx]
    yhat = yhat.loc[idx]
    if weights is None:
        return y, yhat, None
    w = weights.loc[idx]
    return y, yhat, w


def mape_by_secid(y, yhat, eps=1e-6, max_ape=10.0):
    """
    Robust MAPE per secid with safeguards:
    - aligns indices
    - drops non-finite values
    - uses max(|y|, eps) in denominator
    - caps pointwise APE by max_ape to reduce anomalies
    """
    y, yhat, _ = _align_series(y, yhat, None)
    df = pd.DataFrame({"y": y, "yhat": yhat})
    df = df.replace([np.inf, -np.inf], np.nan).dropna()

    def _mape(s):
        yv = s["y"].to_numpy()
        yh = s["yhat"].to_numpy()
        denom = np.maximum(np.abs(yv), eps)
        ape = np.abs(yv - yh) / denom
        ape = np.minimum(ape, max_ape)
        return float(np.mean(ape)) if len(ape) else np.nan

    return df.groupby(level="secid").apply(_mape)


def wmape(y, yhat, weights=None, eps=1e-6, max_ape=10.0):
    """
    Robust WMAPE:
    - aligns indices
    - drops non-finite values
    - uses abs(y) in denominator
    - caps pointwise APE by max_ape
    """
    y, yhat, w = _align_series(y, yhat, weights)
    df = pd.DataFrame({"y": y, "yhat": yhat})
    if w is not None:
        df["w"] = w
    df = df.replace([np.inf, -np.inf], np.nan).dropna()

    yv = df["y"].to_numpy()
    yh = df["yhat"].to_numpy()
    denom = np.maximum(np.abs(yv), eps)
    ape = np.abs(yv - yh) / denom
    ape = np.minimum(ape, max_ape)

    if w is None:
        return float(np.sum(ape * denom) / np.sum(denom)) if len(ape) else np.nan

    wv = df["w"].to_numpy()
    wv = np.maximum(wv, 0.0)
    w_denom = wv * denom
    if np.sum(w_denom) <= 0:
        return np.nan
    return float(np.sum(wv * ape * denom) / np.sum(w_denom))
