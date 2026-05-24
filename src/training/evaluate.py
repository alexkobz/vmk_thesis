import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import f1_score


def mape(y_true, y_pred, eps=1e-8):
    y_true, y_pred = y_true.align(y_pred, join="inner")
    return np.mean(np.abs((y_true - y_pred) / np.clip(np.abs(y_true), eps, None)))


def wmape(y_true, y_pred, weights=None, eps=1e-8):
    y_true, y_pred = y_true.align(y_pred, join='inner')

    if weights is not None:
        y_true, weights = y_true.align(weights, join='inner')
        y_pred = y_pred.reindex(y_true.index)
        num = np.sum(weights * np.abs(y_true - y_pred))
        den = np.sum(weights * np.clip(np.abs(y_true), eps, None))
    else:
        num = np.sum(np.abs(y_true - y_pred))
        den = np.sum(np.clip(np.abs(y_true), eps, None))

    mape = num / den if abs(den) > 1 else np.nan
    if mape > 1:
        return np.nan
    return mape


def sign_f1(y_true, y_pred, average='binary', sample_weight=None):
    y_true_cls = np.asarray(y_true) > 0
    y_pred_cls = np.asarray(y_pred) > 0
    return f1_score(
        y_true_cls,
        y_pred_cls,
        average=average,
        sample_weight=sample_weight,
        zero_division=0,
    )


def wmape_score(y_true: pd.Series, y_pred: pd.Series, weights: pd.Series | None = None):
    y_true, y_pred = y_true.align(y_pred, join='inner')

    if weights is not None:
        y_true, weights = y_true.align(weights, join='inner')
        y_pred = y_pred.reindex(y_true.index)
        return wmape(y_true, y_pred, weights)

    return wmape(y_true, y_pred, None)


def wf1_score(y_true: pd.Series, y_pred: pd.Series, weights: pd.Series | None = None):
    y_true, y_pred = y_true.align(y_pred, join='inner')

    if weights is not None:
        y_true, weights = y_true.align(weights, join='inner')
        y_pred = y_pred.reindex(y_true.index)

        df = pd.DataFrame({
            'y': np.asarray(y_true).ravel(),
            'yhat': np.asarray(y_pred).ravel(),
            'w': np.asarray(weights).ravel(),
        }, index=y_true.index).dropna(subset=['y', 'yhat', 'w'])

        return sign_f1(df['y'], df['yhat'], average='binary', sample_weight=df['w'])

    df = pd.DataFrame({
        'y': np.asarray(y_true).ravel(),
        'yhat': np.asarray(y_pred).ravel(),
    }, index=y_true.index).dropna(subset=['y', 'yhat'])

    return sign_f1(df['y'], df['yhat'], average='binary')


def ic_score(y_true: pd.Series, y_pred: pd.Series):
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred)

    if y_pred.ndim > 1:
        if y_pred.shape[1] == 1:
            y_pred = y_pred.ravel()
        else:
            y_pred = y_pred[:, 0]

    y_pred = y_pred.ravel()
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    if mask.sum() < 2:
        return np.nan

    yt, yp = y_true[mask], y_pred[mask]
    if np.all(yt == yt[0]) or np.all(yp == yp[0]):
        return np.nan

    return float(np.corrcoef(yt, yp)[0, 1])
