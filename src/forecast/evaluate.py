import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

def mape(y_true, y_pred, eps=1e-8):
    return np.mean(np.abs((y_true - y_pred) / np.clip(np.abs(y_true), eps, None)))

def wmape(y_true, y_pred, weights, eps=1e-8):
    return np.sum(weights * np.abs(y_true - y_pred)) / np.sum(weights * np.clip(np.abs(y_true), eps, None))

def sign_f1(y_true, y_pred, average='macro', sample_weight=None):
    y_true_cls = np.sign(y_true)
    y_pred_cls = np.sign(y_pred)
    return f1_score(y_true_cls, y_pred_cls, average=average, sample_weight=sample_weight, zero_division=0)

def panel_mape(y_true: pd.Series, y_pred: pd.Series, weights: pd.Series, level='secid'):
    y_true, y_pred = y_true.align(y_pred, join='inner')
    y_true, weights = y_true.align(weights, join='inner')

    df = pd.DataFrame({
        'y': np.asarray(y_true).ravel(),
        'yhat': np.asarray(y_pred).ravel(),
        'w': np.asarray(weights).ravel()
    }, index=y_true.index)

    mape_by_secid = df.groupby(level=level).apply(
        lambda g: mape(g['y'], g['yhat'])
    )

    wmape_overall = wmape(df['y'], df['yhat'], df['w'])

    return {
        'mape': mape_by_secid,
        'wmape': wmape_overall,
    }

def panel_f1(y_true, y_pred, weights, level='secid'):
    y_true, y_pred = y_true.align(y_pred, join='inner')
    y_true, weights = y_true.align(weights, join='inner')

    df = pd.DataFrame({
        'y': np.asarray(y_true).ravel(),
        'yhat': np.asarray(y_pred).ravel(),
        'w': np.asarray(weights).ravel()
    }, index=y_true.index)
    df[['y', 'yhat']] = df.groupby(level=level)[['y', 'yhat']].ffill()
    df = df.dropna(subset=['y', 'yhat'])
    f1_by_secid = df.groupby(level=level).apply(
        lambda g: sign_f1(g['y'], g['yhat'], average='macro')
    )
    f1_weighted_overall = sign_f1(
        df['y'], df['yhat'],
        average='macro',
        sample_weight=df['w']
    )
    return {
        'f1': f1_by_secid,
        'wf1': f1_weighted_overall,
    }
