import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import f1_score, make_scorer


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
        'mape': mape_by_secid.to_dict(),
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
        'f1': f1_by_secid.to_dict(),
        'wf1': f1_weighted_overall,
    }


# --- outlier helpers ---
def detect_outliers(s: pd.Series, threshold: float = 3.0) -> pd.Series:
    """
    Detect outliers in a Series using z-score.
    Returns boolean mask where True indicates an outlier.
    """
    mean = s.mean()
    std = s.std()
    return (s - mean).abs() > threshold * std


def apply_outlier_filter(s: pd.Series, threshold: float = 3.0, group_level=None) -> pd.Series:
    """
    Replace outliers with previous value (forward fill) per group or globally.
    """
    if group_level is not None:
        def _f(g):
            mask = detect_outliers(g, threshold)
            return g.mask(mask).ffill().bfill()

        return s.groupby(level=group_level).apply(_f)
    else:
        mask = detect_outliers(s, threshold)
        return s.mask(mask).ffill().bfill()


def wmape_score(
    y_true: pd.Series,
    y_pred: pd.Series,
    weights: pd.Series,
    level='secid',
    method: str = 'replace',
    threshold: float = 3.0,
    group_level=None,
    winsor_pct: float = 0.01,
    trim_pct: float = 0.01,
    weight_epsilon: float = 1e-6,
):
    """
    Compute wmape with optional simple outlier handling strategies.
    method: 'replace'|'winsorize'|'trim'|'weighted'|'none'
    """
    y_true, y_pred = y_true.align(y_pred, join='inner')
    y_true, weights = y_true.align(weights, join='inner')

    df = pd.DataFrame({
        'y': np.asarray(y_true).ravel(),
        'yhat': np.asarray(y_pred).ravel(),
        'w': np.asarray(weights).ravel()
    }, index=y_true.index)

    if method is None or method == 'none':
        return wmape(df['y'], df['yhat'], df['w'])

    if method == 'replace':
        # Replace outliers in y and yhat by group or globally
        if group_level is not None:
            df['y'] = apply_outlier_filter(df['y'], threshold=threshold, group_level=group_level)
            df['yhat'] = apply_outlier_filter(df['yhat'], threshold=threshold, group_level=group_level)
        else:
            df['y'] = apply_outlier_filter(df['y'], threshold=threshold, group_level=None)
            df['yhat'] = apply_outlier_filter(df['yhat'], threshold=threshold, group_level=None)

    elif method == 'winsorize':
        lo, hi = df['y'].quantile(winsor_pct), df['y'].quantile(1 - winsor_pct)
        df['y'] = df['y'].clip(lo, hi)
        lo, hi = df['yhat'].quantile(winsor_pct), df['yhat'].quantile(1 - winsor_pct)
        df['yhat'] = df['yhat'].clip(lo, hi)

    elif method == 'trim':
        errors = (df['y'] - df['yhat']).abs()
        cutoff = errors.quantile(1 - trim_pct)
        keep = errors <= cutoff
        df = df.loc[keep]
        if df.empty:
            return np.nan

    elif method == 'weighted':
        # downweight extreme targets
        z = (df['y'] - df['y'].mean()).abs() / (df['y'].std() + weight_epsilon)
        downweight = 1.0 / (1.0 + z)
        df['w'] = df['w'] * downweight

    else:
        raise ValueError(f"Unknown method '{method}' for wmape_score")

    # Final cleanup
    df = df.dropna(subset=['y', 'yhat', 'w'])
    if df.empty:
        return np.nan
    wmape_overall = wmape(df['y'], df['yhat'], df['w'])
    return wmape_overall


def wf1_score(y_true, y_pred, weights, level='secid'):
    y_true, y_pred = y_true.align(y_pred, join='inner')
    y_true, weights = y_true.align(weights, join='inner')

    df = pd.DataFrame({
        'y': np.asarray(y_true).ravel(),
        'yhat': np.asarray(y_pred).ravel(),
        'w': np.asarray(weights).ravel()
    }, index=y_true.index)
    df[['y', 'yhat']] = df.groupby(level=level)[['y', 'yhat']].ffill()
    df = df.dropna(subset=['y', 'yhat'])
    f1_weighted_overall = sign_f1(
        df['y'], df['yhat'],
        average='macro',
        sample_weight=df['w']
    )
    return f1_weighted_overall


def ic_score(y_true: pd.Series, y_pred: pd.Series):
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred)
    # handle multi-output predictions
    if y_pred.ndim > 1:
        if y_pred.shape[1] == 1:
            y_pred = y_pred.ravel()
        else:
            y_pred = y_pred[:, 0]    # choose first output (change if needed)
    y_pred = y_pred.ravel()
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    if mask.sum() < 2:
        return np.nan
    yt, yp = y_true[mask], y_pred[mask]
    if np.all(yt == yt[0]) or np.all(yp == yp[0]):
        return np.nan
    ic = float(np.corrcoef(yt, yp)[0, 1])
    return ic

scorer = make_scorer(ic_score, greater_is_better=True)
