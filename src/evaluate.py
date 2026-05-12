import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, make_scorer
from src.utils import apply_outlier_filter


def mape(y_true, y_pred, eps=1e-8):
    return np.mean(np.abs((y_true - y_pred) / np.clip(np.abs(y_true), eps, None)))


def wmape(y_true, y_pred, weights=None, eps=1e-8):
    y_pred = apply_outlier_filter(y_pred)
    y_true, y_pred = y_true.align(y_pred, join='inner')

    if weights is not None:
        y_true, weights = y_true.align(weights, join='inner')
        y_pred = y_pred.loc[y_true.index]
        num = np.sum(weights * np.abs(y_true - y_pred))
        den = np.sum(weights * np.clip(np.abs(y_true), eps, None))
    else:
        num = np.sum(np.abs(y_true - y_pred))
        den = np.sum(np.clip(np.abs(y_true), eps, None))

    return num / den


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
        y_pred = y_pred.loc[y_true.index]
        return wmape(y_true, y_pred, weights)

    return wmape(y_true, y_pred, None)


def wf1_score(y_true: pd.Series, y_pred: pd.Series, weights: pd.Series | None = None):
    y_true, y_pred = y_true.align(y_pred, join='inner')

    if weights is not None:
        y_true, weights = y_true.align(weights, join='inner')
        y_pred = y_pred.loc[y_true.index]

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


scorer = make_scorer(ic_score, greater_is_better=True)

def get_feature_importances(forecaster, cfg: ForecastConfig, cols: list[str]) -> pd.Series:

    fp = forecaster.get_fitted_params(deep=True)

    if fp.get('estimator') is not None:
        fil = fp.get('estimator').feature_importances_[0]
        fis = fp.get('estimator').feature_importances_[1]
        if cfg.estimator_type == EstimatorType.NGBOOST:
            pass
        features = pd.DataFrame({
            'feature': cols,
            'importance_mean': fil,
            'importance_std': fis,
        }).sort_values('importance_mean', ascending=False).iloc[:20]

        fig, ax = plt.subplots(figsize=(10, 6))
        fig.suptitle("Feature Importance", fontsize=16)
        ax.bar(features['feature'], features['importance_mean'], yerr=features['importance_std'], capsize=5,
               color='skyblue')
        ax.set_xlabel("Feature", fontsize=12)
        ax.set_ylabel("Mean", fontsize=12)
        ax.set_xticklabels(features['feature'], rotation=45, ha='right')
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(config.DATA_DIR / 'artifacts' / 'fi.png')
        plt.show()
        return pd.Series(features['importance_mean'], index=features['feature']).sort_values(ascending=False)
    return pd.Series()

def get_permutation_importances(
    forecaster,
    X_test,
    y_test,
    scoring=scorer,
):
    res = permutation_importance(
        forecaster.estimator_,  # fitted sklearn-like estimator
        X_test,
        y_test,
        scoring=scorer,
        n_repeats=10,
        random_state=42,
        n_jobs=1,  # safer for complex objects
    )
    features = pd.DataFrame({
        'feature': X_test.columns,
        'importance_mean': res.importances_mean,
        'importance_std': res.importances_std,
    }).sort_values('importance_mean', ascending=False).iloc[:20]

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle("Permutation Feature Importance (IC)", fontsize=16)
    ax.bar(features['feature'], features['importance_mean'], yerr=features['importance_std'], capsize=5,
           color='skyblue')
    ax.set_xlabel("Feature", fontsize=12)
    ax.set_ylabel("Mean Decrease in IC", fontsize=12)
    ax.set_xticklabels(features['feature'], rotation=45, ha='right')
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(config.DATA_DIR / 'artifacts' / 'pfi.png')
    plt.show()
    return pd.Series(res.importances_mean, index=X_test.columns).sort_values(ascending=False)

