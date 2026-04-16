import pandas as pd
from matplotlib import pyplot as plt
from sklearn.inspection import permutation_importance

from config import config
from src.forecast import ForecastConfig, EstimatorType
from src.forecast.evaluate import scorer


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

