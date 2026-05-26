import pandas as pd
from matplotlib import pyplot as plt
from sklearn.inspection import permutation_importance
from sklearn.metrics import make_scorer

from config import DATA_DIR
from src.training.evaluate import ic_score


def get_feature_importances(
    forecaster,
    cols: list[str],
    save_path=DATA_DIR / "processed" / "pi.png",
) -> pd.Series:

    fp = forecaster.get_fitted_params(deep=True)

    if fp.get('estimator') is not None:
        fil = fp.get('estimator').feature_importances_[0]
        fis = fp.get('estimator').feature_importances_[1]
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
        if save_path:
            plt.savefig(save_path)
        plt.show()
        return pd.Series(features['importance_mean'], index=features['feature']).sort_values(ascending=False)
    return pd.Series()

def get_permutation_importances(
    forecaster,
    X_test,
    y_test,
    scoring=ic_score,
    save_path=DATA_DIR / 'processed' / 'pfi.png',
) -> pd.Series:
    scorer = make_scorer(scoring, greater_is_better=True)
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
    if save_path:
        plt.savefig(save_path)
    plt.show()
    return pd.Series(res.importances_mean, index=X_test.columns).sort_values(ascending=False)
