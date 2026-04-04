import pandas as pd


def _find_feature_name_provider(fitted_params):
    for v in fitted_params.values():
        if hasattr(v, "get_feature_names_out") or hasattr(v, "feature_names_out_"):
            return v
    return None


def _get_feature_names(fitted_params):
    provider = _find_feature_name_provider(fitted_params)
    if provider is not None:
        if hasattr(provider, "get_feature_names_out"):
            return list(provider.get_feature_names_out())
        return list(provider.feature_names_out_)

    est = fitted_params.get("estimator")
    if est is not None and hasattr(est, "feature_names_in_"):
        if len(est.feature_names_in_) == len(est.feature_importances_):
            return list(est.feature_names_in_)

    n = len(fitted_params["estimator"].feature_importances_)
    return [f"f{i}" for i in range(n)]


def get_feature_importances(obj) -> pd.Series:
    if isinstance(obj, dict) and "forecaster" in obj:
        forecaster = obj["forecaster"]
    else:
        forecaster = obj

    fp = forecaster.get_fitted_params(deep=True)
    rf = fp["estimator"]
    feat_names = _get_feature_names(fp)

    fi = pd.Series(rf.feature_importances_, index=feat_names).sort_values(ascending=False)
    return fi
