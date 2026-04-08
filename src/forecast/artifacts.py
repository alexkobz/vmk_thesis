from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib



def _ensure_dir(path: str | Path) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out

def _serialize_estimator(estimator) -> dict[str, Any] | None:
    if estimator is None:
        return None
    info: dict[str, Any] = {
        "class": f"{estimator.__class__.__module__}.{estimator.__class__.__name__}",
    }
    if hasattr(estimator, "get_params"):
        try:
            info["params"] = estimator.get_params(deep=True)
        except TypeError:
            info["params"] = estimator.get_params()
    else:
        info["repr"] = repr(estimator)
    return info

# def save_run_artifacts(
#     cfg: ForecastConfig,
#     dir: str | Path,
#     filename: str = "metrics.json",
# ) -> dict[str, Path]:
#     out_dir = _ensure_dir(dir)
#     path = out_dir / filename
#     saved: dict[str, Path] = {}
#
#     cfg_dict = asdict(cfg)
#     cfg_dict["estimator"] = _serialize_estimator(cfg.estimator)
#
#     payload = {
#         "scores": result.get("scores", []),
#         "mean_score": result.get("mean_score"),
#         "std_score": result.get("std_score"),
#         "config": cfg_dict,
#     }
#     metrics_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
#     saved["metrics"] = metrics_path
#
#     mape = result.get("last_mape_by_secid")
#     if isinstance(mape, pd.Series):
#         mape_path = out_dir / "mape_by_secid.csv"
#         mape.to_csv(mape_path, header=["mape"])
#         saved["mape_by_secid"] = mape_path
#
#     return saved

# def save_run_artifacts(
#     cfg: ForecastConfig,
#     forecaster,
#     cap_df: pd.DataFrame,
#     dir: str | Path,
#     fold_idx: int,
#     filename: str = "metrics.json",
#     logger: loguru.Logger = loguru.logger,
# ) -> dict[str, Path]:
#     if cfg.save_metrics:
#         os.makedirs(folder_path, exist_ok=True)
#         fold_path = folder_path / f'fold{fold_idx}'
#         os.makedirs(fold_path, exist_ok=True)
#         mape_secid = grouped_mape(cap_y_test, cap_y_pred)
#         f1_secid = grouped_f1(cap_y_test, cap_y_pred)
#         metrics = {
#             'wmape_score': wmape(cap_y_test, cap_y_pred, weights=cap_y_test),
#             'f1_score': wf1(y_test, y_pred, sample_weight=cap_y_test),
#         }
#         logger.info(
#             "Fold {fold}: wmape={wmape_score}, f1={f1_score}",
#             fold=fold_idx,
#             wmape_score=metrics['wmape_score'],
#             f1_score=metrics['f1_score'],
#         )
#         if cfg.ticker:
#             title: str = (
#                 f"{cfg.ticker} cap forecast\n"
#                 # f"MAPE: {mape_secid[cfg.ticker] * 100:.2f}% "
#                 # f"WMAPE: {score * 100:.2f}% "
#                 # f"F1: {score * 100:.2f}% "
#             )
#             cap_y_test_ticker = cap_y_test.xs(cfg.ticker, level="secid")
#             cap_y_pred_ticker = cap_y_pred.xs(cfg.ticker, level="secid")
#             plot_ticker(
#                 cap_y_test_ticker,
#                 cap_y_pred_ticker,
#                 title=title,
#                 save_path=fold_path,
#             )


def serialize_model(forecaster, dir: str | Path, filename: str = "forecaster.joblib") -> Path:
    out_dir = _ensure_dir(dir)
    path = out_dir / filename
    joblib.dump(forecaster, path)
    return path

def deserialize_model(path: str | Path) -> Any:
    clf = joblib.load(path)
    clf._state = "fitted"  # Manually set since __init__ wasn't called BUG sktime
    return clf
