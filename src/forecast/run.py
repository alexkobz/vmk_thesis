from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from config.config import ARTIFACTS_DIR
from src.forecast.artifacts import save_forecaster, save_run_artifacts
from src.forecast.evaluate import mape_by_secid, wmape
from src.forecast.config import ForecastConfig
from src.forecast.cv import (
    align_test_indices,
    build_cv,
    build_relative_fh,
    drop_cap_col,
    filter_secids,
    get_initial_cap,
    split_fold,
)
from src.forecast.model import build_forecaster
from src.forecast.plotting import plot_ticker
from src.utils import restore_cap, split_multiindex_by_date


def run_expanding_cv(
    y: pd.DataFrame,
    X: pd.DataFrame,
    cfg: ForecastConfig,
    save_model: bool = True,
    save_metrics: bool = True,
) -> dict[str, object]:
    artifacts_dir = str(ARTIFACTS_DIR)
    logger.info(
        "Forecast CV start: estimator={estimator} pooling={pooling} cap_col={cap_col} ticker={ticker}",
        estimator=type(cfg.estimator).__name__ if cfg.estimator is not None else "default",
        pooling=cfg.pooling,
        cap_col=cfg.cap_col,
        ticker=cfg.ticker,
    )
    forecaster = build_forecaster(estimator=cfg.estimator, pooling=cfg.pooling)
    cv = build_cv(cfg)

    scores = []
    last_mape_by_secid = None

    for fold_idx, (train_idx, test_idx) in enumerate(split_multiindex_by_date(y, cv), start=1):
        logger.info("Fold {fold}: split sizes train={train} test={test}", fold=fold_idx, train=len(train_idx), test=len(test_idx))
        y_train, y_test, X_train, X_test = split_fold(y, X, train_idx, test_idx)
        y_test, X_test = align_test_indices(y_test, X_test)
        if y_test.empty or X_test.empty:
            logger.info("Fold {fold}: skipped (empty test after alignment)", fold=fold_idx)
            continue
        X_train_m, X_test_m = drop_cap_col(X_train, X_test, cfg.cap_col)

        test_dates = y_test.index.get_level_values("tradedate").unique().sort_values()
        y_train, y_test, X_train_m, X_test_m, secids_keep = filter_secids(
            y_train,
            y_test,
            X_train_m,
            X_test_m,
            test_dates,
        )
        if len(secids_keep) == 0:
            logger.info("Fold {fold}: skipped (no secids kept)", fold=fold_idx)
            continue
        cap0 = get_initial_cap(X_test, secids_keep, cfg.cap_col)
        fh_rel = build_relative_fh(test_dates)

        logger.info("Fold {fold}: fitting forecaster", fold=fold_idx)
        forecaster.fit(y_train, X_train_m)
        logger.info("Fold {fold}: predicting", fold=fold_idx)
        y_pred = forecaster.predict(fh=fh_rel, X=X_test_m)

        cap_y_test = restore_cap(y_test.iloc[:, 0], cap0)
        cap_y_pred = restore_cap(y_pred.iloc[:, 0], cap0)

        last_mape_by_secid = mape_by_secid(cap_y_test, cap_y_pred)
        score = wmape(cap_y_test, cap_y_pred, weights=cap_y_test)
        scores.append(score)
        logger.info("Fold {fold}: wmape={score}", fold=fold_idx, score=score)

        if cfg.ticker:
            save_path = Path(artifacts_dir) / f"{cfg.ticker}_fold{fold_idx}.png"
            plot_ticker(
                cap_y_test,
                cap_y_pred,
                last_mape_by_secid,
                score,
                cfg.ticker,
                save_path=save_path,
            )

    result = {
        "scores": scores,
        "mean_score": float(np.mean(scores)) if scores else np.nan,
        "std_score": float(np.std(scores)) if scores else np.nan,
        "last_mape_by_secid": last_mape_by_secid,
        "forecaster": forecaster,
    }

    if artifacts_dir is not None:
        if save_metrics:
            save_run_artifacts(result, cfg, artifacts_dir)
        if save_model:
            save_forecaster(forecaster, artifacts_dir)

    logger.info(
        "Forecast CV end: folds={folds} mean={mean} std={std}",
        folds=len(scores),
        mean=result["mean_score"],
        std=result["std_score"],
    )
    return result
