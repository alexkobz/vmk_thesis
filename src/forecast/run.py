from __future__ import annotations

import json
import os
from dataclasses import asdict
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

from config import config
from src.forecast.artifacts import serialize_model
from src.forecast.evaluate import panel_mape, panel_f1
from src.forecast.config import ForecastConfig, EstimatorType
from src.forecast.cv import (
    align_test_indices,
    build_cv,
    build_relative_fh,
    drop_cap_col,
    filter_secids,
    get_initial_cap,
    split_fold,
    split_multiindex_by_date,
)
from src.forecast.model import build_forecaster
from src.forecast.plotting import plot_ticker
from src.utils import restore_cap


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

def _get_folder_path(estimator_base_name: str) -> Path:
    now_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    folder_name = f"{estimator_base_name}_{now_str}"
    folder_path = config.ARTIFACTS_DIR / folder_name
    return folder_path

def _replace_outliers_with_prev(series, factor=3):
    """
    Replace outliers in a Series based on mean ± factor * std
    with the previous value in the Series.
    """
    mean = series.mean()
    std = series.std()
    lower = mean - factor * std
    upper = mean + factor * std
    outliers = ~series.between(lower, upper)
    # Replace outliers with previous value
    series[outliers] = series.where(~outliers).ffill()
    return series

def prepare_xy(
    df: pd.DataFrame,
    y_name: str = "log_returns_dailycapitalization_1",
    drop_cols: tuple[str, ...] = (),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    drop = [y_name, *drop_cols]
    y = df[y_name].to_frame()
    X = df.drop(columns=drop)
    return y, X


def run_expanding_cv(
    y: pd.DataFrame,
    X: pd.DataFrame,
    cfg: ForecastConfig,
):
    logger.info(
        "Forecast CV start: estimator={estimator} pooling={pooling} ticker={ticker}",
        estimator=type(cfg.estimator).__name__ if cfg.estimator is not None else "default",
        pooling=cfg.pooling,
        ticker=cfg.ticker,
    )
    forecaster = build_forecaster(estimator=cfg.estimator, pooling=cfg.pooling)
    cv = build_cv(cfg)
    folder_path = _get_folder_path(cfg.estimator.Base.__class__.__name__)
    logger.add(
        folder_path / 'run.log',
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level="INFO",
    )

    for fold_idx, (train_idx, test_idx) in enumerate(split_multiindex_by_date(y, cv), start=1):
        logger.info("Fold {fold}: split sizes train={train} test={test}", fold=fold_idx, train=len(train_idx), test=len(test_idx))
        y_train, y_test, X_train, X_test = split_fold(y, X, train_idx, test_idx)
        y_test, X_test = align_test_indices(y_test, X_test)
        if y_test.empty or X_test.empty:
            logger.info("Fold {fold}: skipped (empty test after alignment)", fold=fold_idx)
            continue
        X_train_m, X_test_m = drop_cap_col(X_train, X_test)

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
        fh_rel = build_relative_fh(test_dates)

        logger.info("Fold {fold}: fitting forecaster", fold=fold_idx)
        forecaster.fit(y_train, X_train_m)

        logger.info("Fold {fold}: predicting", fold=fold_idx)
        y_pred = forecaster.predict(fh=fh_rel, X=X_test_m)
        if cfg.estimator_type == EstimatorType.TREE:
            cap0 = get_initial_cap(X_test, secids_keep)
            cap_y_test = restore_cap(y_test.iloc[:, 0], cap0)
            cap_y_pred = restore_cap(y_pred.iloc[:, 0], cap0)
            cap_y_pred = cap_y_pred.groupby(level=0, group_keys=False).apply(_replace_outliers_with_prev)
        elif cfg.estimator_type == EstimatorType.LINEAR:
            cap_y_test = y_test.iloc[:, 0]
            cap_y_pred = y_pred.iloc[:, 0]
        else:
            raise NotImplementedError

        if cfg.save_metrics:
            os.makedirs(folder_path, exist_ok=True)
            fold_path = folder_path / f'fold{fold_idx}'
            os.makedirs(fold_path, exist_ok=True)

            cfg_dict = asdict(cfg)
            cfg_dict["estimator"] = _serialize_estimator(cfg.estimator)
            mapes = panel_mape(cap_y_test, cap_y_pred, cap_y_test)
            f1s = panel_f1(y_test.iloc[:, 0], y_pred, cap_y_test)

            metrics = {
                'config': cfg_dict,
                'X': X_test_m.columns.tolist(),
                **mapes,
                **f1s,
            }
            metric_path = fold_path / 'metrics.json'
            metric_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2, default=str))

            logger.info(
                "Fold {fold}: wmape={wmape_score}, f1={f1_score}",
                fold=fold_idx,
                wmape_score=metrics['wmape'],
                f1_score=metrics['wf1'],
            )
            if cfg.ticker:
                plot_ticker(
                    cap_y_test,
                    cap_y_pred,
                    ticker=cfg.ticker,
                    mape=metrics['mape'][cfg.ticker],
                    f1=metrics['f1'][cfg.ticker],
                    save_path=fold_path,
                )

    if cfg.save_model:
        serialize_model(forecaster, folder_path)

    return forecaster
