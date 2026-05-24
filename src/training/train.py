import mlflow
import numpy as np
import pandas as pd
from sktime.forecasting.compose import YfromX

from logs.logger import logger
from src.training import flavor
from src.training.evaluate import ic_score, wf1_score, wmape_score
from src.training.ModelConfig import ModelConfig, model_configs
from src.utils import (
    ProcessedCsvReader,
    apply_cli_overrides,
    build_train_parser,
    restore_cap,
    secid,
    tr,
    y_name,
)
from src.utils.training import (
    ExpandingWindowYearSplitter,
    prepare_yfromx_fold,
    align_y,
)

METRIC_FNS = {
    "wmape": wmape_score,
    "wf1": wf1_score,
    "ic": ic_score,
}


def train(cfg: ModelConfig) -> YfromX:
    """
    Function to train and evaluate a ngboost model
    :param cfg:
        ModelConfig
    :return:
        YfromX
    """

    mlflow.set_tracking_uri(tr['tracking_uri'])
    mlflow.set_experiment(f"{cfg.run_name}_experiment")

    if mlflow.active_run() is not None:
        mlflow.end_run()

    with mlflow.start_run(run_name=cfg.run_name) as run:
        logger.info(f"tracking_uri: {mlflow.get_tracking_uri()}")
        logger.info(f"run_id: {run.info.run_id}")

        df = ProcessedCsvReader().read()
        y = df[cfg.y].fillna(0).to_frame()
        X = df[cfg.X].fillna(0)
        cap = df[y_name]
        cap0 = cap.groupby(level=secid).first()

        model = YfromX(
            estimator=cfg.estimator(
                Base=cfg.Base,
                Dist=cfg.Dist,
                n_estimators=cfg.n_estimators,
                learning_rate=cfg.learning_rate,
                random_state=cfg.random_state,
                verbose=cfg.verbose,
            ),
            pooling=cfg.pooling,
        )
        parameters = model.get_params()
        mlflow.log_params(parameters)

        splitter = ExpandingWindowYearSplitter(min_train_years=cfg.min_train_years)
        metrics: list[dict[str, float]] = []

        for fold, (train_idx, test_idx) in enumerate(splitter.split(y), 1):
            y_train, y_test, X_train, X_pred, fh = prepare_yfromx_fold(
                y,
                X,
                train_idx,
                test_idx,
                secid_level="secid",
                date_level="tradedate",
                fill_value=0.0,
            )

            logger.info(f"Fold {fold}: Y Train shape: {y_train.shape}, Test shape: {y_test.shape}")
            logger.info(f"Fold {fold}: X Train shape: {X_train.shape}, Pred shape: {X_pred.shape}")

            model.fit(y_train, X=X_train)
            y_pred = model.predict(fh=fh, X=X_pred)
            
            y_test, y_pred = align_y(y_test, y_pred)
            # y_pred_clipped = y_pred.clip(-0.05, 0.05)
            y_test_cap = restore_cap(y_test, cap0)
            y_pred_cap = restore_cap(y_pred, cap0)
            # y_pred_cap = y_pred_cap.groupby(level="secid").transform(
            #     lambda s: s.clip(s.quantile(0.01), s.quantile(0.99))
            # )

            fold_metrics = {}
            for m in tr["evaluate"]["metrics"]:
                name = m["name"]
                fn = METRIC_FNS[m["function"]]

                if m.get("use_cap", False):
                    y_true, y_pred = y_test_cap, y_pred_cap
                else:
                    y_true, y_pred = y_test, y_pred

                value = fn(y_true, y_pred)
                fold_metrics[name] = value
                mlflow.log_metric(name, value, step=fold)

            metrics.append(fold_metrics)

        for m in tr["evaluate"]["metrics"]:
            name = m["name"]
            vals = [r[name] for r in metrics if name in r]
            if vals:
                mlflow.log_metric(f"{name}_mean", float(np.mean(vals)))
        flavor.log_model(
            sktime_model=model,
            artifact_path="training",
            serialization_format="pickle",
        )

    logger.info(f"Finished {cfg.run_name}")
    return model


if __name__ == "__main__":
    args = build_train_parser().parse_args()
    try:
        cfg = apply_cli_overrides(model_configs[args.model_config], args)
        train(cfg)
    except KeyError:
        logger.info('Invalid model configuration. Please choose from: %s', sorted(model_configs))
