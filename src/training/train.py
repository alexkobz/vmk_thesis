import mlflow
import pandas as pd
import numpy as np

from sktime.forecasting.base import ForecastingHorizon
from src import prepare_xy, restore_cap
from src.forecast.cv import ExpandingWindowYearSplitter, split_fold
from src.forecast.evaluate import wmape_score, wf1_score, ic_score
from src.training import flavor
from src.training.run_cfg import ForecastConfig


def main(cfg: ForecastConfig):
    """
    Function to train and evaluate a ngboost model
    :param cfg:
    :return:
    """
    mlflow.set_tracking_uri("http://localhost:5005/")
    mlflow.set_experiment("ngboost_base_experiment")

    if mlflow.active_run() is not None:
        mlflow.end_run()

    with mlflow.start_run(run_name=cfg.run_name) as run:
        print("tracking:", mlflow.get_tracking_uri())
        print("run_id:", run.info.run_id)

        df = cfg.data_reader()
        y, X, cap = prepare_xy(df)
        cap0 = cap.groupby(level="secid").first()

        model = cfg.model
        parameters = model.get_params()
        mlflow.log_params(parameters)

        fold_metrics = []
        splitter = ExpandingWindowYearSplitter(min_train_years=4)

        for fold, (train_idx, test_idx) in enumerate(splitter.split(y), 1):
            y_train, y_test, X_train, X_test = split_fold(y, X, train_idx, test_idx)

            print(f"Fold {fold}: Y Train shape: {y_train.shape}, Test shape: {y_test.shape}")
            print(f"Fold {fold}: X Train shape: {X_train.shape}, Test shape: {X_test.shape}")

            model.fit(y_train, X=X_train)

            target_dates = X_test.index.get_level_values("tradedate").unique().sort_values()
            fh = ForecastingHorizon(target_dates, is_relative=False)
            y_pred = model.predict(fh=fh, X=X_test)

            y_test = pd.Series(np.asarray(y_test).reshape(-1), index=X_test.index)
            y_pred = pd.Series(np.asarray(y_pred).reshape(-1), index=X_test.index)
            y_test_cap = restore_cap(y_test, cap0)
            y_pred_cap = restore_cap(y_pred, cap0)

            fold_wmape = wmape_score(y_test_cap, y_pred_cap)
            fold_wf1 = wf1_score(y_test, y_pred)
            fold_ic = ic_score(y_test, y_pred)

            fold_metrics.append({"fold": fold, "wmape": fold_wmape, "wf1": fold_wf1, "ic": fold_ic})

            mlflow.log_metric("wmape", fold_wmape, step=fold)
            mlflow.log_metric("wf1", fold_wf1, step=fold)
            mlflow.log_metric("ic", fold_ic, step=fold)

        mlflow.log_metric("wmape_mean", np.mean([m["wmape"] for m in fold_metrics]))
        mlflow.log_metric("wf1_mean", np.mean([m["wf1"] for m in fold_metrics]))
        mlflow.log_metric("ic_mean", np.mean([m["ic"] for m in fold_metrics]))
        flavor.log_model(
            sktime_model=model,
            artifact_path="training",
            serialization_format="pickle",
        )

    return model

if __name__ == "__main__":
    cfg = ForecastConfig()
    main(cfg)
