from __future__ import annotations

import pandas as pd
from loguru import logger
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, MinMaxScaler, OneHotEncoder, StandardScaler

from config.columns import boards, cat_cols, index, lines, mults, types, y_name
from src.steps.process import (
    add_log_returns,
    categorize,
    drop_additional_issues,
    ffill_bfill,
    filter_boards,
    filter_types,
    filter_years,
    filter_zero_target,
    gather_secids,
    replace_target,
    replace_zeros_with_nan,
    set_index,
    fill_days,
)
from src.steps.smoothing import lowess_smooth

def _describe_shape(obj: object) -> str:
    try:
        shape = obj.shape
    except AttributeError:
        return "shape=unknown"
    return f"shape={shape}"


def _wrap_step(step_name: str, fn):
    def _wrapped(df, *args, **kwargs):
        logger.info("Step {step}: start ({shape})", step=step_name, shape=_describe_shape(df))
        out = fn(df, *args, **kwargs)
        logger.info("Step {step}: end ({shape})", step=step_name, shape=_describe_shape(out))
        return out

    return _wrapped


def _add_lr_for_col(df: pd.DataFrame, col: str) -> pd.DataFrame:
    out = add_log_returns(df, col, [1])
    return out[[f"log_returns_{col}_1"]]


def _scale_features(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Step scale: start ({shape})", shape=_describe_shape(df))
    smoothed_cols = [c for c in df.columns if c.startswith("smoothed_")]
    desc = df[smoothed_cols].describe(include="all")

    cols_minmax = desc.loc["min"][desc.loc["min"] > -1e-6].index.tolist()
    cols_standard = desc.loc["min"][desc.loc["min"] < -1e-6].index.tolist()

    preprocess = ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ("mm", MinMaxScaler(), cols_minmax),
            ("st", StandardScaler(), cols_standard),
        ],
        remainder="passthrough",
    )
    df_sc = preprocess.fit_transform(df)
    df[preprocess.get_feature_names_out()] = df_sc
    logger.info("Step scale: end ({shape})", shape=_describe_shape(df))
    return df


def build_prep_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("filter_boards", FunctionTransformer(_wrap_step("filter_boards", filter_boards), kw_args={"boards": boards}, validate=False)),
            ("drop_additional_issues", FunctionTransformer(_wrap_step("drop_additional_issues", drop_additional_issues), kw_args={"y_name": y_name}, validate=False)),
            ("categorize", FunctionTransformer(_wrap_step("categorize", categorize), validate=False)),
            ("filter_types", FunctionTransformer(_wrap_step("filter_types", filter_types), kw_args={"types": types}, validate=False)),
            ("replace_zeros_with_nan", FunctionTransformer(_wrap_step("replace_zeros_with_nan", replace_zeros_with_nan), validate=False)),
            ("gather_secids", FunctionTransformer(_wrap_step("gather_secids", gather_secids), validate=False)),
            ("set_index", FunctionTransformer(_wrap_step("set_index", set_index), kw_args={"index": index, "y_name": y_name}, validate=False)),
            ("fill_days", FunctionTransformer(_wrap_step("fill_days", fill_days), validate=False)),
            ("ffill_bfill", FunctionTransformer(_wrap_step("ffill_bfill", ffill_bfill), validate=False)),
            ("add_log_returns_cap1", FunctionTransformer(_wrap_step("add_log_returns_cap1", add_log_returns), kw_args={"col": y_name, "lags": [1]}, validate=False)),
            ("add_log_returns_close", FunctionTransformer(_wrap_step("add_log_returns_close", add_log_returns), kw_args={"col": "close", "lags": [1]}, validate=False)),
            ("replace_target", FunctionTransformer(_wrap_step("replace_target", replace_target), kw_args={"y_name": y_name}, validate=False)),
            ("add_log_returns_cap5", FunctionTransformer(_wrap_step("add_log_returns_cap5", add_log_returns), kw_args={"col": y_name, "lags": [5]}, validate=False)),
            ("filter_years", FunctionTransformer(_wrap_step("filter_years", filter_years), validate=False)),
            ("filter_zero_target", FunctionTransformer(_wrap_step("filter_zero_target", filter_zero_target), kw_args={"y_name": y_name}, validate=False)),
        ]
    )


def build_full_pipeline() -> Pipeline:
    cols_for_smooth = [f"log_returns_{y_name}_1"] + mults + lines
    smoothed_cols = [f"smoothed_{c}" for c in (mults + lines)]

    smooth_ct = ColumnTransformer(
        [
            (
                f"smooth_{col}",
                FunctionTransformer(_wrap_step(f"smooth_{col}", lowess_smooth), validate=False),
                [col],
            )
            for col in cols_for_smooth
        ],
        remainder="passthrough",
    )

    add_smoothed_lr_ct = ColumnTransformer(
        [
            (
                f"lr_{col}",
                FunctionTransformer(_wrap_step(f"add_smoothed_lr_{col}", _add_lr_for_col), kw_args={"col": col}, validate=False),
                [col],
            )
            for col in smoothed_cols
        ],
        remainder="passthrough",
    )

    return (
        Pipeline(
            [
                ("prep", build_prep_pipeline()),
                ("smooth", smooth_ct),
                ("add_smoothed_lr", add_smoothed_lr_ct),
                ("scale", FunctionTransformer(_scale_features, validate=False)),
            ]
        )
        .set_output(transform="pandas")
    )
