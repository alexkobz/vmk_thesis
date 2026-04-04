from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, MinMaxScaler, OneHotEncoder, StandardScaler

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


DEFAULT_BOARDS = {"TQBR", "EQBR", "TQBS", "EQBS", "TQLV", "EQLV", "TQNL", "EQNL"}
DEFAULT_TYPES = {"common_share", "preferred_share"}


@dataclass(frozen=True)
class PipelineConfig:
    y_name: str
    index: list[str]
    mults: list[str]
    lines: list[str]
    cat_cols: list[str]
    boards: set[str] = field(default_factory=lambda: set(DEFAULT_BOARDS))
    types: set[str] = field(default_factory=lambda: set(DEFAULT_TYPES))


def _add_lr_for_col(df: pd.DataFrame, col: str) -> pd.DataFrame:
    out = add_log_returns(df, col, [1])
    return out[[f"log_returns_{col}_1"]]


def _scale_features(df: pd.DataFrame, cat_cols: list[str]) -> pd.DataFrame:
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
    return df


def build_prep_pipeline(cfg: PipelineConfig) -> Pipeline:
    return Pipeline(
        [
            ("filter_boards", FunctionTransformer(filter_boards, kw_args={"boards": cfg.boards}, validate=False)),
            ("drop_additional_issues", FunctionTransformer(drop_additional_issues, kw_args={"y_name": cfg.y_name}, validate=False)),
            ("categorize", FunctionTransformer(categorize, validate=False)),
            ("filter_types", FunctionTransformer(filter_types, kw_args={"types": cfg.types}, validate=False)),
            ("replace_zeros_with_nan", FunctionTransformer(replace_zeros_with_nan, validate=False)),
            ("gather_secids", FunctionTransformer(gather_secids, validate=False)),
            ("set_index", FunctionTransformer(set_index, kw_args={"index": cfg.index, "y_name": cfg.y_name}, validate=False)),
            ("fill_days", FunctionTransformer(fill_days, validate=False)),
            ("ffill_bfill", FunctionTransformer(ffill_bfill, validate=False)),
            ("add_log_returns_cap1", FunctionTransformer(add_log_returns, kw_args={"col": cfg.y_name, "lags": [1]}, validate=False)),
            ("add_log_returns_close", FunctionTransformer(add_log_returns, kw_args={"col": "close", "lags": [1]}, validate=False)),
            ("replace_target", FunctionTransformer(replace_target, kw_args={"y_name": cfg.y_name}, validate=False)),
            ("add_log_returns_cap5", FunctionTransformer(add_log_returns, kw_args={"col": cfg.y_name, "lags": [5]}, validate=False)),
            ("filter_years", FunctionTransformer(filter_years, validate=False)),
            ("filter_zero_target", FunctionTransformer(filter_zero_target, kw_args={"y_name": cfg.y_name}, validate=False)),
        ]
    )


def build_full_pipeline(cfg: PipelineConfig) -> Pipeline:
    cols_for_smooth = [f"log_returns_{cfg.y_name}_1"] + cfg.mults + cfg.lines
    smoothed_cols = [f"smoothed_{c}" for c in (cfg.mults + cfg.lines)]

    smooth_ct = ColumnTransformer(
        [(f"smooth_{col}", FunctionTransformer(lowess_smooth, validate=False), [col]) for col in cols_for_smooth],
        remainder="passthrough",
    )

    add_smoothed_lr_ct = ColumnTransformer(
        [
            (f"lr_{col}", FunctionTransformer(_add_lr_for_col, kw_args={"col": col}, validate=False), [col])
            for col in smoothed_cols
        ],
        remainder="passthrough",
    )

    return (
        Pipeline(
            [
                ("prep", build_prep_pipeline(cfg)),
                ("smooth", smooth_ct),
                ("add_smoothed_lr", add_smoothed_lr_ct),
                ("scale", FunctionTransformer(_scale_features, kw_args={"cat_cols": cfg.cat_cols}, validate=False)),
            ]
        )
        .set_output(transform="pandas")
    )
