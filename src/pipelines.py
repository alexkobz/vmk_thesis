from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (FunctionTransformer, MinMaxScaler,
                                   OneHotEncoder, StandardScaler)

from src.data_processing import (categorize, drop_additional_issues,
                                 ffill_bfill, fill_periods, filter_boards,
                                 filter_null_cols, filter_secids, filter_years,
                                 gather_secids, replace_zeros_with_nan,
                                 set_index)
from src.feature_engineering import add_log_returns, lowess_smooth
from src.utils import dp, mults, lines, index_cols, fe


def build_data_processing_pipeline() -> Pipeline:

    return Pipeline([
        ("filter_boards", FunctionTransformer(
            filter_boards,
            kw_args={
                "boards": dp['filter_boards']['boards']
            })
        ),
        ("drop_additional_issues", FunctionTransformer(
            drop_additional_issues,
            kw_args={
                "by": dp['drop_additional_issues']['by'],
                "template": dp['drop_additional_issues']['template']
            })
        ),
        ("filter_null_cols", FunctionTransformer(
            filter_null_cols,
            kw_args={
                "cols": mults + lines
            })
        ),
        ("categorize", FunctionTransformer(
            categorize,
            kw_args={
                "cols": dp['categorize']['cols'],
            })
        ),
        ("gather_secids", FunctionTransformer(
            gather_secids,
            kw_args={
            })
        ),
        ("replace_zeros_with_nan", FunctionTransformer(
            replace_zeros_with_nan,
            kw_args={
            })
        ),
        ("set_index", FunctionTransformer(
            set_index,
            kw_args={
                "index_cols": index_cols,
                "sort_by": dp['set_index']['sort_by'],
            })
        ),
        ("fill_periods", FunctionTransformer(
            fill_periods,
            kw_args={
                "freq": dp['fill_periods']['freq'],
            })
        ),
        ("ffill_bfill", FunctionTransformer(
            ffill_bfill,
            kw_args={
            })
        ),
        ("filter_years", FunctionTransformer(
            filter_years,
            kw_args={
                "years": dp['filter_years']['years'],
            })
        ),
        ("filter_secids", FunctionTransformer(
            filter_secids,
            kw_args={
                "num": dp['filter_secids']['num'],
            })
        ),
    ])


def build_feature_engineering_pipeline() -> Pipeline:

    scale_transformer = ColumnTransformer([
        ("cat", OneHotEncoder(sparse_output=False), fe["one_hot_encode"]["cols"]),
        ("mm", MinMaxScaler(), fe["min_max_scale"]["cols"]),
        ("st", StandardScaler(), fe["standard_scale"]["cols"]),
    ],
        remainder="passthrough",
        # переименовывание обрабатываемых колонок, оставляем непереименованными неиспользуемые колонки
        verbose_feature_names_out=lambda transformer_name, feature_name: (
            feature_name
            if transformer_name == "remainder"
            else f"{transformer_name}__{feature_name}"
        ),
    )
    scale_transformer.set_output(transform="pandas")

    feature_engineering_pipeline = Pipeline([
        ("lowess_smooth", FunctionTransformer(
            lowess_smooth,
            kw_args={
                "cols": mults + lines,
                "col_prefix": fe['lowess_smooth']['col_prefix'],
                "frac": fe['lowess_smooth']['frac'],
            })
        ),
        ("add_log_returns", FunctionTransformer(
            add_log_returns,
            kw_args={
                "features": fe['add_log_returns']['features'],
                "col_prefix": fe['add_log_returns']['col_prefix'],
            })
         ),
        ("scale_transformer", scale_transformer),
     ])
    return feature_engineering_pipeline
