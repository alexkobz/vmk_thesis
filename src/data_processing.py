import numpy as np
import pandas as pd

from utils.read_yaml import *
from utils.utils import show_shape


@show_shape
def filter_boards(df: pd.DataFrame, boards: list[str]) -> pd.DataFrame:
    return df[df[boardid].isin(boards)]


@show_shape
def drop_additional_issues(df: pd.DataFrame, by: str | list[str], template: str) -> pd.DataFrame:
    df["base_secid"] = df[secid].str.replace(rf"{template}", "", regex=True)
    agg = (
        df.groupby(by)[y_name]
        .agg(sumcap="sum", maxcap="max")
        .reset_index()
        .sort_values(by)
    )
    agg["is_issue"] = (agg["sumcap"] != agg["maxcap"]).astype(int)
    agg["issue_cummax"] = agg.groupby("base_secid")["is_issue"].cummax()
    agg[issue_cumsum] = agg.groupby("base_secid")["is_issue"].cumsum()

    out = agg.merge(df, on=by, how="left")
    out[y_name] = out["sumcap"]
    return out


@show_shape
def filter_null_cols(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    null_tickers = (
        df.groupby(secid)[cols]
        .apply(lambda g: g.isna().all().all())
        .pipe(lambda s: s[s].index.tolist())
    )
    return df[~df[secid].isin(null_tickers)]


@show_shape
def categorize(df: pd.DataFrame, cols: dict) -> pd.DataFrame:
    for col, val in cols.items():
        df[col] = df[col].fillna(val).astype("category")
    return df


@show_shape
def gather_secids(df: pd.DataFrame) -> pd.DataFrame:
    df[inn] = df[inn].str.zfill(10)

    def canonical_secid(group: pd.DataFrame) -> str:
        min_len = group[secid].str.len().min()
        return group[group[secid].str.len() == min_len][secid].iloc[0]

    df_common = df[df[share_type] == 'common_share']
    df_preferred = df[df[share_type] == 'preferred_share']

    common_map = df_common.groupby(inn).apply(canonical_secid).to_dict()
    preferred_map = df_preferred.groupby(inn).apply(canonical_secid).to_dict()

    df_common[secid] = df_common[inn].map(common_map)
    df_preferred[secid] = df_preferred[inn].map(preferred_map)

    return pd.concat([df_common, df_preferred], ignore_index=True)


@show_shape
def replace_zeros_with_nan(df: pd.DataFrame) -> pd.DataFrame:
    return df.replace(0, np.nan)


@show_shape
def set_index(df: pd.DataFrame, index_cols: str | list[str], sort_by: dict[str, bool]) -> pd.DataFrame:
    return (
        df.sort_values(
            list(sort_by.keys()),
            ascending=list(sort_by.values()),
        )
        .groupby(index_cols)
        .first()
    )


@show_shape
def fill_periods(df: pd.DataFrame, freq = 'D') -> pd.DataFrame:
    df = df.groupby(level=secid).apply(lambda x: x.droplevel(0).asfreq(freq))
    df[is_vacation] = df[year].isna()
    return df


@show_shape
def ffill_bfill(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby(level=secid).ffill().groupby(level=secid).bfill()


@show_shape
def filter_years(df: pd.DataFrame, years: list[int]) -> pd.DataFrame:
    return df[df[year].isin(years)]


@show_shape
def filter_secids(df: pd.DataFrame, num: int) -> pd.DataFrame:
    secid_counts = df.index.get_level_values(0).value_counts()
    df = df[df.index.get_level_values(0).isin(
        secid_counts[secid_counts > num].index
    )]
    return df


# def target_imputer(df: pd.DataFrame) -> pd.DataFrame:
#     outstanding_shares = df[y_name] / df[close]
#     mask = df[y_name].isna() & df[close].notna() & outstanding_shares.notna()
#     df.loc[mask, y_name] = df.loc[mask, close] * outstanding_shares
#     df[y_name] = df[y_name].fillna(df[y_name].median())
#     return df

# def filter_zero_target(df: pd.DataFrame) -> pd.DataFrame:
#     s = df.groupby(level=secid)[y_log].apply(lambda x: np.abs(x).sum())
#     secids_keep = s[s >= 1e-9].index
#     return df.loc[pd.IndexSlice[secids_keep, :]]
