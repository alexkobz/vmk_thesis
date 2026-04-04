import numpy as np
import pandas as pd

from src.utils import restore_cap


def filter_boards(df: pd.DataFrame, boards: set[str]) -> pd.DataFrame:
    return df[df["boardid"].isin(boards)].copy()


def drop_additional_issues(df: pd.DataFrame, y_name: str) -> pd.DataFrame:
    df = df.copy()
    df["base_secid"] = df["secid"].str.replace(r"-0.*$", "", regex=True)

    agg = (
        df.groupby(["base_secid", "tradedate", "boardid"])[y_name]
        .agg(sumcap="sum", maxcap="max")
        .reset_index()
        .sort_values(["base_secid", "tradedate", "boardid"])
    )
    agg["is_issue"] = (agg["sumcap"] != agg["maxcap"]).astype(int)
    agg["issue_cummax"] = agg.groupby("base_secid")["is_issue"].cummax()
    agg["issue_cumsum"] = agg.groupby("base_secid")["is_issue"].cumsum()

    out = agg.merge(df, on=["base_secid", "tradedate", "boardid"], how="left")
    out[y_name] = out["sumcap"]
    return out.copy()


def filter_null_cols(df: pd.DataFrame, mults: list[str], lines: list[str]) -> pd.DataFrame:
    cols = mults + lines
    null_tickers = (
        df.groupby("secid")[cols]
        .apply(lambda g: g.isna().all().all())
        .pipe(lambda s: s[s].index.tolist())
    )
    return df[~df["secid"].isin(null_tickers)].copy()


def categorize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["type"] = df["type"].fillna("common_share").astype("category")
    df["sector"] = df["sector"].fillna("other").astype("category")
    return df


def filter_types(df: pd.DataFrame, types: set[str]) -> pd.DataFrame:
    return df[df["type"].isin(types)].copy()


def replace_zeros_with_nan(df: pd.DataFrame) -> pd.DataFrame:
    return df.replace(0, np.nan).copy()


def gather_secids(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["inn"] = df["inn"].str.zfill(10)

    def canonical_secid(group: pd.DataFrame) -> str:
        min_len = group["secid"].str.len().min()
        return group[group["secid"].str.len() == min_len]["secid"].iloc[0]

    df_common = df[df["type"] == "common_share"].copy()
    df_preferred = df[df["type"] == "preferred_share"].copy()

    common_map = df_common.groupby("inn").apply(canonical_secid).to_dict()
    preferred_map = df_preferred.groupby("inn").apply(canonical_secid).to_dict()

    df_common["secid"] = df_common["inn"].map(common_map)
    df_preferred["secid"] = df_preferred["inn"].map(preferred_map)

    return pd.concat([df_common, df_preferred], ignore_index=True).copy()


def set_index(df: pd.DataFrame, index: list[str], y_name: str) -> pd.DataFrame:
    return (
        df.sort_values(
            ["secid", "tradedate", "volume", y_name],
            ascending=[True, True, False, False],
        )
        .groupby(index)
        .first()
        .copy()
    )


def fill_days(df: pd.DataFrame) -> pd.DataFrame:
    df = df.groupby(level="secid").apply(lambda x: x.droplevel(0).asfreq("D"))
    df["is_vacation"] = df["year"].isna()
    return df


def ffill_bfill(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby(level="secid").ffill().groupby(level="secid").bfill()


def target_imputer(df: pd.DataFrame, y_name: str) -> pd.DataFrame:
    df = df.copy()
    outstanding_shares = df[y_name] / df["close"]
    mask = df[y_name].isna() & df["close"].notna() & outstanding_shares.notna()
    df.loc[mask, y_name] = df.loc[mask, "close"] * outstanding_shares
    df[y_name] = df[y_name].fillna(df[y_name].median())
    return df


def add_log_returns(df: pd.DataFrame, col: str, lags: list[int]) -> pd.DataFrame:
    df = df.copy()
    s_pos = df[col].astype(float).where(df[col] > 0)

    for lag in lags:
        col_name = f"log_returns_{col}_{lag}"
        df[col_name] = (
            s_pos.groupby(level="secid")
            .apply(lambda x: np.log(x).diff(lag))
            .reset_index(level=0, drop=True)
        )
    return df


def replace_target(df: pd.DataFrame, y_name: str) -> pd.DataFrame:
    s = df.groupby(level="secid")[f"log_returns_{y_name}_1"].sum()
    secids_zero = s[s == 0].index

    mask = df.index.get_level_values("secid").isin(secids_zero)
    df.loc[mask, f"log_returns_{y_name}_1"] = df.loc[mask, "log_returns_close_1"]

    cap0 = (
        df.loc[mask, y_name]
        .groupby(level="secid")
        .first()
        .transform(lambda x: x.fillna(x.mean()))
    )
    df.loc[mask, y_name] = restore_cap(df.loc[mask, f"log_returns_{y_name}_1"], cap0)
    return df


def filter_years(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["year"].notna()]


def filter_zero_target(df: pd.DataFrame, y_name: str) -> pd.DataFrame:
    s = df.groupby(level="secid")[f"log_returns_{y_name}_1"].apply(lambda x: np.abs(x).sum())
    secids_keep = s[s >= 1e-9].index
    return df.loc[pd.IndexSlice[secids_keep, :]]
