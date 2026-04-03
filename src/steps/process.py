# Читаемая последовательность очистки и импутации
import numpy as np


def replace_zeros_with_nan(df):
    return df.replace(0, np.nan)

def fill_days(df):
    return (
        df
        .groupby(level="secid")
        .apply(lambda x: x.droplevel(0).asfreq("D"))
    ).ffill()

def impute_col(df, col):
    df[col] = df.groupby('secid')[col].ffill()
    df[col] = df.groupby('secid')[col].bfill()
    return df

def filter_zero_target(df, y_name):
    return df[df[y_name] > 1e-6]
